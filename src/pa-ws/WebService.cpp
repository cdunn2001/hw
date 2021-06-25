// Copyright (c) 2021, Pacific Biosciences of California, Inc.
//
// All rights reserved.
//
// THIS SOFTWARE CONSTITUTES AND EMBODIES PACIFIC BIOSCIENCES' CONFIDENTIAL
// AND PROPRIETARY INFORMATION.
//
// Disclosure, redistribution and use of this software is subject to the
// terms and conditions of the applicable written agreement(s) between you
// and Pacific Biosciences, where "you" refers to you or your company or
// organization, as applicable.  Any other disclosure, redistribution or
// use is prohibited.
//
// THIS SOFTWARE IS PROVIDED BY PACIFIC BIOSCIENCES AND ITS CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL PACIFIC BIOSCIENCES OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
// OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
// ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// File Description:
/// \brief  implementation of the GET, POST and DELETE HTTP endpoint handler
//          wx-daemon REST API.
//
// Programmer: Mark Lakata

#include "WebService.h"

// system includes
#include <chrono>
#include <cstdlib>
#include <arpa/inet.h>
#include <boost/regex.hpp>

// 3rd party includes
#include <json/json.h>
#include <boost/filesystem.hpp>
#include <boost/range/adaptor/map.hpp>

// pacbio-cplusplus-api includes
#include <pacbio/dev/AutoTimer.h>
#include <pacbio/POSIX.h>
#include <pacbio/ipc/JSON.h>
#include <pacbio/ipc/CurlPlusPlus.h>
#include <pacbio/process/ConfigurationBase.h>
#include <pacbio/process/CivetServer.h>
#include <pacbio/process/RESTServer.h>
#include <pacbio/process/ProcessBase.h>
#include <pacbio/text/String.h>
#include <pacbio/text/pugixml_pb.hpp>
#include <pacbio/utilities/ISO8601.h>

#include <dochelp.h>
#include <DashboardWrapper.h>

// application includes
//#include "config/pa-ws-config.h"
#include "version.h"
//#include "HttpHelpers.h"
#include <git-rev.h>

#include "api/SocketObject.h"
#include "api/PostprimaryObject.h"
#include "api/StorageObject.h"
#include "api/TransferObject.h"

#include "mockup.h"

#if 0
#include <WxDaemonHeader.h>
#include "Locked.h"
#include "SraObjectDatabase.h"
#endif

namespace PacBio {
namespace Primary {
namespace PaWs {

using namespace PacBio::Text;
using PacBio::IPC::HttpResponse;
using PacBio::IPC::HttpStatus;
using PacBio::IPC::HttpResponseException;
using namespace PacBio::API;

/// Convert ip address integer (host order) to a xxx.xxx.xxx.xxx IP address string
/// \param ip : host order 32-bit integer address
/// \returns "xxx.xxx.xxx.xxx" formatted string
std::string inet_ntos(long ip)
{
    char dst[ INET_ADDRSTRLEN];
    auto nip = htonl(ip);
    inet_ntop(AF_INET,&nip, dst,sizeof(dst));
    return std::string(dst);
}

WebService::WebService(const PaWsConfig& config,
        std::shared_ptr<PacBio::Threading::IThreadController> threadController
)
    : config_(config)
    , threadController_(threadController)
    , rootUrl_(PacBio::POSIX::gethostname() + ":" + std::to_string(config_.port))
{
    PBLOG_INFO << "WebService constructor. http://" << rootUrl_;
}

WebService::~WebService()
{
    PBLOG_INFO << "WebService destructor";
}

void WebService::InstallService(WebService::SraCallback_t sraCallback)
{
    // listen to two ports, typically 8090 and 8091, at the same time.
    std::string portName = std::to_string(config_.port);
    std::string num_threads = std::to_string(config_.numThreads);

    const char* options[] = {
            "listening_ports", portName.c_str(),
            "error_log_file", "/dev/stderr",
            "num_threads", num_threads.c_str(),
            nullptr
    };

    try
    {
        std::stringstream ss;
        for(const char** p = options; *p ; p++)
        {
            ss << *p << " ";
        }
        PBLOG_INFO << "Options to CivetServer: " << ss.str();

        server_ = std::make_shared<CivetServer>(options);
        if (!server_)
        {
            throw PBException("Could not start CivetServer (WebService) on "
                 + portName);
        }
    }
    catch (const std::exception& ex)
    {
        throw PBException(
            "Could not start WebService " + portName 
            + ". CivetServer constructor exception: " + ex.what());
    }

    auto handler = std::make_unique<WebServiceHandler>(
        config_,
        threadController_
        );
    handler->InstallSraCallback(sraCallback);

    handlers_.emplace_back(std::move(handler));
    server_->addHandler("/", handlers_.back().get());
}

////////////////////////////////////////////////////////////////////////////////

WebServiceHandler::WebServiceHandler(const PaWsConfig& config,
        std::shared_ptr<PacBio::Threading::IThreadController> threadController
)
    : config_(config)
    , threadController_(threadController)
    , processStartTime_(PacBio::Utilities::Time::GetMonotonicTime())
    , rootUrl_(PacBio::POSIX::gethostname() + ":" + std::to_string(config_.port))
{
    PBLOG_INFO << "WebServiceHandler constructor. Port:" << config_.port;
}

WebServiceHandler::~WebServiceHandler()
{
    PBLOG_INFO << "WebServiceHandler destructor";
}

/// Civet "OPTIONS" override. Handle preflight check. This mostly to support CORS.
bool WebServiceHandler::handleOptions(CivetServer* server, struct mg_connection* conn)
{
    (void) server;
    struct mg_request_info* req_info = mg_get_request_info(conn);
    std::string uri(req_info->uri);

    requestsOptions_.Increment(req_info->remote_ip);

    //http://stackoverflow.com/questions/10636611/how-does-access-control-allow-origin-header-work
    //https://developer.mozilla.org/en-US/docs/Web/HTTP/Access_control_CORS
    HttpStatus httpStatus = HttpStatus::OK;
    // "Access-Control-Allow-Origin: *" allows Javascript from one host to do GET requests on another host
    mg_printf(conn, "HTTP/1.1 %d %s\r\n"
                      "Allow: GET, POST, OPTIONS, DELETE\r\n"
                      "Access-Control-Allow-Methods: POST, GET, OPTIONS, DELETE\r\n"
                      "Access-Control-Allow-Headers: Origin, X-Requested-With, Content-Type, Accept\r\n"
                      "Access-Control-Allow-Origin: *\r\n\r\n",
              (int) httpStatus, httpStatus.toString().c_str());
//          std::cout << "RESTServer serving OPTION" << std::endl;
    return true;
}


/// Civet GET override
bool WebServiceHandler::handleGet(CivetServer* server, struct mg_connection* conn)
{
    PacBio::Dev::QuietAutoTimer autoTimer;

    struct mg_request_info* req_info = mg_get_request_info(conn);
    std::string uri(req_info->uri);

    requestsGet_.Increment(req_info->remote_ip);

    std::string contentType = "text";
    std::string respdata;
    HttpStatus httpStatus = HttpStatus::OK;

    SMART_ENUM(Format_t, Unknown, XML, JSON);

    // render the data model using the best format requested:
    Format_t format = Format_t::Unknown;

    // check extention first
    if (String::EndsWith(uri, ".xml"))
    {
        format = Format_t::XML;
        uri = uri.substr(0, uri.size() - 4);
    }
    else if (String::EndsWith(uri, ".json"))
    {
        format = Format_t::JSON;
        uri = uri.substr(0, uri.size() - 5);
    }

    // check Accept: header
    if (format == Format_t::Unknown)
    {
        const char* accept = server->getHeader(conn, "Accept");
        if (accept != nullptr)
        {
            if (!strcmp(accept, "application/json")) format = Format_t::JSON;
            else if (!strcmp(accept, "application/xml")) format = Format_t::XML;
            else if (!strcmp(accept, "text/xml")) format = Format_t::XML;
        }
    }
    // check Content-Type: header
    if (format == Format_t::Unknown)
    {
        const char* contentTypeRequest = server->getHeader(conn, "Content-Type");
        if (contentTypeRequest != nullptr)
        {
            if (!strcmp(contentTypeRequest, "application/json")) format = Format_t::JSON;
            else if (!strcmp(contentTypeRequest, "application/xml")) format = Format_t::XML;
            else if (!strcmp(contentTypeRequest, "text/xml")) format = Format_t::XML;
        }
    }

    PBLOG_DEBUG << "GET From: " << inet_ntos(req_info->remote_ip) << ":" << req_info->remote_port
                    << " : " << uri;

    std::vector<std::string> fields = String::Split(uri, '/');
    std::string firstField = (fields.size() >= 2) ? fields[1] : "";

    switch (format)
    {
    case Format_t::XML:
        respdata = "<?xml version=\"1.1\"?>" + std::string("not supported"); //+ x->RenderXML();
        contentType = "application/xml";
        break;
    case Format_t::JSON:
    default:
    {
        auto response = GET(uri.substr(1));
        httpStatus = response.httpStatus;
        contentType = response.contentType;
        if (httpStatus != HttpStatus::OK)
        {
            PBLOG_ERROR << "From: " << inet_ntos(req_info->remote_ip) << ":" << req_info->remote_port
                        << " " << response.json;
        }
        if (contentType == "application/json")
        {
            contentType = "application/json";
            if (httpStatus != HttpStatus::OK)
            {
                Json::Value repackagedJson;
                repackagedJson["original"] = response.json;
                repackagedJson["httpCode"] = static_cast<int>(httpStatus);
                repackagedJson["errorType"] = httpStatus.toString();
                response.json = repackagedJson;
                PBLOG_ERROR << response.json;
            }
            respdata = PacBio::IPC::RenderJSON(response.json);
        }
        else
        {
            respdata = response.json.asString();
            if (httpStatus != HttpStatus::OK)
            {
                respdata += "httpCode: " + httpStatus.toString();
                PBLOG_ERROR << respdata;
            }
        }
    }
        break;
    }

    const double responseGenerationTime = autoTimer.GetElapsedMilliseconds();
    Respond(conn, httpStatus, contentType.c_str(), respdata);
    const double responseTime = autoTimer.GetElapsedMilliseconds();
    if (config_.logHttpGets)
    {
        PBLOG_INFO << "http GET " << req_info->uri << " generate response:" << responseGenerationTime << " sec " <<
                    "net response:" << responseTime << "sec ";
    }

    return true;
}

/// Civet POST override
bool WebServiceHandler::handlePost(CivetServer* server, struct mg_connection* conn)
{
    struct mg_request_info* req_info = mg_get_request_info(conn);
    std::string uri(req_info->uri);

    requestsPost_.Increment(req_info->remote_ip);

    std::string postData;
    server->getPostData(conn, postData);
    if (!PacBio::Text::String::Contains(uri,"/ping"))
    {
        PBLOG_INFO << "POST From: " << inet_ntos(req_info->remote_ip) << ":" << req_info->remote_port
                << " : " << uri << " " << postData;
    }

    auto response = POST(uri.substr(1), postData);

    if (response.httpStatus != HttpStatus::OK && response.httpStatus != HttpStatus::CREATED)
    {
        PBLOG_ERROR << response.httpStatus.toString() << " body:" << response.json
                    << "\n During processing of POST " << uri << "\n" << postData
                    << "\n request from:" << inet_ntos(req_info->remote_ip) << ":" << req_info->remote_port << " ";
    }

    std::string responseText = PacBio::IPC::RenderJSON(response.json);
    const char* contentType = "application/json";
    Respond(conn, response.httpStatus, contentType, responseText, response.extraHeaders);
    return true;
}

/// Civet DELETE override
bool WebServiceHandler::handleDelete(CivetServer* server, struct mg_connection* conn)
{
    (void) server;
    struct mg_request_info* req_info = mg_get_request_info(conn);
    std::string uri(req_info->uri);

    requestsDelete_.Increment(req_info->remote_ip);

    PBLOG_INFO << "DELETE From: " << inet_ntos(req_info->remote_ip) << ":" << req_info->remote_port
                    << " : " << uri;
    auto response = DELETE(uri.substr(1));
    if (response.httpStatus != HttpStatus::OK)
    {
        PBLOG_ERROR << "During processing of DELETE /" << uri;
        PBLOG_ERROR << "From: " << inet_ntos(req_info->remote_ip) << ":" << req_info->remote_port << " "
                    << response.json;
    }

    std::string responseText = PacBio::IPC::RenderJSON(response.json);
    const char* contentType = "application/json";
    Respond(conn, response.httpStatus, contentType, responseText);
    return true;
}

bool WebServiceHandler::handlePut(CivetServer* /* server*/, struct mg_connection* conn)
{
    struct mg_request_info* req_info = mg_get_request_info(conn);
    std::string uri(req_info->uri);

    requestsPut_.Increment(req_info->remote_ip);

    return false;
}

/// General http formatted response function.
/// \param conn : mongoose connection
/// \param httpStatus : status of response
/// \param contentType : the content type of the response
/// \param respdata : the body of the http packet
void WebServiceHandler::Respond(struct mg_connection* conn, HttpStatus httpStatus, const char* contentType,
                          const std::string& respdata, const std::string& extraHeaders)
{
    // "Access-Control-Allow-Origin: *" allows Javascript from one host to do GET requests on another host
    mg_printf(conn, "HTTP/1.1 %d %s\r\n"
            "Content-Type: %s\r\n"
            "Access-Control-Allow-Methods: POST, GET, OPTIONS\r\n"
            "Access-Control-Allow-Headers: Origin, X-Requested-With, Content-Type, Accept\r\n"
            "Access-Control-Allow-Origin: *\r\n%s\r\n", (int) httpStatus, httpStatus.toString().c_str(), contentType,
            extraHeaders.c_str());
    mg_printf(conn, "%s", respdata.c_str());
}


double WebServiceHandler::GetUpTime() const
{
    double t = PacBio::Utilities::Time::GetMonotonicTime();
    return t - processStartTime_;
}

std::string WebServiceHandler::ConvertTimeSpanToString(double seconds)
{
    std::stringstream ss;
    ss << "Services have been up for " ;
    uint64_t days = seconds/(3600UL*24);
    seconds -= days * (3600UL*24);
    if (days >0) ss << days << " days, ";
    uint64_t hours = seconds/3600UL;
    seconds -= hours * 3600UL;
    if (days >0 || hours > 0) ss << hours << " hours, ";
    uint64_t minutes = seconds / 60;
    seconds -= minutes * 60000;
    if (minutes >0 || hours >0 || days > 0) ss << minutes << " minutes and ";
    ss << std::setprecision(3) << seconds << " seconds.";
    return ss.str();
}

/// The GET handler for pa-ws
/// \param uri :  the uri striped of protocol, host and port name. Should not start with a slash
HttpResponse WebServiceHandler::GET(const std::string& uri)
{
    HttpResponse response;
    response.httpStatus = HttpStatus::OK;

    try
    {
        auto args = String::Split(String::Chomp(uri,'/'), '/');

        PBLOG_DEBUG << "args:" << String::Join(args.begin(), args.end(), ',');

        if (args.size() >= 1)
        {
            if (args[0] == "sockets")
            {
                args.erase(args.begin());
                return GET_Sockets(args);
            }
            else if (args[0] == "postprimaries")
            {
                args.erase(args.begin());
                return GET_Postprimaries(args);
            }
            else if (args[0] == "storages")
            {
                args.erase(args.begin());
                return GET_Storages(args);
            }
            else if (args[0] == "transfers")
            {
                args.erase(args.begin());
                return GET_Transfer(args);
            }
            else if (args[0] == "api")
            {
                response.json = "here is the API";
            }
#if 0
            else if (args[0] == "sras")
            {
                args.erase(args.begin());
                if (args.size() == 0)
                {
                    Json::Value allSras = Json::arrayValue;
                    for (auto sraIndex : GetSraObjectDatabase().GetSraIndices() )
                    {
                        auto so = LockSra(sraIndex);
                        allSras.append(so->Json());
                    }
                    response.json = allSras;
                }
                else
                {
                    auto sraIndex = std::stoul(args[0]);
                    args.erase(args.begin());
                    if (!GetSraObjectDatabase().HasSraIndex(sraIndex))
                    {
                        throw HttpResponseException(HttpStatus::NOT_FOUND,
                                                    uri + " not found");
                    }
                    auto so = LockSra(sraIndex);
                    response.json = so->Json();

                    // now dive down into JSON member if necessary
                    while (args.size() >= 1)
                    {
                        const std::string member = args[0];
                        args.erase(args.begin());
                        if (response.json.isMember(member))
                        {
                            response.json = response.json[member];
                        }
                        else
                        {
                            PBLOG_ERROR << "member " << member << " not found "
                                << " in " << so->Json();
                            throw HttpResponseException(HttpStatus::NOT_FOUND,
                                uri + " not found, member " + member
                                + " does not exist");
                        }
                    }
                }
            }
            else if (args[0] == "api")
            {
                response.json = GetDocString();
                response.contentType = "text/plain";
            }
            else if (args[0] == "dashboard")
            {
                response.json = GetDashboardString();
                response.contentType = "text/plain";
            }
            else if (args[0] == "status")
            {
                response.json["uptime"] = GetUpTime();
                response.json["uptime_message"] = 
                    ConvertTimeSpanToString(response.json["uptime"].asDouble());
                response.json["cpu_load"] = 0.0; // CPU load as floating point,
                                                 // normalized to one core (1.0)
                response.json["time"] = PacBio::Utilities::Time::GetTimeOfDay();
                response.json["version"] = std::string(SHORT_VERSION_STRING) 
                    + "." + cmakeGitHash();
                //       << "\n git branch: " << cmakeGitBranch()
                //       << "\n git hash: " << cmakeGitHash()
                //       << "\n git commit date: " << cmakeGitCommitDate();
                response.json["num_sras"] = 666; // config_.numSRAs;
                response.json["personality"]["version"] = "?";
                response.json["personality"]["build_timestamp"] = "TBD"; // ISO 8091 of the build time, e.g. "2021-02-11 14:01:00Z",
                response.json["personality"]["comment"] = "n/a";         // optional comment about personality for R&D purposes,
                response.json["personality"]["wxinfo"] = "TBD"; // TODO  // all fields of the output of `wxinfo -d`
                response.json["personality"]["frameLatency"] = 512*3; // fixme
            }
            else if (args[0] == "log")
            {
                response.json = Json::arrayValue;
                response.json[0] = "log entry 0";
                response.json[1] = "log entry 1";
            }
#if 0          
            else
            {
                using namespace PacBio::Text;
                std::string uri1 = uri;
                if (uri1 == "") { uri1 = "index.html"; }
                std::string indexPath = webServer_->Config().uiSystemRoot() + "/" + uri1;
                if (PacBio::POSIX::IsFile(indexPath))
                {
                    PBLOG_INFO << "Serving file " << indexPath;
                    response.contentType = GuessContentTypeFromExtension(indexPath);
                    response.json = String::Slurp(indexPath);
                }
                else
                {
                    PBLOG_ERROR << "Local file " << indexPath << " not found";
                    throw HttpResponseException(HttpStatus::NOT_FOUND,
                                                indexPath + " not found");
                }
            }
#endif
#endif
            else
            {
                throw HttpResponseException(HttpStatus::NOT_FOUND, 
                    "path \'" + uri + "' not found");
            }
        }
        else
        {
            throw HttpResponseException(HttpStatus::NOT_FOUND, 
               "path \'" + uri + "' not found");
        }
    }
    catch(const HttpResponseException& ex)
    {
        response.httpStatus = ex.GetHttpStatus();
        response.json["message"] = ex.what();
        response.json["method"] = "GET";
        response.json["uri"] = uri;
    }
    catch(const pugi_pb::xpath_exception& ex)
    {
        response.httpStatus = HttpStatus::INTERNAL_SERVER_ERROR;
        response.json["message"] = std::string("pugi_pb::xpath_exception:(") + ex.what() + ", "
                                   + " xpath_offset:" + std::to_string(ex.result().offset) + ")";
        response.json["method"] = "GET";
        response.json["uri"] = uri;
    }
    catch (const std::exception& ex)
    {
        response.httpStatus = HttpStatus::INTERNAL_SERVER_ERROR;
        response.json["message"] = std::string("std::exception caught: ") + ex.what();
        response.json["method"] = "GET";
        response.json["uri"] = uri;
    }

    return response;
}

HttpResponse WebServiceHandler::GET_Postprimaries(const std::vector<std::string>& args)
{
    HttpResponse response;
    response.httpStatus = HttpStatus::NOT_IMPLEMENTED;

    Json::Value objects = Json::arrayValue;

    if (config_.debug.simulationLevel == 1)
    {
        if (args.size() == 0)
        {
            response.httpStatus = HttpStatus::OK;
            for(int i=1;i<=4;i++)
            {
                auto po = PacBio::API::CreateMockupOfPostprimaryObject("m123456_00000" + std::to_string(i));
                objects.append(po.mid);
    //            objects[po.mid] = po.Serialize();
            }
            response.json = objects;
        }
        else
        {
            response.httpStatus = HttpStatus::OK;
            std::string mid = args[0];
            auto po = PacBio::API::CreateMockupOfPostprimaryObject(mid);
            response.json = po.Serialize();
        }
    }

    return response;
}

HttpResponse WebServiceHandler::GET_Sockets(const std::vector<std::string>& args)
{
    HttpResponse response;
    response.httpStatus = HttpStatus::NOT_IMPLEMENTED;

    std::map<int,PacBio::API::SocketObject> sockets_;
    if (config_.debug.simulationLevel == 1)
    {
        response.httpStatus = HttpStatus::OK;
        for(uint32_t i=1; i<= 4; i++) sockets_[i] = PacBio::API::CreateMockupOfSocketObject(i);
    }

    if (args.size() == 0)
    {
        Json::Value objects = Json::arrayValue;
        for (const auto& o : sockets_)
        {
            assert(o.first == o.second.socketNumber);
            objects.append(o.first);
        }
        response.json = objects;
    }
    else
    {
        auto index = std::stoi(args[0]);
        response.json = sockets_[index].Serialize();

        // reformat the URL, changing slashes to dots, then
        // use Jsoncpp path feature.
        std::string path0 = 
            PacBio::Text::String::Join(args.begin() + 1, args.end(), '.');
        PBLOG_INFO << "grabbing " << path0;
        Json::Path path1(path0);
        response.json = path1.resolve(response.json, Json::nullValue);
    }

    return response;
}


HttpResponse WebServiceHandler::GET_Storages(const std::vector<std::string>& args)
{
    HttpResponse response;
    response.httpStatus = HttpStatus::NOT_IMPLEMENTED;

    if (config_.debug.simulationLevel == 1)
    {
        response.httpStatus = HttpStatus::OK;
        if (args.size() == 0)
        {
            Json::Value objects = Json::arrayValue;
            for(int i=1;i<=4;i++)
            {
                auto so = PacBio::API::CreateMockupOfStorageObject(i,"m123456_00000" + std::to_string(i));
                objects.append(so.mid);
                //objects[so.mid] = so.Serialize();
            }
            response.json = objects;
        }
        else
        {
            std::string mid = args[0];
            int i = mid.back() - '0';
            auto so = PacBio::API::CreateMockupOfStorageObject(i,mid);
            response.json = so.Serialize();
        }
    }

    return response;
}


HttpResponse WebServiceHandler::GET_Transfer(const std::vector<std::string>& args)
{
    HttpResponse response;
    response.httpStatus = HttpStatus::NOT_IMPLEMENTED;

    Json::Value objects = Json::arrayValue;
    if (config_.debug.simulationLevel == 1)
    {
        response.httpStatus = HttpStatus::OK;
        if (args.size() == 0)
        {
            for(uint32_t i=1;i<=4;i++)
            {
                auto t = PacBio::API::CreateMockupOfTransferObject(i, "m123456_00000" + std::to_string(i));
                objects.append(t.mid);
                // objects[t.mid] = t.Serialize();
            }
            response.json = objects;
        }
        else
        {
            std::string mid = args[0];
            int i = mid.back() - '0';
            auto t = PacBio::API::CreateMockupOfTransferObject(i,mid);
            response.json = t.Serialize();
        }

    }

    return response;
}

HttpResponse WebServiceHandler::POST_Postprimaries(const std::vector<std::string>& args, const std::string& postData)
{
    HttpResponse response;
    response.httpStatus = HttpStatus::NOT_IMPLEMENTED;
    return response;
}

HttpResponse WebServiceHandler::POST_Sockets(const std::vector<std::string>& args, const std::string& postData)
{
    HttpResponse response;
    response.httpStatus = HttpStatus::NOT_IMPLEMENTED;

    auto nextArg = args.begin();
    const auto json = PacBio::IPC::ParseJSON(postData);

    if (nextArg != args.end())
    {
        uint32_t socketNumber = std::stoul(*nextArg);
        if (socketNumber < config_.firstSocket || socketNumber > config_.lastSocket)
        {
            throw HttpResponseException(HttpStatus::FORBIDDEN, "POST contains out of range socket number:" + std::to_string(socketNumber));
        }
        nextArg++;
        if (nextArg == args.end())
        {
            throw HttpResponseException(HttpStatus::NOT_FOUND, "POST must include final app name: basecaller, darkcal or loadingcal");
        }
        if (postData == "")
        {
            throw HttpResponseException(HttpStatus::BAD_REQUEST, "POST contained empty data payload (JSON)");
        }
        if (*nextArg == "basecaller")
        {
            nextArg++;
            if (nextArg == args.end())
            {
                response.httpStatus = HttpStatus::OK;
                SocketBasecallerObject sbo(json);
                (void) socketNumber;
            }
            else if (*nextArg == "stop")
            {
                // do something
            }
            else
            {
                throw HttpResponseException(HttpStatus::NOT_FOUND, "POST to basecaller contains unknown command:" + *nextArg);
            }
        }
        else if (*nextArg == "darkcal")
        {
            nextArg++;
            if (nextArg == args.end())
            {
                SocketDarkcalObject sbo(json);
            }
            else if (*nextArg == "stop")
            {
                // do something
            }
            else
            {
                throw HttpResponseException(HttpStatus::NOT_FOUND, "POST to darkcal contains unknown command:" + *nextArg);
            }
        }
        else if (*nextArg == "loadingcal")
        {
            nextArg++;
            if (nextArg == args.end())
            {
                SocketLoadingcalObject sbo(json);
            }
            else if (*nextArg == "stop")
            {
                // do something
            }
            else
            {
                throw HttpResponseException(HttpStatus::NOT_FOUND, "POST to loadingcal contains unknown command:" + *nextArg);
            }
        }
        else
        {
            throw HttpResponseException(HttpStatus::NOT_FOUND, "POST URL must include final app name: basecaller, darkcal or loadingcal");
        }
    }
    else
    {
        throw HttpResponseException(HttpStatus::NOT_FOUND, "need socket number");
    }

    return response;
}

HttpResponse WebServiceHandler::POST_Storages(const std::vector<std::string>& args, const std::string& postData)
{
    HttpResponse response;
    response.httpStatus = HttpStatus::NOT_IMPLEMENTED;

    if (args.size() >= 1)
    {
        if (args[0] == "stop")
        {
            // do something
        }
    }
    return response;
}

HttpResponse WebServiceHandler::POST_Transfer(const std::vector<std::string>& args, const std::string& postData)
{
    HttpResponse response;
    response.httpStatus = HttpStatus::NOT_IMPLEMENTED;

    if (args.size() >= 1)
    {
        if (args[0] == "stop")
        {
            // do something
        }
    }

    return response;
}

std::string WebServiceHandler::GuessContentTypeFromExtension(const std::string& uri)
{
    using namespace PacBio::Text;

    const char*  contentType;
    if (String::EndsWith(uri, ".png"))
    {
        contentType = "image/png";
    }
    else if (String::EndsWith(uri, ".jpg"))
    {
        contentType = "image/jpg";
    }
    else if (String::EndsWith(uri, ".css"))
    {
        contentType = "text/css";
    }
    else if (String::EndsWith(uri, ".html"))
    {
        contentType = "text/html";
    }
    else if (String::EndsWith(uri,".ico"))
    {
        contentType = "image/x-icon";
    }
    else
    {
        contentType = "text/plain";
    }
    return contentType;
}

#if 0
/// clear any error states in the PA pipeline. It might take a few seconds for the "dismiss" message to
/// propagate through the pipeline and have all of the processes report in with a status messages,
/// so this can block up to about "maxRetries" seconds.
void WebServiceHandler::ClearErrors(const int maxRetries)
{
    if (AnyErrored())
    {
        {
            auto pis = LockPaInternalStatus();
            PBLOG_WARN << "At least one subsystem was in error mode: " + pis->RenderJSON();
        }
        PBLOG_INFO << "Attempting to dismiss error condition";
        acqRemote_.DismissError(); // this will dismiss all errors in the pipeline, starting with pa-acq, continuing
        // through pa-t2b and pa-bw. ppa does not have an error state, so it does not need to be cleared, although
        // this is confusing as there is a "PPAState::error" enum, but this is actually an exception event, not
        // a persistent state.
    }

    for (int attempt = maxRetries; attempt >= 0 ; attempt--)
    {
        bool ready = ReadyToStartAcquisition();
        if (ready)
        {
            break;
        }
        else
        {
            if (attempt == 0 )
            {
                auto pis = LockPaInternalStatus();
                throw HttpResponseException( HttpStatus::FORBIDDEN,
                                             "PA realtime is not ready to start acquisition. "
                                             "Can't start new acquisition." + pis->RenderJSON());
            }
            else
            {
                PBLOG_WARN << "PA realtime is not ready to start acquisition, remaining attempts " << attempt;
            }
            POSIX::sleep(1);
        }
    }
}

/// Creates a new acquisition for the database.
/// \param  postJson : the RunMetaData XML as part of a JSON object
/// \returns The HTTP response
HttpResponse WebServiceHandler::CreateAcquisitions(const Json::Value& postJson)
{
    HttpResponse response;
    response.httpStatus = HttpStatus::OK;

    if (!postJson["rundetails"].isString())
    {
        throw PBException("acquisition POST data does not have valid JSON with rundetails fields");
    }

    PBLOG_DEBUG << "POST Acquisitions Raw Run Details";
    PBLOG_DEBUG << postJson["rundetails"];

    std::string rmdString = postJson["rundetails"].asString();

    const int maxRetires = 7;
    ClearErrors(maxRetires);

    AcqResList acqResList = webServer_->CreateAcquisitions(rmdString);

    response.json = acqResList.Json();
    return response;
}
#endif


/// Handle the POST data
/// \param uri : the URI just POSTED to, with the protocol, hostname and port stripped off. Should not start with a /
/// \param postData : the JSON data in the HTTP request body.
/// \returns the HTTP response
HttpResponse WebServiceHandler::POST(
    const std::string& uri, 
    const std::string& postDataString)
{
    HttpResponse response;
    response.httpStatus = HttpStatus::OK;
    std::string acqId = "";

    try
    {
        Json::Value postJson = PacBio::IPC::ParseJSON(postDataString);

        auto args = String::Split(String::Chomp(uri,'/'), '/');

        PBLOG_DEBUG << "args:" << String::Join(args.begin(), args.end(), ',');

        if (args.size() >= 1)
        {
            if (args[0] == "sockets")
            {
                args.erase(args.begin());
                return POST_Sockets(args,postDataString);
            }
            else if (args[0] == "postprimaries")
            {
                args.erase(args.begin());
                return POST_Postprimaries(args,postDataString);
            }
            else if (args[0] == "storages")
            {
                args.erase(args.begin());
                return POST_Storages(args,postDataString);
            }
            else if (args[0] == "transfers")
            {
                args.erase(args.begin());
                return POST_Transfer(args,postDataString);
            }
            else if (args[0] == "api")
            {
                response.json = "here is the API";
            }
            else if (args[0] == "ping")
            {
                response.json["processed"] = PacBio::Utilities::Time::GetTimeOfDay();
                response.json["echo"] = postJson;
            }
            else if (args[0] == "kill")
            {
                PBLOG_INFO << "shutting down system ungracefully";
                POSIX::Sleep(0.01);
                raise(SIGKILL);
            }
            else if (args[0] == "shutdown")
            {
                threadController_->RequestExit();
            }
            else
            {
                throw HttpResponseException(HttpStatus::NOT_FOUND, uri + " not found");
            }
        }
        else
        {
            throw HttpResponseException(HttpStatus::NOT_FOUND, uri + " not found");
        }
    }
    catch(const HttpResponseException& ex)
    {
        response.httpStatus = ex.GetHttpStatus();
        response.json["message"] = ex.what();
        response.json["method"] = "POST";
        response.json["uri"] = uri;
    }
    catch(const pugi_pb::xpath_exception& ex)
    {
        response.httpStatus = HttpStatus::INTERNAL_SERVER_ERROR;
        response.json["message"] = std::string("pugi_pb::xpath_exception:(") + ex.what() + ", "
                                   + " xpath_offset:" + std::to_string(ex.result().offset) + ")";
        response.json["method"] = "POST";
        response.json["uri"] = uri;
    }
    catch(const std::exception& ex)
    {
        response.httpStatus = HttpStatus::INTERNAL_SERVER_ERROR;
        response.json["message"] = std::string("std::exception caught: ") + ex.what();
        response.json["method"] = "POST";
        response.json["uri"] = uri;
    }

    try
    {
        if (acqId != "" && response.httpStatus != HttpStatus::OK && response.httpStatus != HttpStatus::CREATED)
        {
            // TODO
//            auto ao = adb_->LockAcquisition(acqId);
//            ao->ArchiveException(response, "pa-ws.posthandler", uri + "\n" + postData1);
        }
    }
    catch(const std::exception& ex)
    {
        PBLOG_ERROR << "Exception thrown while handling exception! response.httpStatus:" << response.httpStatus.toString()
                    << " response.json:" << response.json
                    << " acqId:" << acqId
                    << " new exception:" << ex.what();
    }
    return response;
}

/// Handle the DELETE, which is liked a request to delete an acquisition or a smrt-transfer.
/// \param uri : the URI just POSTED to, with the protocol, hostname and port stripped off. Should not start with a /
/// \returns the HTTP response
HttpResponse WebServiceHandler::DELETE(const std::string& uri)
{
    HttpResponse response;
    response.httpStatus = HttpStatus::OK;
    try
    {
        auto args = String::Split(String::Chomp(uri,'/'), '/');

        throw HttpResponseException(HttpStatus::NOT_FOUND, "not found:" + args[0]);
    }

    catch (const HttpResponseException& ex)
    {
        response.httpStatus = ex.GetHttpStatus();
        response.json["message"] = ex.what();
        response.json["method"] = "DELETE";
        response.json["uri"] = uri;
    }
    catch (const pugi_pb::xpath_exception& ex)
    {
        response.httpStatus = HttpStatus::INTERNAL_SERVER_ERROR;
        response.json["message"] = std::string("pugi_pb::xpath_exception:(") + ex.what() + ", "
                                   + " xpath_offset:" + std::to_string(ex.result().offset) + ")";
        response.json["method"] = "DELETE";
        response.json["uri"] = uri;
    }
    catch (const std::exception& ex)
    {
        response.httpStatus = HttpStatus::INTERNAL_SERVER_ERROR;
        response.json["message"] = std::string("std::exception caught: ") + ex.what();
        response.json["method"] = "DELETE";
        response.json["uri"] = uri;
    }
    return response;
}


#if 0
/// helper function to go a HTTP GET
/// \param httpClient : the persistent client from the parent
/// \param url : the url to GET
/// \returns a ConfigurationObject derived instance (or anything that supports Load(std:string))
template<typename T>
T HttpGet(PacBio::IPC::CurlPlusPlus& httpClient, std::string& url)
{
    T t;
    std::string s = httpClient.Get(url);
    HttpStatus code = static_cast<HttpStatus>(httpClient.GetHttpCode());
    PBLOG_TRACE << url << " status:" << code << " " << s;
    if (code != HttpStatus::OK && code != HttpStatus::CREATED)
    {
        throw PBException(url + " GET failed with code " + std::to_string((int) code));
    }
    t.Load(s);
    return t;
}
#endif


}}}

