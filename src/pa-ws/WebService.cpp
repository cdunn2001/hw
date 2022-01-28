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
#include <regex>
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
#include <apihelp.h>
#include <DashboardWrapper.h>

// application includes
//#include "config/pa-ws-config.h"
#include "version.h"
//#include "HttpHelpers.h"
#include <git-rev.h>

#include "api/SocketObject.h"
#include "api/PostprimaryObject.h"
#include "api/StorageObject.h"

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
            else if (args[0] == "doc")
            {
                response.json = GetDocString();
                response.contentType = "text/html";
            }
            else if (args[0] == "api")
            {
                std::string api = GetApiString();
                std::regex e ("REPLACE_WITH_LOCALHOST");
                api = std::regex_replace (api, e, PacBio::POSIX::gethostname());

                response.json = api;
                response.contentType = "text/html";
            }
#if 0
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


HttpResponse WebServiceHandler::POST_Postprimaries(const std::vector<std::string>& args, const std::string& postData)
{
    HttpResponse response;
    response.httpStatus = HttpStatus::NOT_IMPLEMENTED;
    const auto json = PacBio::IPC::ParseJSON(postData);

    auto nextArg = args.begin();

    if (nextArg == args.end())
    {
        if (config_.debug.simulationLevel == 1)
        {
            PostprimaryObject ppaObject(json);
            response.httpStatus = HttpStatus::CREATED;
            response.json = ppaObject.mid;
        }
    }
    else
    {
        const std::string mid = *nextArg;
        ++nextArg;
        if (nextArg == args.end()) 
        {
            throw HttpResponseException(HttpStatus::NOT_FOUND, "POST to postprimaries must include command: stop");
        }
        else if (*nextArg == "stop")
        {
            if (config_.debug.simulationLevel == 1)
            {
                response.httpStatus = HttpStatus::OK;
                response.json = "tbd";
            }
        }
    }
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
        if (*nextArg == "reset")
        {
            // TODO
            // for(auto& socket: sockets_)
            // {
            //    socket.reset();  this is pseudocode
            // }
            if (config_.debug.simulationLevel == 1)
            {
                response.httpStatus = HttpStatus::OK;
                response.json = "tbd";
            }
            return response;
        }
        uint32_t socketNumber = std::stoul(*nextArg);
        if (socketNumber < config_.firstSocket || socketNumber > config_.lastSocket)
        {
            throw HttpResponseException(HttpStatus::FORBIDDEN, "POST contains out of range socket number:" + std::to_string(socketNumber));
        }
        nextArg++;
        if (nextArg == args.end())
        {
            throw HttpResponseException(HttpStatus::NOT_FOUND, "POST must include final app name: basecaller, darkcal, loadingcal or reset");
        }
        if (*nextArg == "basecaller")
        {
            nextArg++;
            if (nextArg == args.end() || *nextArg == "start")
            {
                if (postData == "")
                {
                    throw HttpResponseException(HttpStatus::BAD_REQUEST, "POST contained empty data payload (JSON)");
                }
                SocketBasecallerObject sbo(json);
                if (config_.debug.simulationLevel == 1)
                {
                    response.httpStatus = HttpStatus::CREATED;
                    response.json = sbo.Serialize();
                }
                (void) socketNumber;
            }
            else if (*nextArg == "stop")
            {
                // do something
                if (config_.debug.simulationLevel == 1)
                {
                    response.httpStatus = HttpStatus::OK;
                    response.json = "tbd";
                }
            }
            else
            {
                throw HttpResponseException(HttpStatus::NOT_FOUND, "POST to basecaller contains unknown command:" + *nextArg);
            }
        }
        else if (*nextArg == "darkcal")
        {
            nextArg++;
            if (nextArg == args.end() || *nextArg == "start")
            {
                if (postData == "")
                {
                    throw HttpResponseException(HttpStatus::BAD_REQUEST, "POST contained empty data payload (JSON)");
                }
                SocketDarkcalObject sdco(json);
                if (config_.debug.simulationLevel == 1)
                {
                    response.httpStatus = HttpStatus::CREATED;
                    response.json = sdco.Serialize();
                }
            }
            else if (*nextArg == "stop")
            {
                // do something
                if (config_.debug.simulationLevel == 1)
                {
                    response.httpStatus = HttpStatus::OK;
                    response.json = "tbd";
                }
            }
            else
            {
                throw HttpResponseException(HttpStatus::NOT_FOUND, "POST to darkcal contains unknown command:" + *nextArg);
            }
        }
        else if (*nextArg == "loadingcal")
        {
            nextArg++;
            if (nextArg == args.end() || *nextArg == "start")
            {
                if (postData == "")
                {
                    throw HttpResponseException(HttpStatus::BAD_REQUEST, "POST contained empty data payload (JSON)");
                }
                SocketLoadingcalObject slco(json);
                if (config_.debug.simulationLevel == 1)
                {
                    response.httpStatus = HttpStatus::CREATED;
                    response.json = slco.Serialize();
                }
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
        else if (*nextArg == "reset")
        {
            // TODO
            if (config_.debug.simulationLevel == 1)
            {
                response.httpStatus = HttpStatus::OK;
                response.json = "tbd";
            }
        }
        else
        {
            throw HttpResponseException(HttpStatus::NOT_FOUND, "POST URL must include final app name: basecaller, darkcal, loadingcal or reset");
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
            StorageObject so(json);
            // do something
            if (config_.debug.simulationLevel == 1)
            {
                response.httpStatus = HttpStatus::CREATED;
                response.json = so.Serialize();
            }
        } else if (*nextArg == "free")
        {
            // do something
            if (config_.debug.simulationLevel == 1)
            {
                response.httpStatus = HttpStatus::OK;
                response.json = "tbd";
            }
        }
        else
        {
            throw HttpResponseException(HttpStatus::NOT_FOUND, "POST to " + *nextArg + " doesn't exit");
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

/// Handle the DELETE
/// \param uri : the URI just POSTED to, with the protocol, hostname and port stripped off. Should not start with a /
/// \returns the HTTP response
HttpResponse WebServiceHandler::DELETE(const std::string& uri)
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
                return DELETE_Sockets(args);
            }
            else if (args[0] == "postprimaries")
            {
                args.erase(args.begin());
                return DELETE_Postprimaries(args);
            }
            else if (args[0] == "storages")
            {
                args.erase(args.begin());
                return DELETE_Storages(args);
            }
        }
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


HttpResponse WebServiceHandler::DELETE_Sockets(const std::vector<std::string>& args)
{
    HttpResponse response;
    response.httpStatus = HttpStatus::NOT_IMPLEMENTED;

    auto nextArg = args.begin();
    if (nextArg != args.end())
    {
        uint32_t socketNumber = std::stoul(*nextArg);
        if (socketNumber < config_.firstSocket || socketNumber > config_.lastSocket)
        {
            throw HttpResponseException(HttpStatus::FORBIDDEN, "POST contains out of range socket number:" + std::to_string(socketNumber));
        }
        if (config_.debug.simulationLevel == 1)
        {
            response.httpStatus = HttpStatus::OK;
            response.json = "tbd";
        }
    }
    return response;
}

HttpResponse WebServiceHandler::DELETE_Postprimaries(const std::vector<std::string>& args)
{
    HttpResponse response;
    response.httpStatus = HttpStatus::NOT_IMPLEMENTED;

    auto nextArg = args.begin();
    if (nextArg != args.end())
    {
        const std::string mid = *nextArg;
        if (config_.debug.simulationLevel == 1)
        {
            response.httpStatus = HttpStatus::OK;
            response.json = "tbd";
        }
    }
    return response;
}

HttpResponse WebServiceHandler::DELETE_Storages(const std::vector<std::string>& args)
{
    HttpResponse response;
    response.httpStatus = HttpStatus::NOT_IMPLEMENTED;

    auto nextArg = args.begin();
    if (nextArg != args.end())
    {
        uint32_t socketNumber = std::stoul(*nextArg);
        if (socketNumber < config_.firstSocket || socketNumber > config_.lastSocket)
        {
            throw HttpResponseException(HttpStatus::FORBIDDEN, "POST contains out of range socket number:" + std::to_string(socketNumber));
        }
        if (config_.debug.simulationLevel == 1)
        {
            response.httpStatus = HttpStatus::OK;
            response.json = "tbd";
        }
    }
    return response;
}

}}} // namespace

