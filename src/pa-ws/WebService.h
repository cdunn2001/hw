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
/// \brief  declaration of the endpoint handler for pa-ws
//
// Programmer: Mark Lakata

#ifndef PRIMARY_PAWS_WEBSERVICE_H
#define PRIMARY_PAWS_WEBSERVICE_H

#include <map>
#include <functional>
#include <stdint.h>

#include <pacbio/process/CivetServer.h>

#include <app-common/HttpHelpers.h>
#include <app-common/ThreadController.h>

#include "api/SocketObject.h"
#include "PaWsConfig.h"
#if 0
#include "SharedMemoryApiObject.h"
#include "SraMessage.h"
#include "SraObject.h"
#include "SraObjectDatabase.h"
#include "WebServiceConfig.h"
#include "WxDaemonForwards.h"
#endif

namespace PacBio {
namespace Primary {
namespace PaWs {

/// The master service
class WebService
{
public:
    /// Constructor
    WebService(const PaWsConfig& config,
        std::shared_ptr<PacBio::Threading::IThreadController> threadController
    );
    ~WebService();
    WebService(const WebService&) = delete;
    WebService(WebService&&) = delete;
    WebService& operator=(const WebService&) = delete;
    WebService& operator=(WebService&&) = delete;

    typedef std::string SraMessage;
    typedef std::function<void(uint32_t, SraMessage&&) > SraCallback_t;
    void InstallService(SraCallback_t sraCallback);


    CivetServer& GetCivetServer() 
    { 
        if (!server_) throw PBException("Server not ready");
        return *server_;
    }

    std::string GetUrl() const
    {
        return "http://localhost:" + std::to_string(config_.port);
    }

private:
    PaWsConfig config_;
    std::shared_ptr<PacBio::Threading::IThreadController> threadController_;
    std::shared_ptr<CivetServer> server_;
    std::list<std::unique_ptr<CivetHandler>> handlers_;
    std::string rootUrl_;
};

class SocketConfig
{
public:
    using SocketId = std::string;

    SocketConfig(const PaWsConfig& config);

    /// Set the map of socketIds.
    /// 0-based board number is the vector index.
    /// Raise on any dups.
    void SetSocketIds(const std::vector<SocketId>& socketIds);

    /// True iff the board index is known for this socketId.
    bool IsValid(const SocketId& socketId) const;

    /// This is the number in use, not counting Boards that
    /// happen to be powered off.
    uint32_t GetNumBoards() const;

    std::vector<SocketId> GetValidSocketIds() const;

private:
    // Key-sorting is helpful for tests, so avoid hashmap.
    std::set<SocketId> socketId2index_;
};

/// The GET, POST and DELETE handler for this webservice.
class WebServiceHandler :
       public CivetHandler
{
public:
    WebServiceHandler(const PaWsConfig& config,
        std::shared_ptr<PacBio::Threading::IThreadController> threadController);
    ~WebServiceHandler();
    WebServiceHandler(const WebServiceHandler&) = delete;
    WebServiceHandler(WebServiceHandler&&) = delete;
    WebServiceHandler& operator=(const WebServiceHandler&) = delete;
    WebServiceHandler& operator=(WebServiceHandler&&) = delete;

    void InstallSraCallback(WebService::SraCallback_t sraCallback)
    {
        sraCallback_ = sraCallback;
    }

    /// Civet "OPTIONS" override. Handle preflight check. This mostly to support CORS.
    bool handleOptions(CivetServer* server, struct mg_connection* conn) override;

    /// Civet GET override
    bool handleGet(CivetServer* server, struct mg_connection* conn) override;

    /// Civet POST override
    bool handlePost(CivetServer* server, struct mg_connection* conn) override;

    /// Civet DELETE override
    bool handleDelete(CivetServer* server, struct mg_connection* conn) override;

    /// Civet PUT override
    bool handlePut(CivetServer* server, struct mg_connection* conn) override;

    PacBio::IPC::HttpResponse GET(const std::string& uri);

    PacBio::IPC::HttpResponse POST(const std::string& uri, const std::string& data);

    PacBio::IPC::HttpResponse DELETE(const std::string& uri);

    PacBio::IPC::HttpResponse GET_Postprimaries(const std::vector<std::string>& args);
    PacBio::IPC::HttpResponse GET_Sockets(const std::vector<std::string>& args);
    PacBio::IPC::HttpResponse GET_Storages(const std::vector<std::string>& args);

    PacBio::IPC::HttpResponse POST_Postprimaries(const std::vector<std::string>& args, const std::string& postData);
    PacBio::IPC::HttpResponse POST_Sockets(const std::vector<std::string>& args, const std::string& postData);
    PacBio::IPC::HttpResponse POST_Storages(const std::vector<std::string>& args, const std::string& postData);

    PacBio::IPC::HttpResponse DELETE_Postprimaries(const std::vector<std::string>& args);
    PacBio::IPC::HttpResponse DELETE_Sockets(const std::vector<std::string>& args);
    PacBio::IPC::HttpResponse DELETE_Storages(const std::vector<std::string>& args);


    /// General http formatted response function.
    /// \param conn : mongoose connection
    /// \param httpStatus : status of response
    /// \param contentType : the content type of the response
    /// \param respdata : the body of the http packet
    /// \param extraHeaders : formatted header lines, lines must be terminated with \r\n
    void Respond(struct mg_connection* conn, PacBio::IPC::HttpStatus httpStatus, const char* contentType,
                 const std::string& respdata, const std::string& extraHeaders = "");

    /// Raise if socketId does not have an assigned board index.
    static void ValidateSocketId(const SocketConfig& socketConfig, const std::string& socketId);

protected:

#if 0
    /// locks existing SraObject in the database, and returns a special Lock
    /// object that has pointer semantics. A mutex is locked for the duraction
    /// of the Lock lifetime.
    auto LockSra(uint32_t index, bool createIfNecessary=false)
    {
        return GetSraObjectDatabase().LockSra(index, createIfNecessary);
    }
#endif

private:
    class Summarizer
    {
    private:
        std::mutex mutex_;
        std::map<long, uint64_t> stats_;
    public:
        void Increment(long ipAddress)
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stats_[ipAddress]++;
        }
        Json::Value SummarizeRequests();
    };

    const PaWsConfig config_;
    SocketConfig socketConfig_;

    std::shared_ptr<PacBio::Threading::IThreadController> threadController_;
    WebService::SraCallback_t sraCallback_;

    Summarizer requestsGet_;
    Summarizer requestsPost_;
    Summarizer requestsPut_;
    Summarizer requestsDelete_;
    Summarizer requestsOptions_;
    double summarizeHttpRequestsTime_ = 0;
    double processStartTime_ = 0;
    std::string rootUrl_;
    std::map<SocketConfig::SocketId, PacBio::API::SocketObject> sockets_;

    std::string GuessContentTypeFromExtension(const std::string& uri);

    /// \returns a JSON string that summarizes the requests to the pa-ws REST service, for the
    /// last time interval. After calling this function, the statistics will be reset for the
    /// next time interval.
    std::string SummarizeHttpRequests();

    /// \returns seconds since process was started
    double GetUpTime() const;

    // converts the seconds into a human readable expression of
    // days, hours, minutes, seconds
    /// \param seconds a time span expressed in seconds
    /// \returns a string that looks something like "1d 3h 34m 1.345s"
    std::string ConvertTimeSpanToString(double seconds);
};

/// Convert ip address integer (host order) to a xxx.xxx.xxx.xxx IP address string
/// \param ip : host order 32-bit integer address
/// \returns "xxx.xxx.xxx.xxx" formatted string
extern std::string inet_ntos(long ip);

}}}


#endif // include guard
