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

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <pacbio/dev/gtest-extras.h>
#include <pacbio/ipc/CurlPlusPlus.h>
#include <pacbio/ipc/JSON.h>
#include <pacbio/ipc/MessageQueue.h>

#include <app-common/HttpHelpers.h>
#include <app-common/ThreadController.h>

#include <pa-ws/WebService.h>
#include <pa-ws/api/PostprimaryObject.h>
#include <pa-ws/api/SocketObject.h>
#include <pa-ws/api/StorageObject.h>

#include  <pa-ws/mockup.h>

using namespace testing;
using namespace PacBio::Primary::PaWs;
using namespace PacBio::API;
using namespace PacBio::IPC;

class TestThreadController :  public PacBio::Threading::IThreadController
{
public:
    bool ExitRequested() override 
    { 
        return true;
    }
    void RequestExit() override
    {
        // do nothing for this test suite
    }
};

#if 1
class WebService_Test : public testing::Test
{
public:
    PacBio::Logging::LogSeverityContext context{PacBio::Logging::LogLevel::WARN};

    PaWsConfig config;
    std::unique_ptr<WebService> service;
    uint16_t port;
    std::string url;
    PacBio::IPC::CurlPlusPlus client;
    std::shared_ptr<TestThreadController> threadController;
    WebService::SraMessage lastCommand;
    uint32_t lastIndex = 0xFFFFFFFF;

    WebService_Test()
    {
        threadController = std::make_shared<TestThreadController>();
        config.platform = PacBio::Sensor::Platform::Kestrel;
        FactoryConfig(&config);
        config.port = 0; // pick a port that is free
        config.debug.simulationLevel = 1;
        service = std::make_unique<WebService>(config,threadController);
        service->InstallService([this](uint32_t index, const WebService::SraMessage& message)
        {
            Callback(index,message);
        });

        auto ports = service->GetCivetServer().getListeningPorts();
        if(ports.size()<1) throw PBException("No ports available");
        port = ports[0];
        url = "http://localhost:" + std::to_string(port);
    }
    void Callback(uint32_t index, const WebService::SraMessage& message)
    {
        PBLOG_DEBUG << "Callback " << index << " message:" << message;
        lastIndex = index;
        lastCommand = message;
    }
};
#endif

// This test framework is only for directly testing methods directly, not through actual http requests.
class WebServiceHandler_Test : public testing::Test
{
public:
    PacBio::Logging::LogSeverityContext context{PacBio::Logging::LogLevel::DEBUG};

    PaWsConfig config;
    std::unique_ptr<WebServiceHandler> service;
    std::shared_ptr<TestThreadController> threadController;

    WebServiceHandler_Test()
    {
        threadController = std::make_shared<TestThreadController>();
        config.platform = PacBio::Sensor::Platform::Kestrel;
        FactoryConfig(&config);
        config.port = 0; // pick a port that is free
        config.debug.simulationLevel = 1;
        service = std::make_unique<WebServiceHandler>(config,threadController);
    }
};

TEST_F(WebServiceHandler_Test,BasecallerEndpoint)
{
    const std::vector<std::string> args = {"1","basecaller"};
    auto r = service->GET_Sockets(args);
    SocketBasecallerObject sbo(r.json);

    const std::string json1 = R"JSON(
    {
        "movieMaxFrames" : 100,
        "movieMaxSeconds": 60.0,
        "movieNumber": 123,
        "bazUrl" : "http://localhost:23632/storage/m123456/mine.baz",
        "logUrl" : "http://localhost:23632/storage/m123456/log.txt",
        "logLevel" : "DEBUG",
        "chiplayout" : "Minesweeper",
        "crosstalkFilter" :
        [
            [ 0.1,  0.2, 0.3 ],
            [ -0.1, 0.8, -0.2],
            [ 0.4,  0.5, 0.6 ]
        ],
        "traceFileRoi":
        [ [ 0, 0, 13, 32 ] ]
    }
    )JSON";

    auto r2 = service->POST_Sockets(args, json1);
}

// Tests GET on missing socket.
TEST_F(WebService_Test, GetSocketNotFound)
{
    const PacBio::IPC::CurlPlusPlus::ContentBundle content = client.GetContentBundle(url + "/sockets/snafu");
    EXPECT_EQ(content.httpCode, HttpStatus::NOT_FOUND);
    EXPECT_THAT(content.data, HasSubstr("GET socket id='snafu' not found"));
};

// Tests POST
TEST_F(WebService_Test, PostReset)
{
    const auto response = client.Post(url + "/sockets/reset", Json::Value());
    EXPECT_THAT(response, HasSubstr("discard:"));

    const auto code = client.GetHttpCode();
    EXPECT_EQ(code, HttpStatus::OK);
};

// Tests POST/GET consistency.
TEST_F(WebService_Test, BasecallerGetReturnsPostContents)
{
    SocketObject so = CreateMockupOfSocketObject("1");
    auto referenceJson = so.Serialize();
    referenceJson["basecaller"].removeMember("processStatus");
    referenceJson["basecaller"].removeMember("rtMetrics");
    referenceJson["darkcal"].removeMember("processStatus");
    referenceJson["loadingcal"].removeMember("processStatus");
    client.Post(url + "/sockets/1/basecaller", PacBio::IPC::RenderJSON(referenceJson["basecaller"]));

    auto readback = client.Get(url + "/sockets/1");
    auto readbackJson = PacBio::IPC::ParseJSON(readback);
    readbackJson["basecaller"].removeMember("processStatus");
    readbackJson["basecaller"].removeMember("rtMetrics");
    readbackJson["darkcal"].removeMember("processStatus");
    readbackJson["loadingcal"].removeMember("processStatus");

    EXPECT_EQ(PacBio::IPC::RenderJSON(readbackJson),  PacBio::IPC::RenderJSON(referenceJson));
};


TEST_F(WebService_Test, DISABLED_BasecallerGetReturnsPostContents_Timed)
{
    // Today, this is identical to the test above.
    // Someday, we want to call a few times, and check that the
    // median is fast enough. See
    //   http://bitbucket.pacificbiosciences.com:7990/projects/SEQ/repos/pa-kestrel/pull-requests/14/overview?commentId=59887
    SocketObject so = CreateMockupOfSocketObject("1");
    auto referenceJson = so.Serialize();
    referenceJson["basecaller"].removeMember("processStatus");
    referenceJson["basecaller"].removeMember("rtMetrics");
    referenceJson["darkcal"].removeMember("processStatus");
    referenceJson["loadingcal"].removeMember("processStatus");
    client.Post(url + "/sockets/1/basecaller", PacBio::IPC::RenderJSON(referenceJson["basecaller"]));

    auto readback = client.Get(url + "/sockets/1");
    auto readbackJson = PacBio::IPC::ParseJSON(readback);
    readbackJson["basecaller"].removeMember("processStatus");
    readbackJson["basecaller"].removeMember("rtMetrics");
    readbackJson["darkcal"].removeMember("processStatus");
    readbackJson["loadingcal"].removeMember("processStatus");

    EXPECT_EQ(PacBio::IPC::RenderJSON(readbackJson),  PacBio::IPC::RenderJSON(referenceJson));
};

TEST_F(WebService_Test, Failures)
{
    PacBio::Logging::LogSeverityContext context(PacBio::Logging::LogLevel::FATAL);

    client.Get(url + "/junkendpoint");
    EXPECT_EQ(HttpStatus::NOT_FOUND, client.GetHttpCode());

    client.Post(url + "/junkendpoint");
    EXPECT_EQ(HttpStatus::NOT_FOUND, client.GetHttpCode());

    // missing app name (e.g. "basecaller")
    // Really, this should be
    //   405 Method Not Allowed
    // but then we would have to include an Allow header to specify 'GET'.
    // It is simpler to consider this as a missing path.
    client.Post(url + "/sockets/1", Json::Value());
    EXPECT_EQ(HttpStatus::NOT_FOUND, client.GetHttpCode());

    // empty payload
    client.Post(url + "/sockets/1/basecaller");
    EXPECT_EQ(HttpStatus::BAD_REQUEST, client.GetHttpCode());

    // out of range resource number
    client.Post(url + "/sockets/99/basecaller", Json::Value() );
    EXPECT_EQ(HttpStatus::NOT_FOUND, client.GetHttpCode());

    // out of range resource number
    client.Post(url + "/sockets/99/reset", Json::Value());
    EXPECT_EQ(HttpStatus::NOT_FOUND, client.GetHttpCode());
}

TEST_F(WebService_Test,Doc)
{
    auto response = client.Get(url + "/doc");
    EXPECT_EQ(HttpStatus::OK, client.GetHttpCode());
    EXPECT_THAT(response, AnyOf(HasSubstr("Primary Analysis Webservice API")));
}

TEST_F(WebService_Test,Api)
{
    auto response = client.Get(url + "/api");
    EXPECT_EQ(HttpStatus::OK, client.GetHttpCode());
    EXPECT_THAT(response, AnyOf(HasSubstr("here is the API"),HasSubstr("Swagger")));
}

TEST_F(WebService_Test,Ping)
{
    auto response = client.Post(url + "/ping", Json::Value("ding a ling"));
    EXPECT_EQ(HttpStatus::OK, client.GetHttpCode());
    EXPECT_THAT(response, HasSubstr("ding a ling"));
}

TEST(SocketConfig_Test, ValidateSocketId)
{
    // ValidateSocketId is a static method on
    //   WebServiceHandler
    // b/c it can throw an Http exception.
    using PacBio::IPC::HttpResponseException;

    {
        Json::Value json;
        json["socketIds"][0] = "1";
        json["socketIds"][1] = "2";
        PaWsConfig conf(json);
        SocketConfig conf1(conf);
        WebServiceHandler::ValidateSocketId(conf1, "1");
        WebServiceHandler::ValidateSocketId(conf1, "2");
        EXPECT_THROW({
            try
            {
                WebServiceHandler::ValidateSocketId(conf1, "3");
            }
            catch(const PacBio::IPC::HttpResponseException& exc)
            {
                EXPECT_THAT(exc.what(), HasSubstr("found out of range socket id"));
                throw;
            }
        }, HttpResponseException);

        // For now, we require simple integers.
        EXPECT_THROW({
            try
            {
                WebServiceHandler::ValidateSocketId(conf1, "abc");
            }
            catch(const PacBio::IPC::HttpResponseException& exc)
            {
                EXPECT_THAT(exc.what(), HasSubstr("found non-numeric socket id"));
                throw;
            }
        }, HttpResponseException);
    }
}

TEST(SocketConfig_Test, GetValid)
{
    Json::Value json;
    json["socketIds"][1] = "2";
    json["socketIds"][0] = "1";
    PaWsConfig conf(json);
    SocketConfig conf1(conf);
    const std::vector<SocketConfig::SocketId> expectedids {"1", "2"};
    EXPECT_EQ(conf1.GetValidSocketIds(), expectedids);
}
