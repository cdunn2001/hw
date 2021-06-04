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

#include <common/HttpHelpers.h>
#include <common/ThreadController.h>

#include <pa-ws/WebService.h>
#include <pa-ws/api/PostprimaryObject.h>
#include <pa-ws/api/SocketObject.h>
#include <pa-ws/api/StorageObject.h>
#include <pa-ws/api/TransferObject.h>  

#include  <pa-ws/mockup.h>

using namespace testing;
using namespace PacBio::Primary::PaWs;
using namespace PacBio::API;

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
        config.port = 0; // pick a port that is free
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
#if 0
    void Construct(int index)
    {
        SraObject sraObject0;
        sraObject0.sra_index = index;
        client.Post(url + "/sras/" + std::to_string(index), sraObject0.Json());
    }
#endif

};
#endif

// This test framework is only for directly testing methods directly, not through GET or POST.
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
        config.port = 0; // pick a port that is free
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
        "movie_max_frames" : 100,
        "movie_max_time": 60.0,
        "movie_number": 123,
        "baz_url" : "http://localhost:23632/storage/m123456/mine.baz",
        "log_url" : "http://localhost:23632/storage/m123456/log.txt",
        "log_level" : "DEBUG",
        "chiplayout" : "Minesweeper",
        "crosstalk_filter" :
        [
            [ 0.1,  0.2, 0.3 ],
            [ -0.1, 0.8, -0.2],
            [ 0.4,  0.5, 0.6 ]
        ],
        "trace_file_roi":
        [ [ 0, 0, 13, 32 ] ]
    }
    )JSON";

    auto r2 = service->POST_Sockets(args, json1);

}

TEST_F(WebService_Test,Failures)
{
    SocketObject so = CreateMockupOfSocketObject(1);

    client.Post(url + "/sockets/1/basecaller", PacBio::IPC::RenderJSON(so.Serialize()["basecaller"]));

    auto readback = client.Get(url + "/sockets/1");

    EXPECT_EQ(PacBio::IPC::ParseJSON(readback),  so.Serialize());
};

#if 0

TEST_F(WebService_Test,Failures)
{
    PacBio::Logging::LogSeverityContext context(PacBio::Logging::LogLevel::FATAL);

    client.Get(url + "/junkendpoint");
    EXPECT_EQ(HttpStatus::NOT_FOUND, client.GetHttpCode());

    client.Post(url + "/junkendpoint");
    EXPECT_EQ(HttpStatus::NOT_FOUND, client.GetHttpCode());

    // this requires a payload to post, and the payload should have
    // set the sra_index to 99. So this should result in an 
    // internal exception being thrown
    client.Post(url + "/sras/99");
    EXPECT_EQ(HttpStatus::INTERNAL_SERVER_ERROR, client.GetHttpCode());
}

TEST_F(WebService_Test,API)
{
    auto response = client.Get(url + "/api");
    EXPECT_EQ(HttpStatus::OK, client.GetHttpCode());
    EXPECT_THAT(response, HasSubstr("Primary Wolverine Daemon API"));
}

TEST_F(WebService_Test,Ping)
{
    auto response = client.Post(url + "/ping", Json::Value("ding a ling"));
    EXPECT_EQ(HttpStatus::OK, client.GetHttpCode());
    EXPECT_THAT(response, HasSubstr("ding a ling"));
}

TEST_F(WebService_Test,SraPost)
{
    // this test constructs the original SraObject and then reads it back to
    // verify that it is correct. Then it tests some of the other end points
    // to verify that they are processed and generate a SraMessage.
    SraObject sraObject0;
    sraObject0.sra_index = 0;
    sraObject0.configuration.movie_number = 123;
    auto response = client.Post(url + "/sras/0", sraObject0.Json());
    ASSERT_EQ(HttpStatus::CREATED, client.GetHttpCode());
    auto json1 = PacBio::IPC::ParseJSON(response);
    SraObject sraReadback0;
    sraReadback0.Load(json1);
    EXPECT_EQ(0, sraReadback0.sra_index);
    EXPECT_EQ(MovieState_e::IDLE, sraReadback0.status.live.movie_state());
    EXPECT_THAT(sraReadback0.url(), HasSubstr("http://"));
    sraObject0.url = sraReadback0.url();

    SraObject sraObject1;
    sraObject1.sra_index = 1;
    sraObject1.configuration.movie_number = 456;
    auto json2 = client.Post(url + "/sras/1", sraObject1.Json());
    ASSERT_EQ(HttpStatus::CREATED, client.GetHttpCode());
    SraObject sraReadback1;
    sraReadback1.Load(json2);
    EXPECT_EQ(1, sraReadback1.sra_index);
    sraObject1.url = sraReadback1.url();


    response = client.Post(url + "/sras/0/drain");
    ASSERT_EQ(HttpStatus::OK, client.GetHttpCode());
    json1 = PacBio::IPC::ParseJSON(response);
    EXPECT_EQ(lastIndex, 0);
    EXPECT_EQ(lastCommand, SraMessage::Command::DRAIN);
    EXPECT_EQ("\"ok\"", PacBio::IPC::RenderJSON(json1));

    response = client.Post(url + "/sras/1/drain");
    ASSERT_EQ(HttpStatus::OK, client.GetHttpCode());
    json1 = PacBio::IPC::ParseJSON(response);
    EXPECT_EQ(lastIndex, 1);
    EXPECT_EQ(lastCommand, SraMessage::Command::DRAIN);
    EXPECT_EQ("\"ok\"", PacBio::IPC::RenderJSON(json1));

    response = client.Post(url + "/sras/0/transmit");
    ASSERT_EQ(HttpStatus::OK, client.GetHttpCode());
    json1 = PacBio::IPC::ParseJSON(response);
    EXPECT_EQ(lastIndex, 0);
    EXPECT_EQ(lastCommand, SraMessage::Command::TRANSMIT);
    EXPECT_EQ("\"ok\"", PacBio::IPC::RenderJSON(json1));
}

TEST_F(WebService_Test,Reconfigure)
{
    Construct(0);

    SraReconfigure reconfigure;
    reconfigure.expectedPackets = 1;
    reconfigure.expectedZmws = 64;
    reconfigure.layoutDims[0] = 1;
    reconfigure.layoutDims[1] = 512;
    reconfigure.layoutDims[2] = 64;
    reconfigure.numFrames = 512;
    reconfigure.senderToRemotePort = 55555; // dummy port

    reconfigure.remoteHost = "localhost";
    auto reconfigureJson = reconfigure.Json();    
    auto response2 = client.Post(url + "/sras/0/reconfigure",reconfigureJson);
    auto json2 = PacBio::IPC::ParseJSON(response2);
    ASSERT_EQ(HttpStatus::OK, client.GetHttpCode());
}

#endif
