//
// Created by mlakata on 4/21/15.
//

#include <map>
#include <iostream>
#include <future>

#include <gtest/gtest.h>

#include <pacbio/dev/gtest-extras.h>
#include <pacbio/dev/ExpectSession.h>
#include <pacbio/ipc/PoolSCIF.h>
#include <pacbio/logging/Logger.h>
#include <pacbio/process/ProcessBase.h>
#include <pacbio/process/RESTServer.h>
#include <pacbio/POSIX.h>

#include <pacbio/primary/ipc_config.h>
#include <pacbio/primary/TrancheTitle.h>
#include <pacbio/primary/PipelineHost.h>

#include <pacbio-primary-test-config.h>

using namespace PacBio::Process;
using namespace PacBio::Primary;
using namespace PacBio::IPC;
using namespace PacBio::Dev;
using namespace PacBio;
using namespace std;

std::vector<PacBio::IPC::MessageQueueLabel> labelsSim{
        PacBio::IPC::MessageQueueLabel("firstmic", "pac-host", PORT_CTRL_BASECALLER_MIC0)
};

std::vector<PacBio::IPC::MessageQueueLabel> labels{
        pa_t2b0CommandQueue
//        , pa_t2b1CommandQueue
//        , pa_t2b2CommandQueue
};

const uint32_t NUM_TILES_PER_TRANCH = 32;
const uint32_t NUM_TRANCHES = 512*32;
const uint32_t NUM_TOTAL_TILES = NUM_TRANCHES * NUM_TILES_PER_TRANCH;
std::vector<uint16_t> defaultMicPort{ PORT_CTRL_BASECALLER_MIC0,
                                               PORT_CTRL_BASECALLER_MIC1,
                                               PORT_CTRL_BASECALLER_MIC2 };

class Foo : public ProcessBase
{
public:
    PacBio::Primary::PipelineHost pipelineHost;
    uint32_t returnedTiles;
    PoolSCIF_Host<Tile> tilePool_;

    Foo(std::string hostlist) :
            ProcessBase(pa_acqCommandQueue,pa_acqStatusQueue),
            pipelineHost(*this,"up",defaultMicPort),
            returnedTiles(0),
            tilePool_{PoolSCIF_Interface::Node::Host, PORT_SCIF_ACQUISITION, 0}
    {
        pipelineHost.AddExpectedNodeList(hostlist);
    }

    Foo(std::vector<PacBio::IPC::MessageQueueLabel> hostlist) :
        ProcessBase(pa_acqCommandQueue,pa_acqStatusQueue),
        pipelineHost(*this,"up",defaultMicPort),
        returnedTiles(0),
        tilePool_{PoolSCIF_Interface::Node::Host, PORT_SCIF_ACQUISITION, 0}
    {
        for(auto& node : hostlist)
        {
            pipelineHost.AddExpectedNode(node);
        }


        ADD_COMMAND("heartbeat",
        {
            pipelineHost.AddHeartbeat(msg.GetData());
        } );
        ADD_COMMAND("deed",
        {
            Title<Tile> title(&msg);
            for (uint32_t i = 0; i < title.GetNumOffsets(); i++)
            {
                uint64_t offset = title.GetOffset(i);
                Tile*  tile = tilePool_.GetPointerByOffset(offset);
#if 1  // fixme
                (void) tile;
#else
                AddFreeTile(tile);
#endif
                returnedTiles ++;
                if (returnedTiles >= NUM_TOTAL_TILES )
                {
                    Abort();
                }
            }
            PBLOG_DEBUG << "Deed: returnedTiles:" << returnedTiles << "," << NUM_TOTAL_TILES;
        });
    }

    void Run()
    {
        PBLOG_INFO << "Creating pool";
        tilePool_.ResizeByElements(NUM_TOTAL_TILES);
        tilePool_.Create();
        pipelineHost.AssignPool(&tilePool_);

        PBLOG_INFO << "Waiting for heartbeats";
        PBLOG_INFO << "All phis are alive, connection SCIF";
        pipelineHost.ConnectToAllMics([](){});

        PBLOG_INFO << "sending a title";

        int k = 0;
        int zmwIndex = 0;
        uint32_t row = 32;
        uint32_t col = 32;
        AutoTimer timer("transfer",NUM_TRANCHES * NUM_TILES_PER_TRANCH * sizeof(Tile),"bytes");

        for(uint32_t i=0;i<NUM_TRANCHES;i++)
        {
            TrancheTitle title;
            title.ConfigWord( 0x12345678);
            title.FrameCount(512);
            title.FrameIndexStart(0);
            title.TimeStampStart(0);
            title.ZmwIndex(zmwIndex);
            title.ZmwNumber(row*65536 + col);
            for(uint32_t j=0;j<NUM_TILES_PER_TRANCH;j++)
            {
                Tile* tile = tilePool_.GetPointerByIndex(k++);
                title.AddOffset(tilePool_.GetOffset(tile));
                tile->SetPattern(j);
            }
            title.MicOffset(0);
            title.Lane(0);
            title.SuperChunkIndex(0);
            pipelineHost.SendDeed("deed",title,title.MicOffset());

            zmwIndex+= 16;
            col += 16;
            if (col >= 1024+32)
            {
                col = 32;
                row++;
            }
        }

        while(true)
        {
            if (ExitRequested()) break;

            int count = pipelineHost.MicPresentCount();
            PBLOG_INFO << "phis visible:" << count;
            POSIX::sleep(1);
        }
        PBLOG_INFO << "Rate: " << timer.GetRate() << " bytes/s";
        PBLOG_INFO << "Quit";
        pipelineHost.Quit();
    }
};

class PipelineHostTest : public ::testing::Test
{
    void SetUp()
    {

    }
};

RESTServer restServer(PORT_HTTP_ACQUISITION, POSIX::GetCurrentWorkingDirectory());

RESTDataModelLock GetREST_Root()
{
    return restServer.CheckoutRoot();
}


class ProxyWrapper
{
public:
    ExpectSession session;

    ProxyWrapper()
    {
        const char* execmic = "./t2bProxy-k1om-prefix/src/t2bProxy-k1om-build/t2bProxy";
        string rsaKey = string(sequelBaseDir) + "/common/pbi.id_rsa";
        {
            PRINTF("Copying %s\n", execmic);

            ExpectSession cp;
            cp.ShowOutput();
            cp.Spawn("scp", "-i",rsaKey.c_str(), execmic, "pbi@mic0:");
            cp.ExpectEOF();
            // COMMAND scp ${execmic} /opt/intel/tbb/lib/mic/libtbb.so.2  /opt/intel/tbb/lib/mic/libtbb_debug.so.2 /opt/intel/tbb/lib/mic/libtbbmalloc_debug.so.2  pbi@mic0:

            PRINTF("Copied\n");
        }

        session.ShowOutput();
        session.Spawn("ssh", "-t","-t","-i",rsaKey.c_str(),"pbi@mic0", "ls; /home/pbi/t2bProxy");
        session.ExpectExact("--- t2bProxy ---");

        PRINTF("proxy ready\n");
    }
    void Monitor()
    {
        session.Timeout(100);
        session.ExpectRegex("INFO.*pipeline: done");
    }
    ~ProxyWrapper()
    {
      //  session.KillAll();
    }
};

TEST(PipelineHost,PortMaps)
{
    {
        Foo foo("mic0,mic1,mic2");
        ASSERT_EQ(3,foo.pipelineHost.Labels().size());
        EXPECT_EQ("mic0:46610", foo.pipelineHost.Labels()[0].GetFullAddress());
        EXPECT_EQ("mic1:46640", foo.pipelineHost.Labels()[1].GetFullAddress());
        EXPECT_EQ("mic2:46650", foo.pipelineHost.Labels()[2].GetFullAddress());
    }
    {
        Foo foo("default");
        ASSERT_EQ(3,foo.pipelineHost.Labels().size());
        EXPECT_EQ("mic0:46610", foo.pipelineHost.Labels()[0].GetFullAddress());
        EXPECT_EQ("mic1:46640", foo.pipelineHost.Labels()[1].GetFullAddress());
        EXPECT_EQ("mic2:46650", foo.pipelineHost.Labels()[2].GetFullAddress());
    }
    {
        Foo foo("3mic");
        ASSERT_EQ(3,foo.pipelineHost.Labels().size());
        EXPECT_EQ("mic0:46610", foo.pipelineHost.Labels()[0].GetFullAddress());
        EXPECT_EQ("mic1:46640", foo.pipelineHost.Labels()[1].GetFullAddress());
        EXPECT_EQ("mic2:46650", foo.pipelineHost.Labels()[2].GetFullAddress());
    }
    {
        Foo foo("2mic1host");
        ASSERT_EQ(3,foo.pipelineHost.Labels().size());
        EXPECT_EQ("mic0:46610", foo.pipelineHost.Labels()[0].GetFullAddress());
        EXPECT_EQ("mic1:46640", foo.pipelineHost.Labels()[1].GetFullAddress());
        EXPECT_EQ("pac-host:46650", foo.pipelineHost.Labels()[2].GetFullAddress());
    }
    {
        Foo foo("2mic");
        ASSERT_EQ(2,foo.pipelineHost.Labels().size());
        EXPECT_EQ("mic0:46610", foo.pipelineHost.Labels()[0].GetFullAddress());
        EXPECT_EQ("mic1:46640", foo.pipelineHost.Labels()[1].GetFullAddress());
    }
    {
        Foo foo("1host");
        ASSERT_EQ(1,foo.pipelineHost.Labels().size());
        EXPECT_EQ("pac-host:46610", foo.pipelineHost.Labels()[0].GetFullAddress());
    }
}

// This test talks to a t2bProxy *simulator* (x86, not MIC) running on the same host.
// The operator is expected to launch t2bProxy manually.
TEST(PipelineHostTest,SimulatorTestManual)
{
    const char* execmic = "./t2bProxy-x86_64-prefix/src/t2bProxy-x86_64-build/t2bProxy";
    PRINTF("Please issue '%s' in another terminal session'\n",
           execmic);

    Foo foo(labelsSim);
    foo.CreateThread("foo", [&]() { foo.Run(); });
    foo.MainEventLoopAndJoin();
};

// This test talks to mic0.
// The operator is expected to ssh to mic0, and run t2bProxy manually.
TEST(PipelineHostTest,MicTestManual)
{
    const char* execmic = "./t2bProxy-k1om-prefix/src/t2bProxy-k1om-build/t2bProxy";
    PRINTF("Please issue 'scp %s pbi@mic0:', then 'ssh pbi@mic0 LD_LIBRARY_PATH=/home/pbi /home/pbi/t2bProxy'\n",
           execmic);

    Foo foo(labels);
    foo.CreateThread("foo", [&]() { foo.Run(); });
    foo.MainEventLoopAndJoin();
};

// This tests talks to mic0, mic1 and mic2.
// This test is 100% automatic. The mic executables are copied and launched automatically.
TEST(PipelineHostTest,MicTestAuto)
{
    ProxyWrapper _;

    auto f = std::async(std::launch::async,[]() {
        Foo foo(labels);
        foo.CreateThread("foo", [&]() { foo.Run(); });
        foo.MainEventLoopAndJoin();
    });

    _.Monitor();
    (void) f.get();
};