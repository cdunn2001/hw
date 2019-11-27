//
// Created by mlakata on 6/29/16.
//

#include <gtest/gtest.h>

#include <pacbio/primary/FromAcquisition.h>
#include <pacbio/process/RESTServer.h>
#include <pacbio/primary/ipc_config.h>
#include <pacbio/ipc/JSON.h>
#include <pacbio/ipc/PoolSCIF.h>
#include <pacbio/ipc/PoolShmXSI.h>

using namespace PacBio::Primary;
using namespace PacBio::Process;

#if 0
// TODO fix these tests

class RESTServerEx : public RESTServer
{
public:
    RESTServerEx() : RESTServer(0, "/")
    {
    }
};

RESTServerEx restServer;

RESTDataModelLock GetREST_Root()
{
    return restServer.CheckoutRoot("testFromAcquisition.cpp:29");
}

class FromAcquisitionExposed : public FromAcquisition
{
public:
    using FromAcquisition::FromAcquisition;
    using FromAcquisition::poolz_;
};

class FromAcquisitionTest : public ::testing::Test
{
public:
    const uint32_t numReadBuffers = 16;
    const size_t rbBytes = ReadBuffer::SizeOf();
    PacBio::IPC::PoolMalloc<ReadBuffer> readBufferPool;
    FromAcquisitionTest()
        : readBufferPool{numReadBuffers, rbBytes}
    {
        readBufferPool.Create();
    }
    Tranche* GetInitializedTranche()
    {
        Tranche* t = new Tranche;
        t->AssignTraceDataPointer( Tranche::Pixelz::Factory(16384));
        for (uint32_t j = 0; j < numReadBuffers; j++)
        {
            ReadBuffer* r = ReadBuffer::Initialize::FromMemoryLocation(readBufferPool.GetPointerByIndex(j));
            t->calls[j] = r;
            t->calls[j]->Reset(j);
        }
        return t;
    }
};

TEST_F(FromAcquisitionTest,SCIF)
{
    PacBio::Logging::LogSeverityContext context(PacBio::Logging::LogLevel::WARN);

    PacBio::IPC::MessageQueueLabel a("ctl",12345);
    PacBio::IPC::MessageQueueLabel b("stat",12346);
    uint32_t MAX_NUM_THREADS = 1;
    PacBio::ThreadSafeQueue<Tranche*> trancheReadyQueue;
    uint64_t numElements = 100;
    size_t elementSize = 16384 * 64;

    Json::Value poolConfigJson = PacBio::IPC::ParseJSON(R"(
     {
            "elementSize" : 32768,
            "node" : "Host",
            "numElements" : 6019128,
            "poolName" : "acq",
            "poolType" : "SHM_XSI",
            "port" : 46606
    }
        )");
    poolConfigJson["numElements"] = numElements;
    poolConfigJson["elementSize"] = elementSize;

    PacBio::IPC::PoolSCIF_Host <Tranche::Pixelz> host(PacBio::IPC::PoolSCIF_Interface::Node::Host,PORT_SCIF_ACQUISITION,1,elementSize);
    host.Create();

    FromAcquisitionExposed fromAcq(a, b, MAX_NUM_THREADS, trancheReadyQueue, elementSize);
    fromAcq.Init(poolConfigJson);
    EXPECT_EQ(host.GetTotalSize(), fromAcq.poolz_->GetTotalSize());

    fromAcq.SetSCIF_Ready();
    fromAcq.SetPipelineReady();
    EXPECT_TRUE(fromAcq.EverythingReady());

    Tranche* t = GetInitializedTranche();
    fromAcq.AddFreeTranche(t);
    EXPECT_EQ(1,fromAcq.NumFreeTranches());

    TrancheTitle title;

    PacBio::IPC::Deed deed("deed",title);
    EXPECT_TRUE(fromAcq.ReceiveDeed(deed, std::chrono::milliseconds(50000)));
    EXPECT_EQ(0,fromAcq.NumFreeTranches());
}

TEST_F(FromAcquisitionTest,Shared)
{
    PacBio::Logging::LogSeverityContext context(PacBio::Logging::LogLevel::WARN);

    PacBio::IPC::MessageQueueLabel a("ctl",12345);
    PacBio::IPC::MessageQueueLabel b("stat",12346);
    uint32_t MAX_NUM_THREADS = 1;
    PacBio::ThreadSafeQueue<Tranche*> trancheReadyQueue;
    uint64_t numElements = 100;
    size_t elementSize = 16384 * 64;

    Json::Value poolConfigJson = PacBio::IPC::ParseJSON(R"(
 {
        "elementSize" : 32768,
        "node" : "Host",
        "numElements" : 6019128,
        "poolName" : "acq",
        "poolType" : "SHM_XSI",
        "port" : 46606
}
    )");
    poolConfigJson["numElements"] = numElements;
    poolConfigJson["elementSize"] = elementSize;
    PacBio::IPC::PoolShmXSI <Tranche::Pixelz> host(PORT_SCIF_ACQUISITION,numElements,elementSize);
    host.Create();

    FromAcquisitionExposed fromAcq(a, b, MAX_NUM_THREADS, trancheReadyQueue, elementSize);
    fromAcq.Init(poolConfigJson);
    EXPECT_EQ(host.GetTotalSize(), fromAcq.poolz_->GetTotalSize());

    fromAcq.SetSCIF_Ready();
    fromAcq.SetPipelineReady();
    EXPECT_TRUE(fromAcq.EverythingReady());

    Tranche* t = GetInitializedTranche();
    fromAcq.AddFreeTranche(t);
    EXPECT_EQ(1,fromAcq.NumFreeTranches());

    TrancheTitle title;

    PacBio::IPC::Deed deed("deed",title);
    EXPECT_TRUE(fromAcq.ReceiveDeed(deed, std::chrono::milliseconds(50000)));
    EXPECT_EQ(0,fromAcq.NumFreeTranches());
}

#if 0
// TODO add tests for these methods
      void ReceiveMessage(Tranche* msg);
      void MainThread(PacBio::Process::ThreadedProcessBase& parent);
#endif

#endif
