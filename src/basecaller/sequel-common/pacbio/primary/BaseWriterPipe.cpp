// Copyright (c) 2014, Pacific Biosciences of California, Inc.
// Copyright (c) 2014, Pacific Biosciences of California, Inc.
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
/// \brief  manages the pipeline to the basewriter service.
//
// Programmer: Mark Lakata

#include <atomic>
#include <future>
#include <string>

#include <boost/align/aligned_allocator.hpp>

#include <pacbio/ipc/PoolFactory.h>
#include <pacbio/ipc/TitleBase.h>
#include <pacbio/ipc/ThreadSafeQueue.h>
#include <pacbio/dev/AutoTimer.h>
#include <pacbio/POSIX.h>
#include <pacbio/process/RESTServer.h>
#include <pacbio/smrtdata/NucleotideLabel.h>
#include <pacbio/Utilities.h>
#include <pacbio/primary/Tranche.h>
#include <pacbio/primary/ipc_config.h>
#include <pacbio/primary/PrimaryConfig.h>
#include <pacbio/primary/ChipLayout.h>
#include <pacbio/primary/REST_RootInterface.h>


#include "BaseWriterPipe.h"
#include <pacbio/primary/BaseWriterRemoteControl.h>
#include <pacbio/primary/SequelDefinitions.h>

using namespace PacBio;
using namespace PacBio::Primary;
using namespace PacBio::IPC;
using namespace PacBio::Dev;
using namespace PacBio::SmrtData;

static uint64_t gotMarks = 0;
static uint64_t waitedForTranches = 0;

std::vector<int> BaseWriterPipe::currentFrames_;

BaseWriterPipe::BaseWriterPipe(const MessageQueueLabel& baseCallerCommandQueueLabel2,
                               const MessageQueueLabel& baseWriterCommandQueueLabel,
                               uint32_t micOffset,
                               uint32_t numZMWs,
                               bool sos) :
        baseWriterRemoteControl_(new BaseWriterRemoteControl("pac-host") ),
        heartbeat_(baseCallerCommandQueueLabel2),
        micOffset_(micOffset),
        downstreamDeedReturn_(baseCallerCommandQueueLabel2, std::chrono::milliseconds(100)),
        simulateBasewriter_(false),
        numZMWs_(numZMWs),
        sos_(sos),
        chip_(GetPrimaryConfig().chipClass())
{
    (void)baseWriterCommandQueueLabel;
    currentFrames_.resize(numZMWs);
}

BaseWriterPipe::~BaseWriterPipe()
{
    delete baseWriterRemoteControl_;
}

void BaseWriterPipe::AllocPool(uint32_t maxReadBuffers)
{
    auto& poolConfigBw(GetPrimaryConfig().poolConfigBw);
    poolConfigBw.numElements = maxReadBuffers;
    ReadBuffer::Configure(GetPrimaryConfig());
    poolConfigBw.elementSize = ReadBuffer::SizeOf();

    pool_.reset(PacBio::IPC::PoolFactory::ConstructClient<ReadBuffer>(poolConfigBw));
    IPoolWithCopies* ipwc = dynamic_cast<IPoolWithCopies*>(pool_.get());
    if (ipwc)
    {
        pool_->MallocPool();
    }
    PBLOG_INFO << "BaseWriterPipe::BaseWriterPipe";
}

void BaseWriterPipe::Init()
{
    if(!simulateBasewriter_) pool_->Connect();
}

void BaseWriterPipe::SetSCIF_Ready()
{
    heartbeat_.SetSCIF_Ready();
    heartbeat_.SetPipelineReady();
}

bool BaseWriterPipe::IsConnected() const
{
    return heartbeat_.SCIF_Ready();
}

void BaseWriterPipe::AddFreeOffset(uint64_t offset)
{
#if 1
    PBLOG_TRACE << "BaseWriterPipe::AddFreeOffset: " << offset;
#endif
    freeReadBufferQueue_.Push(offset);
}

void BaseWriterPipe::AddFreeTitle(const TitleBase& title)
{
    for(uint32_t zmw=0;zmw<title.GetNumOffsets();zmw++)
    {
        uint64_t offset = title.GetOffset(zmw);
        AddFreeOffset(offset);
    }
}

ReadBuffer* BaseWriterPipe::GetPoolPointerByIndex(uint32_t index)
{
    return pool_->GetPointerByIndex(index);
}

static std::atomic<uint32_t> receivedDeeds {0};
void BaseWriterPipe::ReturnDeedThread(PacBio::Process::ThreadedProcessBase& parent)
{
    while(!parent.ExitRequested())
    {
        SmallMessage msg = downstreamDeedReturn_.Receive();
        if (msg)
        {
            if (msg.GetName() == "readbufferdeed")
            {
                Title<ReadBuffer> title;
                title.DeserializeMessage(&msg);
                PBLOG_TRACE << " BaseWriterPipe::ReturnDeedThread readbufferdeed offset:" << title.GetOffset(0);
                AddFreeTitle(title);
                receivedDeeds++;
            }
            else if (msg.GetName() == "init")
            {
                PBLOG_INFO << " BaseWriterPipe::ReturnDeedThread init";
                Init();
                SetSCIF_Ready();
            }
            else if (msg.GetName() == "ping")
            {
                PBLOG_INFO << " BaseWriterPipe::ReturnDeedThread ping";
            }
            else
            {
                throw PBException("bad message:" + msg.GetName());
            }
        }
    }
}

/// Does DMA scheduling
void BaseWriterPipe::SendPhase1(Tranche* tranche)
{
    if(simulateBasewriter_) return;

    IPoolWithCopies* ipwc = dynamic_cast<IPoolWithCopies*>(pool_.get());

    if (tranche->Type() == Tranche::MessageType::Data)
    {
        assert(& tranche->Title() != nullptr);

        TrancheTitle& title(tranche->titleOut_);
        title.Reset();
        title.InitFromTranche(*tranche, micOffset_);
#if 0
        if(tranche->ZmwLaneIndex() < 10)
        {
            PBLOG_INFO << "BaseWriterPipe::SendPhase1(Tranche* tranche)" << *tranche;
        }
#endif

        for (uint32_t zmw = 0; zmw < zmwsPerTranche; zmw++)
        {
            ReadBuffer* rb = tranche->calls[zmw];
            uint64_t offset;
            {
//                AutoTimer t("pop time");
                offset = freeReadBufferQueue_.Pop();
                if (offset & 0xFFF)   throw PBException("ReadBuffer is not 4K aligned!");
//                PBLOG_DEBUG << "pop time:" << t.GetElapsedMilliseconds();
            }
            title.AddOffset(offset);

            PBLOG_TRACE << "BaseWriterPipe::SendPhase1: new readbuffer " << (void*) rb << " offset:" << offset << " events:" << rb->NumSamples() << " zmwIndex:" << rb->ZmwIndex();

            uint32_t zmwIndex = rb->ZmwIndex();
            uint32_t numEvents = rb->NumSamples();
            uint32_t numHFmetrics = rb->NumMetrics();
            PBLOG_TRACE << "BaseWriterPipe::SendPhase1: creating ReadBuffer, events:"
                        << numEvents << " numMetrics:" << numHFmetrics << " zmwindex:"
                        << std::hex << zmwIndex << std::dec;

#if 1
            if (zmwIndex != title.ZmwIndex() + zmw)
            {
                PBLOG_WARN << "zmwIndex != title.ZmwIndex() + zmw, " << zmwIndex << " " << title.ZmwIndex() + zmw;
            }
#endif

            size_t len = pool_->RoundUp4K(rb->Size());
            if (ipwc)
            {
                if (reinterpret_cast<uint64_t>(rb) & 0xFFF)
                {
                    PBExceptionStream() << "ReadBuffer is not 4K aligned! was " << (void*)rb;
                }
                ipwc->CopyBytesToRemotePoolAsync(offset, rb, len);
            }
            else
            {
                PBLOG_TRACE << "memcpy(" << "offset:" << offset << "dst:" << (void*)pool_->GetPointerByOffset(offset) << " rb:" << (void*)rb << " len:"<< len;
                memcpy(pool_->GetPointerByOffset(offset),rb,len);
            }
        }
        if (ipwc)
        {
            tranche->markOut_ = ipwc->GetMark();
        }
        gotMarks++;
    }
    else
    {
        PBLOG_WARN << "BaseWriterPipe::SendPhase1 Funny ITranche data type " << (int)tranche->Type();
    }
}

void BaseWriterPipe::SendPhase2(Tranche* tranche)
{
    if(simulateBasewriter_)
    {
        return;
    }

    IPoolWithCopies* ipwc = dynamic_cast<IPoolWithCopies*>(pool_.get());

    if (tranche->Type() == Tranche::MessageType::Data)
    {
        if (ipwc)
        {
            ipwc->WaitForMark(tranche->markOut_);
        }
        waitedForTranches++;

        if (simulateBasewriter_)
        {

        }
        else
        {
            PBLOG_TRACE << "BaseWriterPipe::SendPhase2 SendDeed";
            baseWriterRemoteControl_->SendReadBufferDeed(tranche->titleOut_);
        }

#if 1
        PBLOG_TRACE<< "sending " << tranche->titleOut_ << " firstoffset:" << tranche->titleOut_.GetOffset(0);

        for(int i=0;i<16;i++)
        {
            // for debug
            ReadBuffer* rb = tranche->calls[i];
            if (rb->ZmwIndex() != tranche->ZmwIndex() + i)
            {
                PBLOG_WARN << "SendPhase2 Mismatch zmwIndex != title.ZmwIndex(), " << rb->ZmwIndex() << " " <<
                           tranche->ZmwIndex();
            }
            rb->Check();
        }
#endif
    } else {
        PBLOG_WARN << "BaseWriterPipe::SendPhase1: Funny ITranche data type " << (int) tranche->Type();
    }
}


void BaseWriterPipe::RunSimulationSetup(const std::string& filename, uint32_t numSimZmws, const std::string& chipLayoutName)
{
    if (numZMWs_ != numSimZmws) throw PBException("Internal inconsistency");
    baseWriterRemoteControl_->Setup(filename, numZMWs_, chipLayoutName);
}

class DmaThreadClass
{
public:
    DmaThreadClass(ThreadSafeQueue<Tranche*>& freeQueue,
                   ThreadSafeQueue<Tranche*>& dmaQueue,
                   std::function<void(void)> init,
                   std::function<void(Tranche*)> f) :
            freeQueue_(freeQueue),
            dmaQueue_(dmaQueue),
            init_(init),
            f_(f)
    {
        dmaThread_ = std::async(std::launch::async,[this]() {
            PacBio::Utilities::StdThread::SetThreadName("bistdma");
            Tranche* t;
            uint32_t sentDeeds = 0;
            init_();
            while (1)
            {
                t = dmaQueue_.Pop();
                if (t == nullptr) break;

                f_(t);
                sentDeeds ++;
                freeQueue_.Push(t);
            }
            return sentDeeds;
        });
    }
    ~DmaThreadClass()
    {
        if (dmaThread_.valid())
        {
            (void) SentDeeds();
        }
        PBLOG_DEBUG << "~DmaThreadClass done";
    }
    uint32_t SentDeeds()
    {
        PBLOG_DEBUG << "SentDeeds = ??";
        dmaQueue_.Push(nullptr);
        uint32_t count = dmaThread_.get(); // this waits and gets the result
        PBLOG_DEBUG << "SentDeeds = " << count;
        return count;
    }
    std::future<uint32_t> dmaThread_;
    ThreadSafeQueue<Tranche*>& freeQueue_;
    ThreadSafeQueue<Tranche*>& dmaQueue_;
    std::function<void(void)> init_;
    std::function<void(Tranche*)> f_;
};

void BaseWriterPipe::AssignCallBuffers(Tranche* tranches, size_t numTranches)
{
   auto ipwc = GetIPoolWithCopies();
    if (ipwc)
    {
        for (uint32_t i = 0; i < numTranches; i++)
        {
            // initialized read buffer pointers
            for (uint32_t j = 0; j < zmwsPerTranche; j++)
            {
                tranches[i].calls[j] =
                        ReadBuffer::Initialize::FromMemoryLocation(GetPoolPointerByIndex(i * zmwsPerTranche + j));
            }
        }
    }
    else
    {
        // ack, this is a memory leak. Never freed...
        ReadBuffer* rbbase = PacBio::HugePage::Malloc<ReadBuffer>(trancheQueueSize * zmwsPerTranche, ReadBuffer::SizeOf());
        for (uint32_t i = 0; i < numTranches; i++)
        {
            // initialized read buffer pointers
            for (uint32_t j = 0; j < zmwsPerTranche; j++)
            {
                tranches[i].calls[j] =
                        reinterpret_cast<ReadBuffer*>(
                                reinterpret_cast<uint8_t*>(rbbase) + ((i * 16 + j) * ReadBuffer::SizeOf()));
            }
        }
    }
}

void BaseWriterPipe::RunSimulationContinue(uint32_t numBasesPerSuperchunk,  uint32_t numSuperChunks, std::mutex& threadMutex, std::function<void(void)> polledFunction)
{
    uint64_t numBytes = 0;

    AutoTimer timer1("simulation",512*32,"frames");
    AutoTimer timer2("simulation",numBytes,"bytes");
    PBLOG_INFO << "ReadBuffer size: " << sizeof(ReadBuffer);

    uint32_t sentDeeds = 0;

    std::vector<Tranche> tranches(2); // the t2b simulator thread is so light weight that it only needs 2 local tranches
    ThreadSafeQueue<Tranche*> freeQueue;
    ThreadSafeQueue<Tranche*> dmaQueue;

    {
        DmaThreadClass dmaThread(freeQueue, dmaQueue,
               /*init*/  [this](){
                                     PBLOG_DEBUG << "BaseWriterPipe::RunSimulation: Start";
                                     baseWriterRemoteControl_->Start();
                                 },
       /*each tranche*/  [this](Tranche* t){
                                     SendPhase2(t);
                                 });

        AssignCallBuffers(&tranches[0], tranches.size());
        for (auto& t : tranches)
        {
            t.type = Tranche::MessageType::Data;
            freeQueue.Push(&t);
        }

        PBLOG_INFO << "RunSimulation: Simulating numSuperChunks=" << numSuperChunks;
        for (uint32_t ichunk = 0; ichunk < numSuperChunks; ichunk++)
        {
            PBLOG_INFO << "RunSimulation: Simulating chunk=" << ichunk;
            uint32_t zmw = 0;
            while (zmw < numZMWs_)
            {
                if (zmw % 1024 == 0)
                {
                    PBLOG_DEBUG << "zmw: " << zmw << " of " << numZMWs_;
                    static double tLastHousekeeping = 0;

                    double t = PacBio::Utilities::Time::GetMonotonicTime();
                    double deltaTime = (t - tLastHousekeeping);
                    if (deltaTime >= 1.0 )
                    {
                        EmitHeartbeat();
                        tLastHousekeeping = t;
                    }
                }
                polledFunction();

                // setup the buffering tranches
                Tranche* dummyTranche = freeQueue.Pop();
                dummyTranche->Create2C2ATitle(); //fixme for spider
                int tranche = zmw/16;
                const int numMics = 1;
                int micoffset = tranche % numMics;
                int lane = tranche / numMics;
                dummyTranche->Title().MicOffset(micoffset);
                dummyTranche->Title().PixelLane(lane);
                dummyTranche->Title().SuperChunkIndex(ichunk);
                dummyTranche->Title().FrameCount(GetPrimaryConfig().cache.framesPerSuperchunk);
                dummyTranche->Title().FrameIndexStart(ichunk * GetPrimaryConfig().cache.framesPerSuperchunk);
                dummyTranche->Title().TimeStampStart(ichunk * GetPrimaryConfig().cache.framesPerSuperchunk*100000);
                dummyTranche->Title().TimeStampDelta(100000);
                dummyTranche->ZmwIndex(zmw);


                StartDummyData();
                for (uint16_t i = 0; i < zmwsPerTranche; i++)
                {
                    ReadBuffer* rb = dummyTranche->calls[i];
                    {
                        std::lock_guard<std::mutex> lock(threadMutex);
                        rb->Reset(zmw + i);

                        switch (chip_)
                        {
                            case ChipClass::Sequel:
                                if (!sos_)
                                    GetDummyData<BasecallingMetricsT::Sequel>(rb, numBasesPerSuperchunk,
                                                                              numSuperChunks, ichunk);
                                else
                                    GetDummyData<BasecallingMetricsT::Spider>(rb, numBasesPerSuperchunk,
                                                                              numSuperChunks, ichunk);
                                break;
                            case ChipClass::Spider:
                                GetDummyData<BasecallingMetricsT::Spider>(rb, numBasesPerSuperchunk,
                                                                          numSuperChunks, ichunk);
                                break;
                            default:
                                throw PBException("Unknown chip type for writing data ");
                        }
                    }
                    numBytes += rb->Size();
                }


                SendPhase1(dummyTranche);
                dmaQueue.Push(dummyTranche);
                zmw += zmwsPerTranche;
            }
            PBLOG_INFO << "RunSimulation: completed ichunk " << ichunk << " zmw: " << zmw << " of " << numZMWs_;

        }
        PBLOG_INFO << "RunSimulation: done with numSuperChunks=" << numSuperChunks;
        timer2.SetCount(numBytes);
        sentDeeds = dmaThread.SentDeeds();

    } // end of DmaThreadClass scope
    PBLOG_DEBUG << "BaseWriterPipe::RunSimulation: exited BIST main loop";
    while(receivedDeeds < sentDeeds)
    {
        PBLOG_DEBUG << "BaseWriterPipe::RunSimulation: waiting" << receivedDeeds << " " << sentDeeds << " " <<freeReadBufferQueue_.Size();
        PacBio::POSIX::sleep(1);
    }

    PBLOG_DEBUG << "BaseWriterPipe::RunSimulation: Stop";
    baseWriterRemoteControl_->Stop();
}


void BaseWriterPipe::StartDummyData()
{
    if (currentFrames_.size() == 0) { throw PBException("not initilized!"); }
    for(auto& x : currentFrames_) x= 0 ;
}

template <typename TMetric>
void BaseWriterPipe::GetDummyData(ReadBuffer* rb, uint32_t numEvents, uint32_t numSuperChunk, uint32_t currSuperChunk)
{
    uint32_t zmw = rb->ZmwIndex();
    if (zmw >= numZMWs_) throw PBException("Bad zmw " + std::to_string(zmw) + " >= " + std::to_string(numZMWs_));

    if (zmw >= currentFrames_.size())  throw PBException("Bad zmw " + std::to_string(zmw) + " >= " + std::to_string(currentFrames_.size()));

    if (currSuperChunk < numSuperChunk)
    {
        rb->NumSamples(numEvents);
        int ipd = 8;
        for (uint32_t i = 0;
             i < numEvents;
             ++i)
        {
            Basecall &basecall(rb->EditSample(i));
            basecall.Base(NucleotideLabel::C);
            basecall.DeletionTag(NucleotideLabel::N);
            basecall.SubstitutionTag(NucleotideLabel::N);
            basecall.DeletionQV(4);
            basecall.SubstitutionQV(6);
            basecall.InsertionQV(7);

            auto &pulse = basecall.GetPulse();
            pulse.MergeQV(5);
            pulse.Start(currentFrames_[zmw] + ipd);
            pulse.Width(5);
            pulse.MeanSignal(200.0f).MidSignal(210.0f).MaxSignal(220.0f);

            currentFrames_[zmw] += ipd + pulse.Width();
        }
    }

    uint32_t numHFmetrics = GetPrimaryConfig().cache.blocksPerTranche;
    uint32_t metricLag = 2;

    if (currSuperChunk == 0)
    {
        numHFmetrics -= metricLag;
    }
    else if (currSuperChunk == numSuperChunk - 1)
    {
        numHFmetrics += metricLag;
    }

    rb->NumMetrics(numHFmetrics);

	for (uint32_t i = 0; i < numHFmetrics; ++i)
	{
		TMetric& metric(rb->EditMetric<TMetric>(i));
		auto& tm = metric.TraceMetrics();
		tm.StartFrame(i * 1024).NumFrames((i + 1) * 1024);
        metric.NumPulseFrames(20);
        metric.NumPulses(20);
        metric.NumBaseFrames(20);
        metric.NumBases(20);
        metric.PkmidSignal() = {{ 200.0f, 150.0f, 300.0f, 210.0f }};
        metric.Bpzvar() = {{ 30.0f, 32.0f, 24.0f, 45.0f }};
        metric.PkmidNumFrames() = {{ 4, 4, 4, 4 }};
        metric.NumPkmidBasesByAnalog() = {{ 5, 5, 5, 5 }};
	}
}

PacBio::Primary::BaseWriterRemoteControl& BaseWriterPipe::GetControl() {
    if (!baseWriterRemoteControl_ )
    {
        throw PBException("baseWriterRemoteControl_ not constructed");
    }
    return *baseWriterRemoteControl_;
}

void BaseWriterPipe::EmitHeartbeat()
{
    if (!simulateBasewriter_) GetControl().EmitHeartbeat(heartbeat_.Serialize());

    const auto freeReadBufferQueueSize = freeReadBufferQueue_.Size();
    {
        auto root = GetREST_Root();

        root->ModifyNumber("BaseWriterPipe/freeReadBufferQueue_", freeReadBufferQueueSize);
        root->ModifyNumber("BaseWriterPipe/gotMarks", gotMarks);
        root->ModifyNumber("BaseWriterPipe/waitedForTranches", waitedForTranches);
    }
}

void BaseWriterPipe::SendMessage(const PacBio::IPC::Message& mesg)
{
    if (!simulateBasewriter_)  GetControl().SendMessage(mesg);
}

void BaseWriterPipe::RunDownstreamSimulation()
{
    simulateBasewriter_ = true;

    MessageSocketSender bwSender(downstreamDeedReturn_.Label());
    bwSender.Send(Announcement("init"));
}

