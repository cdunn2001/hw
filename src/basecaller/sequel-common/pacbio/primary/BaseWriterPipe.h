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
#pragma once

#include <memory>

#include <pacbio/ipc/ThreadSafeQueue.h>
#include <pacbio/ipc/MessageQueue.h>
#include <pacbio/ipc/PoolFactory.h>
#include <pacbio/primary/ZmwResultBuffer.h>
#include <pacbio/primary/SCIF_Heartbeat.h>
#include <pacbio/process/ProcessBase.h>

namespace PacBio {

namespace IPC {
class TitleBase;
}

namespace Primary {

class Tranche;
class BaseWriterRemoteControl;
class ChipLayout;

/// The BaseWriterPipe is a class that abstracts the "producer" end of the pipe between the base caller (pa-t2b) and the
/// writer (pa-bw). It is used both by pa-t2b (as the garden mode) and pa-bw (as the built-in self test mode).
/// It manages a memory pool of ReadBuffers, copies the ReadBuffers locally to the remote process using SCIF DMA
/// and IPC messages to manage ownership of the ReadBuffers.
class BaseWriterPipe
{
public:
    /// Constructor.
    /// \param baseCallerCommandQueueLabel - upstream socket label
    /// \param baseWriterCommandQueueLabel - downstream socket label
    /// \param micOffset - the virtual MIC index (usually 0, 1 or 2). This should be 0 for simulations (emulating mic0)

    BaseWriterPipe(const PacBio::IPC::MessageQueueLabel& baseCallerCommandQueueLabel,
                   const PacBio::IPC::MessageQueueLabel& baseWriterCommandQueueLabel, uint32_t micOffset,
                   uint32_t numZMWs, bool sos
    );
    ~BaseWriterPipe();

    size_t AvailableReadBuffers() const { return freeReadBufferQueue_.Size(); }

    /// Connects to the downstream pool via SCIF. The peer must be online already (negotiated previously)
    void Init();

    /// Set a flag that the SCIF connection is ready.
    void SetSCIF_Ready();

    /// Recycles the contents of the title, by putting the ReadBuffers back in the free queue.
    void AddFreeTitle(const PacBio::IPC::TitleBase& title);

    /// Returns a ReadBuffer from the pool by index.
    ReadBuffer* GetPoolPointerByIndex(uint32_t index);

    /// Schedules a DMA operation to copy the ReadBuffers from local to remote SCIF nodes. SendPhase2 must be
    /// called after this call to complete the transaction.  Running SendPhase1 and SendPhase2 in separate threads
    /// increases efficiency by pipelining the DMA scheduling.
    void SendPhase1(PacBio::Primary::Tranche* tranche);

    /// Waits until DMA operation is complete for the indicated tranche. SendPhase1 must be called previously,
    /// either in the same thread (OK for testing) or different thread (recommended).
    void SendPhase2(PacBio::Primary::Tranche* tranche);

    /// Send an IPC message to the downstream peer.
    void SendMessage(const PacBio::IPC::Message& mesg);

    /// Assign the readbuffers to the calls array.
    void AssignCallBuffers(Tranche* tranches, size_t numTranches);

    /// Simulation methods.

    /// Setup the basecaller simulator.
    /// \param filename - ?
    /// \param numSimZmws - number of simulated ZMWs
    /// \param chipLayoutName - official layout name
    void RunSimulationSetup(const std::string& filename,uint32_t numSimZmws, const std::string& chipLayoutName);

    /// Runs the actual simulator, sending ReadBuffers to the downstream peer.
    /// \param numSimZmws - number of simulated ZMWS
    /// \param numBasesPerSuperchunk - number of bases to simulate per superchunk
    /// \param numSuperchunks - number of super chunks to simulate
    /// \param mutex - a mutex to make this thread safe
    /// \param f - a function to call in the main loop of the simulator. This can be used to abort or update statistics.
    void RunSimulationContinue(uint32_t numBasesPerSuperchunk, uint32_t numSuperchunks, std::mutex& mutex,std::function<void(void)> f);

    /// Aborts the simulation, so that the thread can end.
    void RunSimulationAbort();

    /// Fill the ReadBuffer with suitable dummy base calls.
    /// \param rb - the buffer to fill
    /// \param numEvents - number of bases to instead into the ReadBuffer
    /// \param numSuperChunk - the number of super chunks total
    /// \param currSuperChunk - the number of the current super chunk
    template <typename TMetric>
    void GetDummyData(ReadBuffer* rb, uint32_t numEvents, uint32_t numSuperChunk, uint32_t currSuperChunk);

    /// Initializes the dummy data
    void StartDummyData();

    /// Returns a handle to the remote control
    PacBio::Primary::BaseWriterRemoteControl& GetControl();

    /// Emits a SCIF heartbeat to the downstream peer. When the downstream peer is ready AND it has
    /// heard a heartbeat from this upstream peer, then this upstream peer will call Init() and complete
    /// the SCIF connection.
    void EmitHeartbeat();

    /// This is a thread that handles the stream of returned title deeds from the downstream peer.
    void ReturnDeedThread(PacBio::Process::ThreadedProcessBase& parent);

    /// Returns true if the SCIF connection is completed.
    bool IsConnected() const;

    /// Run a basewriter simulation, instead of actually sending data to a real external basewriter process.
    void RunDownstreamSimulation();

    /// \returns interface if the class supports/requires local copies from remote pool to local pool.
    /// \returns nullptr if the class does not support local copies from the remote pool to the local pool.
    /// An example is the SCIF interface, which requires a DMA copy to bring remote data to the local client. In this
    /// case, this method will return a useful non-nullptr interface, and the various methods exposed by IPoolWithCopies
    /// need to be called for the data to be correctly handled.
    /// The shared memory implementations (PoolShmRealtime and PoolShmXSI) do not require a copy, and this method will
    /// return nullptr.
    PacBio::IPC::IPoolWithCopies* GetIPoolWithCopies()  { return dynamic_cast<PacBio::IPC::IPoolWithCopies*>(pool_.get()); }

    /// allocate the pool object
    /// \param maxReadBuffers - the size of the client size pool. This may or may not be equal to the server side pool
    /// depending on whether the pool is shared memory or not.
    void AllocPool(uint32_t maxReadBuffers);

private:
    void AddFreeOffset(uint64_t offset);
private:
    std::unique_ptr<PacBio::IPC::PoolBaseTyped<ReadBuffer>> pool_;
    PacBio::ThreadSafeQueue<uint32_t> freeReadBufferQueue_;
    static std::vector<int> currentFrames_;
    PacBio::Primary::BaseWriterRemoteControl* baseWriterRemoteControl_;
    SCIF_Heartbeat heartbeat_;
    uint32_t micOffset_;
    PacBio::IPC::MessageSocketReceiver downstreamDeedReturn_;
    bool simulateBasewriter_;
    uint32_t numZMWs_;
    bool sos_;
    ChipClass chip_;
};

}}

#include <pacbio/primary/SCIF_Heartbeat.h>

#include <pacbio/process/ProcessBase.h>
