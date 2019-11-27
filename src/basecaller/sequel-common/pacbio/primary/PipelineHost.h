// Copyright (c) 2017, Pacific Biosciences of California, Inc.
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
// Description:
/// \brief delcaration of the Pipeline Host object which manages communication from the host side

#ifndef PA_ACQUISITION_PIPELINEHOST_H
#define PA_ACQUISITION_PIPELINEHOST_H

#include <vector>
#include <string>
#include <iostream>
#include <chrono>

#include <pacbio/ipc/MessageQueue.h>
#include <pacbio/process/ProcessBase.h>
#include <pacbio/primary/Tile.h>
#include <pacbio/primary/IPipelineHost.h>

namespace PacBio
{
 namespace IPC
 {
  class PoolBase;

  class Message;
 }
 namespace Primary
 {


/// A class object that represents the state of a particular MIC
class PhiProxy
{
public:
    PhiProxy(const PacBio::IPC::MessageQueueLabel& label);

    std::chrono::milliseconds Age() const;
    void Send(const PacBio::IPC::Message& mesg);
    std::string Name() const { return name_;}
    void Update(int pid0);
    void AddReady(bool flag) { ready_ = flag;}
    void ClearReady() { ready_ = false;}
    bool IsReady() const {return ready_;}
    int Pid() const {return pid_;}
    void UpdateTimestamp();

private:
    PacBio::IPC::MessageSocketSender command_;
    std::string name_;
    bool ready_;
    std::chrono::steady_clock::time_point timestamp_;
    int pid_;
};

class PipelineHost : public IPipelineHost
{
public:
    class SuspensionContext
    {
    public:
        SuspensionContext(PipelineHost&  host) : host_(host)
        {
            host_.housekeepingDisabled_ = true;
        }
        ~SuspensionContext()
        {
            host_.housekeepingDisabled_ = false;
            for(auto&& phi : host_.mics_)
            {
                phi.UpdateTimestamp();
            }
        }
    private:
        PipelineHost&  host_;
    };
public:
    PipelineHost(PacBio::Process::ProcessBase& parent, const std::string& direction, const std::vector<uint16_t>& defaultT2bPorts);
    void AddExpectedNode(const PacBio::IPC::MessageQueueLabel& label);
    void CreatePool(const PacBio::IPC::PoolFactory::PoolConfig& poolConfig);

    bool     IsPipelineAlive() const;    ///<! all Mics are present and heartbeats are current (not stale)
    uint32_t MicPresentCount() const ; ///<! present due to heartbeat
    uint32_t MicReadyCount() const;    ///<! ready due to SCIF registration
    void     ConnectToAllMics(std::function<void(void)> f);

    void AddHeaderToPipeline(PacBio::Primary::Tile* headerTile);

    void AddHeartbeat(const std::string& data); // phi zmq is alive
    void AddReady(const std::string& data); // phi SCIF pool is connected
    void ReportStatus();
    void Quit();
    void AssignPool(PacBio::IPC::PoolBase* pool) { pool_ = pool;}
    void AddExpectedNodeList(const std::string& nodeList);
    void DumpState(std::ostream& s);
    bool CheckHealth(bool* warning = nullptr);

    // overrides
    void SendDeed(const std::string& name, PacBio::IPC::TitleBase& title, uint32_t micOffset) override;
    void SendMessageDownStream(const PacBio::IPC::Message& message) override;
    PacBio::IPC::PoolBase& GetPool() override { return *pool_;}
    bool IsEnabled() const override;
    uint32_t MicTotalCount() const override;
    const std::vector<PacBio::IPC::MessageQueueLabel>& Labels() const { return labels_; }
private:
    void     ConnectSCIFToAllMics();
    PhiProxy& FindPhi(const std::string& name);

private:
    PacBio::Process::ProcessBase& parent_;
    std::vector<PacBio::IPC::MessageQueueLabel> labels_;               ///< list of all active Phi command queue labels
    std::list<PhiProxy> mics_;             ///< list of all active Phi command sockets
    std::vector<PhiProxy*> micsLookup_;
    PacBio::IPC::MessageSocketSender selfHost_;
    mutable std::mutex mutex_;
    PacBio::IPC::PoolBase* pool_;
    std::string direction_;
    const std::vector<uint16_t>& defaultT2bPorts_;
    std::atomic<bool> housekeepingDisabled_;
    friend class SuspensionContext;
    PacBio::IPC::PoolFactory::PoolConfig poolConfig_;
};
 }
}


#endif //PA_ACQUISITION_PIPELINEHOST_H
