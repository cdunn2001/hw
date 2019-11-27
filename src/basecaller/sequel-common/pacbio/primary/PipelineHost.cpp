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
/// \brief implementation of the Pipeline Host object which manages communication from the host side


#include <vector>
#include <set>

#include <json/json.h>

#include <pacbio/logging/Logger.h>
#include <pacbio/ipc/PoolBase.h>
#include <pacbio/ipc/Message.h>
#include <pacbio/ipc/JSON.h>
#include <pacbio/dev/AutoTimer.h>
#include <pacbio/POSIX.h>
#include <pacbio/text/String.h>

#include <pacbio/primary/ipc_config.h>
#include <pacbio/primary/TrancheTitle.h>
#include <pacbio/primary/SCIF_Heartbeat.h>
#include <pacbio/primary/PipelineHost.h>

#ifdef SUPPORT_REST
#include "REST_Root.h"
#endif

using namespace PacBio::Primary;
using namespace PacBio::Dev;
using namespace PacBio::IPC;
using namespace PacBio::Text;
using namespace PacBio::Logging;
using namespace PacBio;
using namespace std;

static std::chrono::milliseconds timeout(12000);

namespace PacBio
{
namespace Primary
{
/// constructor for PhiProxy
PhiProxy::PhiProxy(const PacBio::IPC::MessageQueueLabel& label) :
  command_(label) ,
  name_(label.GetFullAddress()),
  ready_(false),
  timestamp_(std::chrono::steady_clock::now() - timeout - timeout),
  pid_(-1)
{
    command_.SetNoLinger();
#if 1
    command_.SetHighWaterMark(10000);
#endif
}

///
void PhiProxy::Send(const PacBio::IPC::Message& mesg)
{
    command_.Send(mesg);
}

/// Time in milliseconds since a heartbeat was received for this Phi
std::chrono::milliseconds PhiProxy::Age() const
{
    return std::chrono::duration_cast<std::chrono::milliseconds>( std::chrono::steady_clock::now() - timestamp_);
}

/// Update the age and PID of this Phi
void PhiProxy::Update(int pid)
{
    UpdateTimestamp();
    pid_  = pid;
}
void PhiProxy::UpdateTimestamp()
{
    timestamp_ = std::chrono::steady_clock::now();
}


////////////////////////////////////////////////////////////////////////////////////////



PipelineHost::PipelineHost(PacBio::Process::ProcessBase& parent, const std::string& direction, const std::vector<uint16_t>& defaultT2bPorts) :
    parent_(parent),
    selfHost_(pa_acqCommandQueue),
    pool_(nullptr),
    direction_(direction),
    defaultT2bPorts_(defaultT2bPorts),
    housekeepingDisabled_(false)

{

    selfHost_.SetNoLinger();

    for(auto& x: mics_)
    {
        micsLookup_.push_back(&x);
    }

}

void PipelineHost::AddExpectedNode(const PacBio::IPC::MessageQueueLabel& label)
{
    labels_.push_back(label);
    mics_.emplace_back(label);
    micsLookup_.push_back(&mics_.back());
}


PhiProxy& PipelineHost::FindPhi(const std::string& name)
{
    for(auto& phi : mics_)
    {
        if (phi.Name() == name) return phi;
    }
    std::stringstream ss;
    ss << "Don't recognize t2b processor: \"" << name << "\"";
    ss << " Expected t2b processors:";
    for(auto& phi : mics_)
    {
        ss << " " << phi.Name();
    }
    throw PBException(ss.str());
}

void PipelineHost::AddHeartbeat(const std::string& data)
{
    SCIF_Heartbeat heartbeat;
    heartbeat.Deserialize(data);
    static std::set<std::string> seen;

    std::lock_guard<std::mutex> l(mutex_);
    {
        try
        {
            PhiProxy& phi = FindPhi(heartbeat.Name());
            if (phi.Age() > timeout || phi.Pid() != heartbeat.PID())
            {
                PBLOG_INFO << "client:" << heartbeat.Name() << " pid:" << heartbeat.PID() << " entered the network (age was " << phi.Age().count() << "ms)";
            }
            phi.Update(heartbeat.PID());

            phi.AddReady(heartbeat.SCIF_Ready() && heartbeat.PipelineReady());
        }
        catch (const std::exception& ex)
        {
            if (seen.find(heartbeat.Name()) == seen.end())
            {
                PBLOG_WARN << "Discarding heartbeat: " << ex.what();
                seen.insert(heartbeat.Name());
            }
            return;
        }
    }
}


void PipelineHost::ReportStatus()
{
    PBLOG_INFO << "clients: present:" << MicPresentCount();
}

uint32_t PipelineHost::MicTotalCount() const
{
    return labels_.size();
}

uint32_t PipelineHost::MicPresentCount() const
{
    std::lock_guard<std::mutex> l(mutex_);
    {
        uint32_t alive = 0;
        for(auto& phi : mics_)
        {
            if (phi.Age() < timeout )
            {
                alive++;
            }
        }
        return alive;
    }
}

bool PipelineHost::IsPipelineAlive() const
{
    return MicPresentCount() == labels_.size();
}

uint32_t PipelineHost::MicReadyCount() const
{
    std::lock_guard<std::mutex> l(mutex_);
    {
        uint32_t alive = 0;
        for(auto& phi : mics_)
        {
            if (phi.IsReady()) alive++;
        }
        return alive;
    }
}

void PipelineHost::CreatePool(const PacBio::IPC::PoolFactory::PoolConfig& poolConfig)
{
    if (poolConfig.numElements() == 0) throw PBException("poolConfig.numElements == 0!");
    if (poolConfig.elementSize() == 0) throw PBException("poolConfig.elementSize == 0!");

    poolConfig_ = poolConfig;
}


void PipelineHost::ConnectSCIFToAllMics()
{
    assert(pool_);


#ifdef DONT_USE_SCIF
    auto root = GetREST_Root();
    root->ModifyString("pipeline/state", "notsupported");
    return;
#else
    for(auto& phi : mics_)
    {
        phi.ClearReady();
    }

    // only do this when all mics are there
    for(auto& s : mics_)
    {
        Json::Value initMessage;
        initMessage["direction"] = direction_;
        initMessage["poolConfig"] = poolConfig_.Json();
        std::string msgJson =  PacBio::IPC::RenderJSON(initMessage);
        PBLOG_INFO << "Sending init message to " << s.Name() << " " << msgJson;
        Announcement msg("init", msgJson);
        s.Send(msg);
    }

    // Waiting for mics to announce themselves
    PBLOG_INFO << "Waiting for " << labels_.size() << " t2b clients ...";
#ifdef SUPPORT_REST
    int counter = 0;
#endif
    double t0 = PacBio::Utilities::Time::GetMonotonicTime();
    double t = PacBio::Utilities::Time::GetMonotonicTime();
    while (pool_->NumConnections() < labels_.size())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (parent_.ExitRequested()) return;
#ifdef SUPPORT_REST
        {
            auto root = GetREST_Root();
            root->ModifyString("pipeline/connections", std::to_string(pool_->NumConnections()));
            root->ModifyString("pipeline/state", "waiting");
            root->ModifyString("pipeline/counter", std::to_string(counter));
            counter++;
        }
#endif
        t = PacBio::Utilities::Time::GetMonotonicTime();
        const double ConnectSCIFToAllMics_TimeoutSeconds = 60.0;
        if (t - t0 > ConnectSCIFToAllMics_TimeoutSeconds)
        {
            PBLOG_WARN << "Elapsed time waiting for connections: " << (t-t0)
                    << " seconds, limit: " << ConnectSCIFToAllMics_TimeoutSeconds;
            throw PBException("Timeout waiting for pool connections");
        }
    }
    PBLOG_INFO << "Elapsed time waiting for connections: " << (t-t0) << " seconds";

#ifdef SUPPORT_REST
    {
        auto root = GetREST_Root();
        root->ModifyString("pipeline/state", "connected");
    }
#endif

    PBLOG_INFO << "waiting for t2b clients to be ready";
#ifdef SUPPORT_REST
    counter = 0;
#endif
    t0 = PacBio::Utilities::Time::GetMonotonicTime();
    while (MicReadyCount() < labels_.size())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (parent_.ExitRequested()) return;
#ifdef SUPPORT_REST
        {
            auto root = GetREST_Root();
            root->ModifyString("pipeline/readys", std::to_string(MicReadyCount()));
            root->ModifyString("pipeline/state", "waitingForReady");
            root->ModifyString("pipeline/counter", std::to_string(counter));
            counter++;
        }
#endif
        t = PacBio::Utilities::Time::GetMonotonicTime();
        const double AllMicsReady_TimeoutSeconds = 60.0;
        if (t - t0 > AllMicsReady_TimeoutSeconds)
        {
            PBLOG_WARN << "Elapsed time waiting for readiness: " << (t-t0)
                    << " seconds, limit: " << AllMicsReady_TimeoutSeconds;
            throw PBException("Timeout waiting for t2b client readiness");
        }
    }
    PBLOG_INFO << "Elapsed time waiting for readiness: " << (t-t0) << " seconds";
    PBLOG_INFO << "All t2b clients are ready";

    // quick ping test. Test response of each MIC.
    for (auto& phi : mics_)
    {
        {
            QuietAutoTimer _(10);
            for (int i = 0; i < 10; i++)
            {
                phi.Send(POST("ping", std::to_string(i)));
            }
            PBLOG_INFO << "Ping rate: " << _.GetRate() << " ping/sec";
        }
    }
#endif
}



void PipelineHost::SendDeed(const std::string& name, PacBio::IPC::TitleBase& title, uint32_t micOffset)
{
    if (pool_) title.BindTo(*pool_);
    Deed deed(name, title);

    if (micsLookup_.size() == 0)
    {
        selfHost_.Send(deed);
    }
    else
    {
        if (micOffset >=  micsLookup_.size() )
        {
            throw PBException("MicOffset is out of range");
        }
#ifdef DONT_USE_SCIF
    {
        static MessageSocketSender hostCommand(pa_acqCommandQueue);

        // bounce deed back to ourselves
        hostCommand.Send(deed);
    }
#else
        micsLookup_[micOffset]->Send(deed);
#endif
    }
}

void PipelineHost::SendMessageDownStream(const PacBio::IPC::Message& message)
{
    for(auto& s : mics_)
    {
        s.Send(message);
    }
}


void PipelineHost::Quit()
{
    for(auto& phi : mics_)
    {
        Announcement msg("quit");
        phi.Send(msg);
    }
}

/// Returns true if the pipeline specification is not empty, that is
/// at least one MIC has been assigned to the pipeline.
bool  PipelineHost::IsEnabled() const
{
    // may want to make this more sophisticated. This doesn't
    // actually check if the mics are online or not, only
    // that the Host is expecting them to be online.
    return MicTotalCount() > 0;
}

/// Connect to all mics through a two stage process.
/// First, wait for all MICs to send at heartbeat to the pipeline host.
/// Once they all all present, then pipeline sends a Init message to each of them,
/// and then waits again for all MICs to complete the SCIF connection.
/// The callback is called every 1.0 seconds, for the purposes of updating status, etc.
void PipelineHost::ConnectToAllMics(std::function<void(void)> f)
{
    if (MicTotalCount() > 0)
    {
        while (!IsPipelineAlive())
        {
            PBLOG_INFO << "Waiting, t2b clients so far: " << MicPresentCount() << "/" << MicTotalCount() << ", all here:" << IsPipelineAlive();
            if (parent_.ExitRequested(std::chrono::milliseconds(1000)))
            {
                throw PBException("ExitRequested");
            }
            f();
        }
        PBLOG_INFO << "All t2b clients seen, connection : " << MicPresentCount() << "/" << MicTotalCount() << ", all here:" << IsPipelineAlive();

        ConnectSCIFToAllMics();
        PBLOG_INFO << MicTotalCount() << " t2b clients connected";
    }
    else
    {
        PBLOG_INFO << "No pipeline t2b clients defined, will not send to pipeline!";
    }
}

/// Add a list of nodes (in string format) to the pipeline host.
/// The format of the nodeList0 is a "name:port,name:port,name:port"
/// For example, "mic0:23610,mic1;23640,mic2:23650". If the
/// port numbers are omitted, a set of default port numbers is used.
/// If nodeList0 is "default" then it is replaced with "mic0,mic1,mic2".

void PipelineHost::AddExpectedNodeList(const std::string& nodeList0)
{
    if (nodeList0 == "") return;

    std::string nodeList = nodeList0;

    const std::map<std::string,std::string>  pipelinePresets =
            {
                    // 3 node lists
                    { "default"  , "mic0,mic1,mic2" },
                    { "3mic"     , "mic0,mic1,mic2" },
                    { "2mic1host", "mic0,mic1,pac-host:2"},
                    { "1mic2host", "mic0,pac-host:1,pac-host:2"},
                    // 2 node lists
                    { "2mic"     , "mic0,mic1" },
                    { "1mic1host", "mic0,pac-host:1"},
                    // 1 node lists
                    { "1mic"     , "mic0" },
                    { "1host"    , "pac-host:0"},
                    { "none"     , ""}
            };
    const auto& f = pipelinePresets.find(nodeList);
    if (f != pipelinePresets.end())
    {
        nodeList = f->second;
    }

    if (nodeList !="")
    {
        vector<string> pipelineNodes = String::Split(nodeList, ',');
        for (string& node : pipelineNodes)
        {
            vector<string> pieces = String::Split(node, ':');
            uint16_t port;
            std::string hostname;
            try
            {
                hostname = pieces[0].c_str();
                if (pieces.size() == 1)
                {
                    uint32_t micOffset = hostname.back() - '0';
                    port = defaultT2bPorts_.at(micOffset);
                }
                else
                {
                    port = static_cast<uint16_t>(std::stoi(pieces[1]));
                    if (port < 10)
                    {
                        port = defaultT2bPorts_.at(port);
                    }
                }
            }
            catch (const std::out_of_range& /*ex*/)
            {
                throw PBException("Problem parsing node descriptor for port number: \"" + node +
                                  "\". Unrecognized host or malformed port number");
            }
            PacBio::IPC::MessageQueueLabel label("", hostname, port);
            AddExpectedNode(label);
        }
    }
}

/// Dumps the state of the pipeline host object to a output stream.
void PipelineHost::DumpState(std::ostream& s)
{
    s << "PipelineHost: IsEnabled:         " << IsEnabled() << std::endl;
    s << "PipelineHost: IsPipelineAlive:   " << IsPipelineAlive() << std::endl;
    s << "PipelineHost: t2b client Expected Count " << MicTotalCount() << std::endl;
    s << "PipelineHost: t2b client  Present Count  " << MicPresentCount() << std::endl;
    s << "PipelineHost: t2b client  Ready Count    " << MicReadyCount() << std::endl;
    int i=0;
    for(auto& phi : mics_)
    {
        s << " PipelineHost.t2b[" << i <<"] name:" << phi.Name();
        s << ", age:" << phi.Age().count() << "ms";
        s << ", pid:" << phi.Pid();
        s << ", Is Ready:" << phi.IsReady() << std::endl;
        i++;
    }
}


bool PipelineHost::CheckHealth(bool* warning)
{
    static int aliveFailures = 0;
    if (warning != nullptr) *warning = false;

    if (housekeepingDisabled_) return true;

    bool pipelineAlive = IsPipelineAlive();
    if (IsEnabled() && !pipelineAlive)
    {
        if (warning != nullptr) *warning = true;
        aliveFailures++;
        PBLOG_WARN << "Pipeline is not responding. (" << aliveFailures << " sequential failures)";
        LogStream logStream;
        DumpState(logStream);
        if (aliveFailures > 200)
        {
            PBLOG_FATAL << "Pipeline is not responding. Aborting process to allow full system restart.";
            return false;
        }
    }
    else
    {
        aliveFailures = 0;
    }
    return true;
}
}}
