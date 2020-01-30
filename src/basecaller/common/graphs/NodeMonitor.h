// Copyright (c) 2020, Pacific Biosciences of California, Inc.
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

#ifndef PACBIO_GRAPHS_NODE_MONITOR_H
#define PACBIO_GRAPHS_NODE_MONITOR_H

#include <mutex>

#include <pacbio/PBException.h>
#include <pacbio/dev/profile/ScopedProfilerChain.h>

namespace PacBio {
namespace Graphs {
namespace detail {

// Internal use class used to monitor the performance of
// PBGraphNodes.  Should only be used by the graph node
// classes themselves.  These would not even be visible
// in a header file except the graphs implementations are
// heavily templated and cannot be pushed to a .cpp file.
struct NodeMonitor
{
    using FastTimer = PacBio::Dev::Profile::FastTimer;
    // Dumb data bundle struct, to keep track of how long
    // a node was in a given state
    struct TimingData
    {

        TimingData& operator+=(const TimingData& t)
        {
            count += t.count;
            idleTime += t.idleTime;
            partTime += t.partTime;
            fullTime += t.fullTime;
            avgOccupancy += t.avgOccupancy;
            avgDuration += t.avgDuration;

            return *this;
        }

        int count = 0;             // Invcation count
        float idleTime = 0;        // Time spent idle
        float partTime = 0;        // Time spent doing work, but at less than full occupancy
        float fullTime = 0;        // Time spent at full occupancy
        // running sums for average occpancy and average serial execution.  Note these
        // are misnomers, and are not trutly average until properly normalized
        float avgOccupancy = 0.0f;
        float avgDuration = 0.0f;
    };

    // RAII class intended to be created automatically at the start of invocation
    // for a graph node.  Automatically handles the necessary Increment/DecrementThread
    // calls
    class Monitor
    {
    public:
        Monitor(NodeMonitor* node)
            : node_(node)
        {
            node_->IncrementThread();
        }

        Monitor(const Monitor&) = delete;
        Monitor(Monitor&& o) {node_ = o.node_; o.node_ = nullptr;}
        Monitor& operator=(const Monitor&) = delete;
        Monitor& operator=(Monitor&& o) {node_ = o.node_; o.node_ = nullptr; return *this;}

        ~Monitor()
        {
            if (node_)
            {
                node_->DecrementThread(timer_.GetElapsedMilliseconds());
            }
        }

    private:
        NodeMonitor* node_;
        FastTimer timer_;
    };

    NodeMonitor(int maxThreads)
        : maxThreads_(maxThreads)
    {}

    // Generate a report.  This should only be called
    // when the owning node is idle (no active threads)
    TimingData Timings();

    // Creates a Monitor instance to keep track of the current
    // invocation.
    Monitor StartScope()
    {
        return Monitor(this);
    }
private:
    void IncrementThread();
    void DecrementThread(float nodeDurationMS);

    bool started_ = false;
    FastTimer stateTimer_;
    std::mutex m_;
    int threadCount_ = 0;
    int maxThreads_;

    TimingData timings_;
};

}}}

#endif //PACBIO_GRAPHS_NODE_MONITOR_H
