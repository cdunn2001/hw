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

#include <common/graphs/NodeMonitor.h>

namespace PacBio {
namespace Graphs {


namespace detail {

NodeMonitor::TimingData NodeMonitor::Timings()
{
    std::lock_guard<std::mutex> lm(m_);

    if (threadCount_ != 0) throw PBException("Unexpected report requested while threads are active");
    timings_.idleTime += stateTimer_.GetElapsedMilliseconds();
    started_ = false;

    TimingData ret;
    std::swap(ret, timings_);
    stateTimer_ = FastTimer();
    return ret;
}

void NodeMonitor::IncrementThread()
{
    std::lock_guard<std::mutex> lm(m_);

    auto duration = stateTimer_.GetElapsedMilliseconds();
    stateTimer_ = FastTimer();
    if(threadCount_ == 0)
    {
        if (started_ == true)
        {
            timings_.idleTime += duration;
        }
        else
        {
            started_ = true;
        }
    } else
    {
        timings_.partTime += duration;
    }
    timings_.accumulatedOccupancy += duration*threadCount_;

    threadCount_++;
    if(threadCount_ > maxThreads_) throw PBException("Unexpected thread count in graph node");
}
void NodeMonitor::DecrementThread(float nodeDurationMS)
{
    std::lock_guard<std::mutex> lm(m_);

    auto stateDuration = stateTimer_.GetElapsedMilliseconds();
    stateTimer_ = FastTimer();
    if(threadCount_ == maxThreads_)
    {
        timings_.fullTime += stateDuration;
    } else
    {
        timings_.partTime += stateDuration;
    }
    timings_.accumulatedOccupancy += stateDuration*threadCount_;
    timings_.count++;
    timings_.accumulatedDuration += nodeDurationMS;

    if(threadCount_ == 0) throw PBException("Unexpected thread count in graph node");
    threadCount_--;
}

}}}

