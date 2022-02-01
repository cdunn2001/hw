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
//
// File Description:
///  \brief Defines the Barrier class used for thread synchronization.

#ifndef BARRIER_H_
#define BARRIER_H_

#if 0
/// This is the boost implementation of the Barrier, which is lacking the
/// .kill() method.
#include <boost/thread/barrier.hpp>

typedef boost::barrier Barrier;
#else

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

#include <pacbio/POSIX.h>
#include <pacbio/PBException.h>
#include <pacbio/utilities/StdThread.h>

/// Based on boost::barrier, but with the kill() method that
/// will instantly unblock the barrier and throw an exception of
/// type Barrier::Exception.
/// Usage (see boost::barrier, this is the same API): 
/// 1. construct the Barrier instance with the number of threads that
/// need to reunite at the barrier. 
/// 2. give each thread a reference to the Barrier instance, and at the 
/// point of reunification, call .wait(). This will block all threads until
/// all threads call .wait() simultaneously.
/// Any thread can call .kill() on the barrier which effectively violently
/// disables the barrier. All threads will throw Barrier::Exception inside
/// the wait call if the barrier is killed, regardless if the kill happens
/// before or after the wait call.  The .kill() method is only used to shut
/// down an application.

class Barrier
{
public:
    /// This is the exception type that will be thrown if .kill() is called.
    class Exception : public std::runtime_error
    {
    public:
        Exception(const Barrier& b) 
           : std::runtime_error(std::runtime_error("Barrier::wait::killed."
               + std::to_string(b.counter_) + "/" 
               + std::to_string(b.limit_) + "/"
               + b.name_) )
               {}
    };

    /// \param limit The count of threads that will meet at the rendezvous
    /// point.
    Barrier(int limit,const std::string& name)
        : counter_(0)
        , limit_(limit)
        , kill_(false)
        , name_(name)
    {
    }

    /// Will block until `limit` threads simultaneously call .wait().
    /// Can be interrupted by a call to .kill(), which will throw
    /// Barrier::Exception.
    void wait()
    {
        PBLOG_DEBUG << "barrier.wait(" << name_ << ") called from \""
            << PacBio::Utilities::StdThread::GetCurrentThreadName() 
            << "\" at " << PacBio::GetBackTrace(3);
        counter_ ++;
        cv_.notify_all();
        while (counter_ < limit_)
        {
            if (kill_)
            {
                PBLOG_WARN << "barrier.wait(" << name_ << ") interrupted in \""
                    << PacBio::Utilities::StdThread::GetCurrentThreadName() 
                    << "\" at " << PacBio::GetBackTrace(5);
                throw Exception(*this);
            }
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait_for(lock,std::chrono::milliseconds(10));
        }
    }
    /// Causes all threads that are blocked by .wait()
    void kill()
    {
        kill_ = true;
        cv_.notify_all();
    }
    friend Exception;
private:
    std::atomic<int> counter_;
    const std::atomic<int> limit_;
    std::atomic<bool> kill_;
    std::condition_variable cv_;
    std::mutex mutex_;
    std::string name_;
};
#endif

#endif
