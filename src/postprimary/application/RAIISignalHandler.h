// Copyright (c) 2014-2016, Pacific Biosciences of California, Inc.
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted (subject to the limitations in the
// disclaimer below) provided that the following conditions are met:
//
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
//  * Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
//  * Neither the name of Pacific Biosciences nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE
// GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY PACIFIC
// BIOSCIENCES AND ITS CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL PACIFIC BIOSCIENCES OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
// USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
// OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGE.
//
// \brief An RAII wrapper around a signal handler.

#ifndef PA_WS_RAIISIGNALHANDLER_H
#define PA_WS_RAIISIGNALHANDLER_H

#include <atomic>
#include <signal.h>

/// Installs a signal handler that raises an atomic_bool when the indicated signal is received.
class RAIISignalHandler
{
public:
    RAIISignalHandler(int signum, std::atomic_bool& atomicFlag)
            : signum_(signum)
            , atomicFlag_(atomicFlag)
    {
        Install();
    }
    ~RAIISignalHandler()
    {
        Uninstall();
    }
    void Install()
    {
        if( GetThese().count(signum_) != 0) throw PBException("Can't install more than one signal handler for a "
                                                               " specific signal number");
        GetThese()[signum_] = this;
        new_action_.sa_handler = StaticSignalHandler;
        sigemptyset (&new_action_.sa_mask);
        new_action_.sa_flags = 0 ;
        int ret = sigaction(signum_, &new_action_, &old_action_);
        if (ret !=0) throw PBExceptionErrno("sigaction for SIGPIPE failed");
    }
    void Uninstall()
    {
        sigaction(signum_, &old_action_, NULL);
        GetThese().erase(signum_);
    }
    // Only static function pointers can be installed using sigaction. So we have to take the signal number
    // and reroute it to the object that created the handler.
    static void StaticSignalHandler(int signalNumber)
    {
        auto object = GetThese().find(signalNumber);
        if (object != GetThese().end())
        {
            object->second->SignalHandler(signalNumber);
        }
    }
    void SignalHandler(int signalNumber)
    {
        std::cerr << "signal " << signalNumber << " received" << std::endl;
        atomicFlag_ = true;
    }
private:
    int signum_;
    std::atomic_bool& atomicFlag_;
    struct sigaction new_action_;
    struct sigaction old_action_;
    static std::map<int,RAIISignalHandler*>& GetThese()
    {
        static std::map<int, RAIISignalHandler*> these;
        return these;
    }
};

#endif //PA_WS_RAIISIGNALHANDLER_H
