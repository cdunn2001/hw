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
///  \brief Defines the pa-ws process, which will launch worker threads and service the main thread


#ifndef PA_WS_PROCESS_H
#define PA_WS_PROCESS_H

#include <pacbio/process/OptionParser.h>
#include <pacbio/process/ProcessBase.h>

#include <app-common/ThreadController.h>

//#include "SraThread.h"
#include "PaWsConfig.h"

namespace PacBio {
namespace Primary {
namespace PaWs {

class WebService;

class PaWsProcess : protected PacBio::Process::ThreadedProcessBase
{
public:
    PaWsProcess();
    ~PaWsProcess() override;

    /// Direct connection from actual main entry point to a member function of this class.
    /// The caller should just map a valid argc and argv from the main entry point to this method.
    /// \param argc - number of command line arguments include the path to the binary
    /// \param argv - command line arguments. Must be nullptr terminated.
    /// \returns the exit code that the process should end with.
    int Main(int argc, const char* argv[]);

    /// This is a ProcessBase override. Exceptions that are caught by the
    /// framework call this function so that the application can do more
    /// with the exception - such as log it, or post it to a remote logging
    /// server. etc.
    void SendException(const std::string& message) override;

    /// See SendException(const std::string&)
    void SendException(const std::exception& ex) override;

    friend class PaWsThreadController;
protected:
    /// Creates the command line parser that will parse argv
    PacBio::Process::OptionParser CreateOptionParser();

    /// Converts the options into local flags
    void HandleLocalOptions(PacBio::Process::Values& options);

    /// Starts the daemon, and when the daemon exits, returns a Linux process exit code.
    /// Values in the range [0,127] are normal exit codes. See ExitCodes.h for the definitions. 
    /// 0 means a successful exit with no errors. This is returned on a normal systemd service shutdown.
    /// Values > 0 indicate some sort of software caught problem.
    /// Values in the range [128,255] indicate the system caught a signal and ended the process. These values
    /// are not returned by this method, but returned by the system when the process exits.
    /// See waitpid for interpretation of values.
    /// \returns process exit code
    int Run();

    /// Launches all of the threads and waits for them to complete.
    /// \returns process exit code
    void RunAllThreads();

    /// \returns a constant reference to the global configuration struct for wx-daemon
    const PaWsConfig& GetPaWsConfig() const { return paWsConfig_; }
protected:
    std::vector<std::string> commandLine_;
//    std::vector<std::unique_ptr<SraThread>> sraThreads_;
    bool enableWatchdog_ = true;
    bool runBist_ = false;
    std::shared_ptr<WebService> webService_;
private:
    PaWsConfig  paWsConfig_;
};

/// I'm playing around with an interface class that allows threads to
/// query or control the class through a minimal API. This may or may not
/// stay.
class PaWsThreadController :  public PacBio::Threading::IThreadController
{
public:
    PaWsThreadController(PaWsProcess& process) : process_(process) {}
    bool ExitRequested() override 
    { 
        return process_.ExitRequested();
    }
    void RequestExit() override
    {
        process_.RequestExit();
    }
    PaWsProcess& process_;
};


}}}

#endif // PA_WS_PROCESS_H
