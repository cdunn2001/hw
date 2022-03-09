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
///  \brief Defines the pa-cal process, which will launch worker threads and service the main thread


#ifndef KES_CAL_PROCESS_H
#define KES_CAL_PROCESS_H

#include <optional>

#include <pacbio/process/OptionParser.h>
#include <pacbio/process/ProcessBase.h>

#include <app-common/ThreadController.h>

#include <pa-cal/PaCalConfig.h>
#include <pa-cal/ExitCodes.h>

namespace PacBio::Calibration {

class WebService;

class PaCalProcess : public PacBio::Process::ThreadedProcessBase
{
public:
    struct Settings
    {
        bool enableWatchdog = true;
        bool createDarkCalFile = true;
        PaCalConfig  paCalConfig;
        int32_t sra = 0;
        int32_t movieNum = 0;
        int32_t numFrames = 0;
        double timeoutSeconds = 0.0;
        std::string inputDarkCalFile;
        std::string outputFile;
    };

public:
    PaCalProcess() = default;
    ~PaCalProcess() override;

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

    /// Creates the command line parser that will parse argv
    static PacBio::Process::OptionParser CreateOptionParser();

    /// Converts the options into a PaCalProcess::Settings struct.  Any validation
    /// errors will be logged.
    ///
    /// \param options a Values class holding the CLI options
    /// \return std::optional<Settings> that is either empty if there were
    ///         any parsing/validation errors, and otherwise containing a
    ///         struct with the values extracted from the options input
    static std::optional<Settings> HandleLocalOptions(PacBio::Process::Values& options);

protected:
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
    int RunAllThreads();
private:
    std::vector<std::string> commandLine_;
    Settings settings_;
};

} // namespace PacBio::Calibration

#endif // PA_WS_PROCESS_H
