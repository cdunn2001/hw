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
///  \brief Defines the pa-cal process

#include "PaCalProcess.h"

#include <memory>

#include <json/json.h>

// library includes
#include <pacbio/configuration/MergeConfigs.h>
#include <pacbio/logging/Logger.h>
#include <pacbio/POSIX.h>
#include <pacbio/process/OptionParser.h>
#include <pacbio/process/ProcessBase.h>
#include <pacbio/sensor/Platform.h>
#include <pacbio/text/String.h>
#include <pacbio/utilities/Finally.h>
#include <pacbio/utilities/StdThread.h>

// local includes
#include "ExitCodes.h"
#include "PaCalConfig.h"
#include "PaCalConstants.h"

// auto generated header files (found in build directory, not source tree)
#include <version.h>
#include <git-rev.h>

#include <PaCalHeader.h>

using namespace std;
using namespace PacBio;
using namespace PacBio::Configuration;
using namespace PacBio::Process;
using namespace PacBio::Logging;
using namespace PacBio::Utilities;
using namespace PacBio::Sensor;

namespace PacBio {
namespace Primary {
namespace Calibration {

PaCalProcess::PaCalProcess()
  : paCalConfig_(Platform(Platform::Kestrel))
  {

  }

PaCalProcess::~PaCalProcess()
{
    Abort();
    Join();
}  
  
OptionParser PaCalProcess::CreateOptionParser()
{
    OptionParser parser = ProcessBase::OptionParserFactory();
    std::stringstream ss;
    ss << "Primary calibration application\n\n"
       << PacBio::PaCalHeader::cmakeBuildType << " build"
       << "\n git branch: " << cmakeGitBranch()
       << "\n git hash: " << cmakeGitHash()
       << "\n git commit date: " << cmakeGitCommitDate();
    parser.description(ss.str());
    parser.version(std::string(SHORT_VERSION_STRING) + "." + cmakeGitHash());

    parser.epilog("Ports: \n"
    );

    parser.add_option("--config").action_append().help("Loads JSON configuration. Can be file name or inline JSON object, e.g. \"{ ... }\"");
    parser.add_option("--strict").action_store_true().help("Strictly check all configuration options. Do not allow unrecognized configuration options");
    parser.add_option("--showconfig").action_store_true().help("Shows the entire configuration namespace with current values and exits");
    parser.add_option("--listports").action_store_true().help("Echoes all network ports to the console and exits.");

    const std::string execPath = PacBio::POSIX::GetCurrentExecutablePath();

    parser.add_option("--nowatchdog").action_store_true().type_bool().set_default(false).help("Disable watchdog");

    return parser;
}

void PaCalProcess::HandleLocalOptions(PacBio::Process::Values &options)
{
    enableWatchdog_ = ! options.get("nowatchdog");

    Json::Value json = MergeConfigs(options.all("config"));
    PBLOG_DEBUG << json; // this does NOT work with --showconfig
    paCalConfig_ = PaCalConfig(json);
    FactoryConfig(&paCalConfig_);
    paCalConfig_.Update(json);
    auto validation = paCalConfig_.Validate();
    if (validation.ErrorCount() > 0)
    {
        validation.PrintErrors();
        throw PBException("Json validation failed");
    }

    if (options.get("showconfig"))
    {
        std::cout << paCalConfig_.Serialize() << std::endl;
        exit(0);
    }

    if (options.get("listports"))
    {
        std::cout << std::endl;
        exit(0);
    }

#if 0
// TODO fix logging. I want to see the thread ID (or preferably a symbolic thread name)
// along with the rest of the default columns. This experiment failed in a big way.
// this makes a mess. Somehow the output of the logger is getting captured by the stdout capture class, and then 
// recursively relogging every line.
    const std::string DEFAULT_PB_LOG_FORMAT = ">|> %TimeStamp% -|- %Severity% -|- %Channel% -|- %HostName%|P:%PBProcessID%|T:%PBThreadID% -|- %Message%";
    const std::string DEFAULT_PB_LOG_FILE_SETTINGS = "[Sinks.MySink]\nDestination=Console\nAutoFlush=true\nAsynchronous=true\nFormat=\"" + DEFAULT_PB_LOG_FORMAT + "\"\n";
//    const std::string DEFAULT_PB_LOG_FILE_SETTINGS = "[Sinks.TextFile]\nDestination=TextFile\nAutoFlush=true\nAsynchronous=true\nFormat=\"" + DEFAULT_PB_LOG_FORMAT + "\"\n";

     std::stringstream settings;
     settings << DEFAULT_PB_LOG_FILE_SETTINGS;
     boost::log::init_from_stream(settings);
#endif
}

void PaCalProcess::RunAllThreads()
{
#if 0
    {
        PacBio::Logging::LogStream ls;
        DisplayAffinity(ls);
    }
#endif

    std::shared_ptr<PaCalThreadController> dtc =
        std::make_shared<PaCalThreadController>(*this);

    PBLOG_INFO << "PaCalProcess entering event loop";
    while(!ExitRequested())
    {
        PacBio::POSIX::Sleep(1.0);
    }
    PBLOG_INFO << "PaCalProcess exiting event loop.";

    PBLOG_INFO << "Joining...";
    Join();
    PBLOG_INFO << "All threads joined";
}

int PaCalProcess::Run()
{
    int exitCode = ExitCode::DefaultUnknownFailure;

    Console::SetWindow(400, 400);

    ThreadedProcessBase::SetXtermTitle("pa-cal");

    PBLOG_INFO << "pa-cal: Version " << VERSION_STRING << " CMAKE_BUILD_TYPE:" << PacBio::PaCalHeader::cmakeBuildType;
    PBLOG_INFO << "git branch: " << cmakeGitBranch();
    PBLOG_INFO << "git commit hash: " << cmakeGitHash();
    PBLOG_INFO << "git commit date: " << cmakeGitCommitDate();
    for (const auto &file : PacBio::Text::String::Split(cmakeGitStatus(), '\n'))
    {
        PBLOG_INFO << "git status: " << file;
    }

    stringstream ss;
    for (auto arg : commandLine_)
    {
        ss << arg << " ";
    }
    PBLOG_INFO << "command line: " << ss.str();
    PBLOG_INFO << "Process Id:" << getpid();

    PBLOG_TRACE << "Testing 'trace' level logging";
    PBLOG_DEBUG << "Testing 'debug' level logging";
    PBLOG_INFO << "Testing 'info' level logging";
    try
    {
        RunAllThreads();
        exitCode = ExitCode::DefaultUnknownFailure;
        PBLOG_INFO << "main: RunAllThreads() normal exit, code:" << exitCode;
    }
    catch (const std::exception& ex)
    {
        PBLOG_ERROR << "main: fatal exception: " << ex.what();
        exitCode = ExitCode::StdException;
    }
    catch (...)
    {
        PBLOG_ERROR << "main: fatal uncaught exception";
        exitCode = ExitCode::UncaughtException;
    }

    try
    {
        auto exitCode1 = Join();
        if (exitCode == ExitCode::DefaultUnknownFailure) exitCode = exitCode1;
        PBLOG_DEBUG << "main: main event loop exit, code:" << exitCode1
             << " pid:" << POSIX::GetPid();
    }
    catch (const std::exception& ex)
    {
        exitCode = ExitCode::StdException;
        PBLOG_ERROR << "main: exception2: " << ex.what();
    }
    catch (...)
    {
        exitCode = ExitCode::UncaughtException;
        PBLOG_ERROR << "main: uncaught exception2";
    }


    PBLOG_DEBUG << "main: exit code:" << exitCode;
    return exitCode;
}

int PaCalProcess::Main(int argc, const char *argv[])
{
    int exitCode = ExitCode::DefaultUnknownFailure; // this the app hasn't decided what the exit code should be
    try
    {
        // save the command line for later
        for (const char **arg = argv; *arg; arg++)
            commandLine_.push_back(*arg);

        auto parser = CreateOptionParser();
        Values &options = parser.parse_args(argc, argv);
        vector<string> args = parser.args();
        HandleGlobalOptions(options);
        HandleLocalOptions(options);
        exitCode = Run();
    }
    // These top level exception handlers should never be called, but are here to prevent an exception leak
    // from calling `terminate()`. They also do not write to the logger.
    catch (const std::system_error &ex)
    {
        if (ex.code().value() != 0)
        {
            std::cerr << "exit_exception at main(): " << ex.what() << ", exit code" << ex.code().value() << endl;
            exitCode = ex.code().value();
        }
        else
        {
            exitCode = ExitCode::StdException;
        }
    }
    catch (const std::exception &ex)
    {
        std::cerr << "std::exception caught at main(): " << ex.what() << endl;
        exitCode = ExitCode::CommandParsingException;
    }
    catch (...)
    {
        std::cerr << "Unknown Exception caught at main(): " << endl;
        exitCode = ExitCode::CommandParsingException;
    }
    return exitCode;
}

void PaCalProcess::SendException(const std::string& message)
{
    // the message is also caught at a separate try/catch block, so
    // debug level is ok for this message.
    PBLOG_ERROR << "PaCalProcess Exception message caught:" << message;
}

void PaCalProcess::SendException(const std::exception& ex)
{
    // the message is also caught at a separate try/catch block, so
    // debug level is ok for this message.
    PBLOG_DEBUG << "PaCalProcess std::exception caught:" << ex.what();
}

}}} //namespace
