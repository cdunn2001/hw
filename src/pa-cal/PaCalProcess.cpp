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
#include <pacbio/datasource/DataSourceBase.h>
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
#include "FrameAnalyzer.h"
#include "PaCalConfig.h"
#include "PaCalConstants.h"

// auto generated header files (found in build directory, not source tree)
#include <version.h>
#include <git-rev.h>

#include <PaCalHeader.h>

using namespace std;
using namespace PacBio;
using namespace PacBio::Configuration;
using namespace PacBio::DataSource;
using namespace PacBio::Process;
using namespace PacBio::Logging;
using namespace PacBio::Utilities;
using namespace PacBio::Sensor;

namespace PacBio::Calibration {

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

    parser.add_option("--config").action_append().help("Loads JSON configuration. Can be file name or inline JSON object, e.g. \"{ ... }\"");
    parser.add_option("--showconfig").action_store_true().help("Shows the entire configuration namespace with current values and exits");

    parser.add_option("--nowatchdog").action_store_true().type_bool().set_default(false).help("Disable watchdog");

    parser.add_option("--sra").type_int().set_default(0).help("Which SRA to use when connecting to wxdaemon");
    parser.add_option("--movieNum").type_int().set_default(0).help("The expected movie number, which should agree with what "
                                                                   "comes over the wire via the Wolverine");
    parser.add_option("--numFrames").type_int().set_default(512).help("Number of frames to use during the collection. "
                                                                      "Note: Currently only the specific value of 512 is supported,"
                                                                      "presumably this may be relaxed in the future");

    parser.add_option("--timeoutSeconds").type_double().set_default(60*5).help("pa-cal will self abort if this timeout expires");

    parser.add_option("--inputDarkCalFile").type_string().help("Optional dark cal file to be loaded, necessary for "
                                                               "dynamic loading workflows");

    parser.add_option("--outputFile").type_string().help("Destination file, containing the collected frame mean/variance information");
    return parser;
}

std::optional<PaCalProcess::Settings> PaCalProcess::HandleLocalOptions(PacBio::Process::Values &options)
{
    Settings ret;
    try
    {
        ret.enableWatchdog = ! options.get("nowatchdog");

        std::vector<std::string> cliValidationErrors;
        ret.sra = options.get("sra");
        if (ret.sra < 0) cliValidationErrors.push_back("--sra cannot be negative");

        ret.movieNum = options.get("movieNum");
        if (ret.movieNum < 0) cliValidationErrors.push_back("--movieNum cannot be negative");

        ret.numFrames = options.get("numFrames");
        if (ret.numFrames != 512) cliValidationErrors.push_back("--numFrames currently only accepts a value of 512");

        ret.timeoutSeconds = options.get("timeoutSeconds");
        if (ret.timeoutSeconds <= 0) cliValidationErrors.push_back("--timeoutSeconds must be strictly positive");

        ret.inputDarkCalFile = options["inputDarkCalFile"];

        ret.outputFile = options["outputFile"];
        if (ret.outputFile.empty()) cliValidationErrors.push_back("Must supply value for --outputFile option");

        Json::Value json = MergeConfigs(options.all("config"));
        ret.paCalConfig = PaCalConfig(json);
        auto jsonValidation = ret.paCalConfig.Validate();
        if (jsonValidation.ErrorCount() > 0)
        {
            jsonValidation.PrintErrors();
        }

        for (const auto& err : cliValidationErrors)
        {
            PBLOG_ERROR << err;
        }

        if (cliValidationErrors.size() + jsonValidation.ErrorCount() > 0)
        {
            return {};
        }

        if (options.get("showconfig"))
        {
            std::cout << ret.paCalConfig.Serialize() << std::endl;
            exit(0);
        }

        PBLOG_INFO << ret.paCalConfig.Serialize();
        return ret;
    } catch(std::exception& e)
    {
        PBLOG_ERROR << "Caught exception while parsing options: " << e.what();
    } catch(...)
    {
        PBLOG_ERROR << "Caught unexpected exception type while parsing options";
    }
    return {};

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

std::unique_ptr<DataSourceBase> CreateSource(const PaCalConfig& cfg)
{
    // TODO actually create datasources...
    //      Should be handled by PTSD-1107 and PTSD-1113
    return cfg.source.Visit(
        [](const SimInputConfig& cfg) -> std::unique_ptr<DataSourceBase> { return nullptr; },
        [](const WXIPCDataSourceConfig& cfg) -> std::unique_ptr<DataSourceBase> { return nullptr; }
    );
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

    CreateThread("Analysis", [this, dtc]()
    {
        try
        {
            auto source = CreateSource(settings_.paCalConfig);
            bool success = AnalyzeSourceInput(std::move(source), dtc, settings_.movieNum, settings_.outputFile);
            if (success) PBLOG_INFO << "Main analysis has completed";
            else PBLOG_INFO << "Main analysis not successful";
        } catch (const std::exception& ex)
        {
            PBLOG_ERROR << "Caught exception thrown by analysis thread: " << ex.what();
            PBLOG_ERROR << "Analysis thread will now terminate early";
            dtc->RequestExit();
        }
        PBLOG_INFO << "Analysis Thread Complete";
    });

    PBLOG_INFO << "PaCalProcess waiting to complete analysis";
    Dev::QuietAutoTimer timer;
    while(!ExitRequested())
    {
        std::this_thread::sleep_for(chrono::seconds{1});
        if (timer.GetElapsedMilliseconds() > settings_.timeoutSeconds*1000)
        {
            PBLOG_ERROR << "Timeout limit exceeded, attempting to self-terminate process...";
            RequestExit();
        }
    }

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
        auto settings = HandleLocalOptions(options);
        if (settings.has_value())
        {
            settings_ = settings.value();
            exitCode = Run();
        } else
        {
            exitCode = ExitCode::CommandParsingException;
        }
    }
    // These top level exception handlers should never be called, but are here to prevent an exception leak
    // from calling `terminate()`. They also do not write to the logger.
    catch (const std::system_error &ex)
    {
        if (ex.code().value() != 0)
        {
            std::cerr << "exit_exception at main(): " << ex.what() << ", exit code" << ex.code().value() << endl;
            exitCode = ex.code().value();
            exitCode = ExitCode::StdException;
        }
        else
        {
            exitCode = ExitCode::StdException;
        }
    }
    catch (const std::exception &ex)
    {
        std::cerr << "std::exception caught at main(): " << ex.what() << endl;
        exitCode = ExitCode::StdException;
    }
    catch (...)
    {
        std::cerr << "Unknown Exception caught at main(): " << endl;
        exitCode = ExitCode::DefaultUnknownFailure;
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

} // namespace PacBio::Calibration
