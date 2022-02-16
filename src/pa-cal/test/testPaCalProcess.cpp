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

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <pacbio/dev/gtest-extras.h>
#include <pa-cal/PaCalProcess.h>
#include <pa-cal/ExitCodes.h>

using namespace PacBio::Calibration;
using namespace testing;

/// This class exposes protected members for the purpose of white box testing.
class PaCalProcessEx : public PaCalProcess
{
public:
    using PaCalProcess::PaCalProcess;
    using PaCalProcess::Run;
};


TEST(PaCalProcess,Help)
{
    PaCalProcess process;
    int argc = 2;
    const char* argv[] = {"./binary", "--help", nullptr};
    EXPECT_EXIT(process.Main(argc, argv), ExitedWithCode(0), "");
}

TEST(PaCalProcess,Version)
{
    PaCalProcess process;
    int argc = 2;
    const char* argv[] = {"./binary", "--version", nullptr};
    EXPECT_EXIT(process.Main(argc, argv), ExitedWithCode(0), "");
}

TEST(PaCalProcess,BadArg)
{
    PaCalProcess process;
    int argc = 2;
    const char* argv[] = {"./binary", "--badArg", nullptr};
    EXPECT_EXIT(process.Main(argc, argv),
        ExitedWithCode(2),
        R"(error: no such option: --badArg)");
}

TEST(PaCalProcess,OptionParser)
{
    auto parser = PaCalProcess::CreateOptionParser();
    EXPECT_THAT(parser.description(), HasSubstr("Primary calibration application"));
}

TEST(PaCalProcess,LocalOptions)
{
    PacBio::Logging::LogSeverityContext ls(PacBio::Logging::LogLevel::FATAL);
    auto parser = PaCalProcess::CreateOptionParser();

    auto parseCli = [&parser](std::vector<const char*> args)
    {
        args.insert(args.begin(), "./binary");
        args.insert(args.end(), nullptr);
        auto options = parser.parse_args(args.size()-1, args.data());

        return PaCalProcess::HandleLocalOptions(options);
    };

    auto settings = parseCli({"--nowatchdog"});
    // We've failed to set a mandatory argument
    EXPECT_FALSE(settings.has_value());

    settings = parseCli({"--nowatchdog", "--outputFile=location"});
    EXPECT_TRUE(settings.has_value());
    if (settings.has_value())
    {
        EXPECT_FALSE(settings->enableWatchdog_);
        EXPECT_EQ(settings->outputFile_, "location");
    }

    settings = parseCli({"--nowatchdog", "--outputFile=location", "--movieNum=-4"});
    // Now we've set things with an invalid value
    EXPECT_FALSE(settings.has_value());

    // A more comprehensive setting
    settings = parseCli({"--nowatchdog", "--outputFile=location", "--movieNum=4",
                         "--sra=2", "--timeoutSeconds=80.2"});
    EXPECT_TRUE(settings.has_value());
    if (settings.has_value())
    {
        EXPECT_FALSE(settings->enableWatchdog_);
        EXPECT_EQ(settings->outputFile_, "location");
        EXPECT_EQ(settings->movieNum_, 4);
        EXPECT_EQ(settings->sra_, 2);
        EXPECT_DOUBLE_EQ(settings->timeoutSeconds_, 80.2);
    }
}


TEST(PaCalProcess,DISABLED_Run)
{
    PaCalProcessEx process;
    int exitCode = process.Run();
    EXPECT_EQ(666,exitCode);
}
