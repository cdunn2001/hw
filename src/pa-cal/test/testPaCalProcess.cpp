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
    using PaCalProcess::enableWatchdog_;
    using PaCalProcess::CreateOptionParser;
    using PaCalProcess::HandleLocalOptions;
    using PaCalProcess::Run;
    using PaCalProcess::GetPaCalConfig;
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
    PaCalProcessEx process;
    auto parser = process.CreateOptionParser();
    EXPECT_THAT(parser.description(), HasSubstr("Primary calibration application"));
}

// BENTODO test rework cli
TEST(PaCalProcess,LocalOptions)
{
    PaCalProcessEx process;
    EXPECT_TRUE(process.enableWatchdog_);

    auto parser = process.CreateOptionParser();
    int argc = 3;
    const char* argv[] = {"./binary", "--nowatchdog",
        "--config=platform=Kestrel", nullptr};
    auto options = parser.parse_args(argc, argv);

    process.HandleLocalOptions(options);

    // TODO lift the option handling out of the PaCalProcess class.
    // We shouldn't have to expose
    EXPECT_FALSE(process.enableWatchdog_);
}


TEST(PaCalProcess,DISABLED_Run)
{
    PaCalProcessEx process;
    int exitCode = process.Run();
    EXPECT_EQ(666,exitCode);
}
