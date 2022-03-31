#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <app-common/ProgressMessage.h>
#include <pacbio/POSIX.h>
#include <pacbio/process/StdioHandler.h>
#include <pacbio/utilities/SmartEnum.h>

using namespace testing;
using namespace PacBio::IPC;

TEST(ProgressMessage,A)
{
    std::stringstream ss;
    // Grab all stderr to the ss stream, for later processing.
    PacBio::Process::StdErrHandler grabber([&ss](const char* text){
        ss << text;
    });

    SMART_ENUM(Stage, A, B, C);
    ProgressMessage<Stage>::Table table = {
        { "A",    { false, 0,  1 } },
        { "B",    { false, 1,  9 } },
        { "C",    {  true, 2, 80 } }
    };

    ProgressMessage<Stage> pm(table,"GOODMORNING", 2 /* should write to stderr */);
    uint64_t counterMax = 10;
    double timeoutForNextStatus = 10.0;

    int globalThing = 1;

    ProgressMessage<Stage>::ThreadSafeStageReporter reporter(&pm, Stage::A, counterMax, timeoutForNextStatus, 
        [&globalThing](Json::Value& metrics){
            metrics["foo"] = globalThing;
        });
    reporter.SetMinimumInterval(1 /*ms*/);

    globalThing = 100;
    PacBio::POSIX::Sleep(0.010);
    reporter.Update(1);
    fflush(stderr);
    PacBio::POSIX::Sleep(0.100);

    const auto capturedStdout = ss.str();
    EXPECT_THAT(capturedStdout, ::testing::HasSubstr(R"(GOODMORNING {"counter":0,"counterMax":10,"metrics":{"foo":1})"));
    EXPECT_THAT(capturedStdout, ::testing::HasSubstr(R"(GOODMORNING {"counter":1,"counterMax":10,"metrics":{"foo":100})"));
}

