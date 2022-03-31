#include <gtest/gtest.h>

#include <app-common/ProgressMessage.h>
#include <pacbio/utilities/SmartEnum.h>

using namespace testing;
using namespace PacBio::IPC;

TEST(ProgressMessage,A)
{
    SMART_ENUM(Stage, A, B, C);
    ProgressMessage<Stage>::Table table = {
        { "A",    { false, 0,  1 } },
        { "B",    { false, 1,  9 } },
        { "C",    {  true, 2, 80 } }
    };

    ProgressMessage<Stage> pm(table,"foo",1);
    uint64_t counterMax = 10;
    double timeoutForNextStatus = 10.0;

    int globalThing = 1;

    ProgressMessage<Stage>::ThreadSafeStageReporter reporter(&pm, Stage::A, counterMax, timeoutForNextStatus, 
        [&globalThing](Json::Value& metrics){
            metrics["foo"] = globalThing;
        });

    {
        auto metrics = reporter.GetMetrics();
        (*metrics)["foo"] = 100;
    }
    reporter.Flush();

    reporter.Update(1);
}

