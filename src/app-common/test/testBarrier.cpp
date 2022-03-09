#include <gtest/gtest.h>

#include <future>
#include <atomic>

#include <pacbio/logging/Logger.h>

#include <app-common/Barrier.h>

using namespace testing;
using namespace PacBio::Utilities;

TEST(Barrier,Simple)
{
    // test that two threads stop at a barrier.
    // worker thread increments a counter, and 
    // primary thread checks the counter increments
    // only before and after the barrier is reached.
    std::atomic<int> counter;
    counter = 0;

    Barrier barrier(2,"foo");
    std::condition_variable cv;
    std::mutex mtx;

    auto f1 = std::async(std::launch::async,[&barrier,&cv, &counter](){
        counter++;
        cv.notify_all();
        barrier.wait();
        counter++;
        return counter.load();
    });

    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock);
    EXPECT_EQ(counter, 1);

    barrier.wait();
    
    EXPECT_EQ(2,f1.get());
    EXPECT_EQ(counter, 2);
}

TEST(Barrier,Kill)
{
    // test that a thread stops at a barrier.
    // worker thread increments a counter, and 
    // primary thread checks the counter increments
    // once before the barrier is reached.
    // Then the primary thread kills the barrier,
    // the worker thread throws.

    PacBio::Logging::LogSeverityContext context{PacBio::Logging::LogLevel::FATAL};

    std::atomic<int> counter;
    counter = 0;

    Barrier barrier(2,"foo");
    std::condition_variable cv;
    std::mutex mtx;

    auto f1 = std::async(std::launch::async,[&barrier,&cv,&counter](){
        counter++;
        cv.notify_all();
        barrier.wait();
        counter++;
        return counter.load();
    });

    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock);
    EXPECT_EQ(counter, 1);

    barrier.kill();
    
    EXPECT_THROW(f1.get(), Barrier::Exception);
    EXPECT_EQ(counter, 1);
}
