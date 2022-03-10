#include <gtest/gtest.h>

#include <app-common/Locked.h>
#include <future>

using namespace testing;
using namespace PacBio::Threading;

struct Foo : public PacBio::Threading::Lockable<10>
{
    int value = 0;
};


TEST(Lock,Simple)
{
    // test that 'a' can be locked, written to, and read from
    Foo a;

    {
        auto aa = Locked<Foo>(&a);
        aa->value = 1;
    }
    {
        auto bb = Locked<Foo>(&a);
        EXPECT_EQ(1,bb->value);
    }
}

TEST(Lock,Thread)
{
    // test that 'a' can be accessed by two threads pounding on it
    // with inherently non-atomic operations (+= and -=).  If Locked works,
    // then += and -= will be atomic.
    Foo a;
    {
        auto aa = Locked<Foo>(&a);
        aa->value = 2;
    }
    const int loops = 100'000;
    // these two loops will competitively add or subtract
    // a lot of values to the lock object. Because they are
    // symmetric, the net result should be zero change.
    auto inc = std::async(std::launch::async,[&a,loops](){
        for(int i=0;i<loops;i++)
        {
            auto bb = Locked<Foo>(&a);
            bb->value+= i;
        }
    });
    auto dec = std::async(std::launch::async,[&a,loops](){
        for(int i=0;i<loops;i++)
        {
            auto bb = Locked<Foo>(&a);
            bb->value-= i;
        }
    });
    inc.get();
    dec.get();

    {
        auto aa = Locked<Foo>(&a);
        EXPECT_EQ(2,aa->value);
    }
}

TEST(Lock,Timeout)
{
    // try to lock the same object twice. Second lock should
    // timeout.
    Foo a;

    {
        auto aa = Locked<Foo>(&a);
        aa->value = 2;

        EXPECT_THROW(auto bb = Locked<Foo>(&a), std::exception);
    }
}

TEST(Lock, UnexpectedDestruction)
{
    // This test tries to delete the locked object before it is unlocked.
    // The destructor of the Lockable is supposed to rescue this
    // situation, as destroying std::timed_mutex while locked is UB.
    // ( Annecdotal reports that gcc and clang don't care, while MSVC will
    // terminate() if mutex is destroyed while locked).
    {
        auto a = std::make_unique<Foo>();

        auto aa = Locked<Foo>(a.get());
        aa->value = 2;

        EXPECT_TRUE(aa);

        // destroy a before the Locked is destroyed. Make sure this doesn't throw
        // or seg fault.
        a.reset();

        EXPECT_FALSE(aa);

        // the underlying object is destroyed. The Locked instance is now
        // pointing to nothing, so operator-> needs to throw.
        EXPECT_THROW(aa->value = 3,std::exception);

        // now the aa goes out of scope, and should not throw or seg fault.
    }
}
