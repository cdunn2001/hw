// Copyright (c) 2020, Pacific Biosciences of California, Inc.
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

#include <pacbio/utilities/SmartEnum.h>

#include <common/graphs/GraphManager.h>
#include <common/graphs/GraphNode.h>
#include <common/graphs/GraphNodeBody.h>

#include <gtest/gtest.h>

using namespace PacBio::Graphs;

namespace {

// Just a dummy enum to satisfy the API, we don't
// really care about the performance reporting in
// these tests
// Intel compiler complains about unused features of the smart enum for some reason.
#pragma warning disable 177
SMART_ENUM(STAGES, ONE, TWO, THREE, FOUR, FIVE);

struct CountingLeaf : public LeafBody<int>
{
    CountingLeaf(int* sum)
        : sum_(sum)
    {}

    size_t ConcurrencyLimit() const override { return 1; }
    float MaxDutyCycle() const override { return 1.0; }

    void Process(int val) override
    {
        *sum_ += val;
    }

private:
    int* sum_;
};

struct QuadraticTransform : public TransformBody<int, int>
{
    size_t ConcurrencyLimit() const override { return 1; }
    float MaxDutyCycle() const override { return 1.0; }

    int Process(int val) override
    {
        return val*val;
    }
};

struct SubtractMin : public MultiTransformBody<int, int>
{
    SubtractMin(size_t window)
        : window_(window)
    {}

    size_t ConcurrencyLimit() const override { return 1; }
    float MaxDutyCycle() const override { return 1.0; }

    void Process(int val) override
    {
        data_.push_back(val);
        if (data_.size() == window_)
        {
            auto min = *std::min_element(data_.begin(), data_.end());
            for (auto val1 : data_) PushOut(val1 - min);
            data_.resize(0);
        }
    }
private:
    size_t window_;
    std::vector<int> data_;
};

struct RAIILeaf : public LeafBody<int>
{
    RAIILeaf() { createCount++; }
    size_t ConcurrencyLimit() const override { return 1; }
    float MaxDutyCycle() const override { return 1.0; }

    void Process(int val) override { (void) val; }

    ~RAIILeaf() { destructCount++; }

    static int destructCount;
    static int createCount;
};
int RAIILeaf::destructCount = 0;
int RAIILeaf::createCount = 0;

struct RAIITransform : public TransformBody<int, int>
{
    RAIITransform() { createCount++; }
    size_t ConcurrencyLimit() const override { return 1; }
    float MaxDutyCycle() const override { return 1.0; }

    int Process(int val) override { return val; }

    ~RAIITransform() { destructCount++; }

    static int destructCount;
    static int createCount;
};
int RAIITransform::destructCount = 0;
int RAIITransform::createCount = 0;

struct RAIIMultiTransform : public MultiTransformBody<int, int>
{
    RAIIMultiTransform() { createCount++; }
    size_t ConcurrencyLimit() const override { return 1; }
    float MaxDutyCycle() const override { return 1.0; }

    void Process(int val) override { this->PushOut(val); }

    ~RAIIMultiTransform() { destructCount++; }

    static int destructCount;
    static int createCount;
};
int RAIIMultiTransform::destructCount = 0;
int RAIIMultiTransform::createCount = 0;

// Uses a leaf node to sum a vector
TEST(GraphAPI, LeafNode)
{
    int sum = 0;

    GraphManager<STAGES> graph;
    auto* inputNode = graph.AddNode(std::make_unique<CountingLeaf>(&sum), STAGES::ONE);

    std::array<int, 8> vals {1,4,67,3,7,4,5,23};
    for (auto val : vals) inputNode->ProcessInput(val);

    graph.Synchronize();
    EXPECT_EQ(std::accumulate(vals.begin(), vals.end(), 0), sum);
}

// Adds a transform node to the previous setup, to sum the
// square of a vector
TEST(GraphAPI, TransformNode)
{
    int sum = 0;

    GraphManager<STAGES> graph;
    auto* inputNode = graph.AddNode(std::make_unique<QuadraticTransform>(), STAGES::ONE);
    inputNode->AddNode(std::make_unique<CountingLeaf>(&sum), STAGES::TWO);

    std::array<int, 8> vals {1,4,67,3,7,4,5,23};
    for (auto val : vals) inputNode->ProcessInput(val);

    graph.Synchronize();
    EXPECT_EQ(std::accumulate(vals.begin(), vals.end(), 0, [](int sum, int val) { return sum += val*val; }), sum);
}

// Adds a multitransform node to the previous setup.  Now
// sums the square of a vector after each element has had the
// minimum element subtracted away
TEST(GraphAPI, MultiTransformNode)
{
    int result = 0;

    GraphManager<STAGES> graph;
    auto* inputNode = graph.AddNode(std::make_unique<SubtractMin>(8), STAGES::ONE);
    inputNode->AddNode(std::make_unique<QuadraticTransform>(), STAGES::TWO)
             ->AddNode(std::make_unique<CountingLeaf>(&result), STAGES::THREE);

    std::array<int, 8> vals {1,4,67,3,7,4,5,23};
    for (int i = 0; i < 7; ++i)
        inputNode->ProcessInput(vals[i]);

    graph.Synchronize();
    EXPECT_EQ(0, result);

    inputNode->ProcessInput(vals[7]);
    graph.Synchronize();

    auto min = *std::min_element(vals.begin(), vals.end());
    auto equation = [min](int sum, int val) { val -= min; return sum += val*val; };
    EXPECT_EQ(std::accumulate(vals.begin(), vals.end(), 0, equation), result);
}

// The graph manager assumes ownership of all nodes/bodies involved.
// This checks to make sure it's properly releasing resources
TEST(GraphAPI, RAIICheck)
{
    RAIILeaf::createCount = 0;
    RAIILeaf::destructCount = 0;
    RAIITransform::createCount = 0;
    RAIITransform::destructCount = 0;
    RAIIMultiTransform::createCount = 0;
    RAIIMultiTransform::destructCount = 0;
    {
        GraphManager<STAGES> graph;
        graph.AddNode(std::make_unique<RAIIMultiTransform>(), STAGES::ONE)
            ->AddNode(std::make_unique<RAIITransform>(), STAGES::TWO)
            ->AddNode(std::make_unique<RAIILeaf>(), STAGES::THREE);
    }

    EXPECT_EQ(RAIILeaf::createCount, 1);
    EXPECT_EQ(RAIILeaf::destructCount, 1);
    EXPECT_EQ(RAIITransform::createCount, 1);
    EXPECT_EQ(RAIITransform::destructCount, 1);
    EXPECT_EQ(RAIIMultiTransform::createCount, 1);
    EXPECT_EQ(RAIIMultiTransform::destructCount, 1);
}

TEST(GraphAPI, MoveOnlyTypes)
{
    // Dummy type, to force move-only-semantics, along with an
    // added guard so compute stages can detect if they received
    // an invalid/empty post-move object as input
    struct MoveOnly
    {
        MoveOnly() = default;
        MoveOnly(const MoveOnly&) = delete;
        MoveOnly(MoveOnly&& other)
        {
            valid_ = other.valid_;
            other.valid_ = false;
        }
        MoveOnly& operator=(const MoveOnly&) = delete;
        MoveOnly& operator=(MoveOnly&& other)
        {
            valid_ = other.valid_;
            other.valid_ = false;
            return *this;
        }
        ~MoveOnly() = default;

        operator bool() const { return valid_; }

    private:
        bool valid_ = true;
    };

    // First double check the MoveOnly class' validity checks work
    {
        MoveOnly m1;
        ASSERT_TRUE(m1);

        MoveOnly m2(std::move(m1));
        EXPECT_FALSE(m1);
        EXPECT_TRUE(m2);

        m1 = std::move(m2);
        EXPECT_TRUE(m1);
        EXPECT_FALSE(m2);
    }

    struct MoveLeaf : public LeafBody<MoveOnly>
    {
        size_t ConcurrencyLimit() const override { return 1; }
        float MaxDutyCycle() const override { return 1.0; }

        void Process(MoveOnly val) override {
            EXPECT_TRUE(val);
        }
    };

    struct MoveLeafConst : public LeafBody<const MoveOnly>
    {
        size_t ConcurrencyLimit() const override { return 1; }
        float MaxDutyCycle() const override { return 1.0; }

        void Process(const MoveOnly& val) override {
            EXPECT_TRUE(val);
        }
    };

    struct MoveTransform : public TransformBody<MoveOnly, MoveOnly>
    {
        size_t ConcurrencyLimit() const override { return 1; }
        float MaxDutyCycle() const override { return 1.0; }

        MoveOnly Process(MoveOnly val) override {
            EXPECT_TRUE(val);
            return val;
        }
    };

    struct MoveTransformConst : public TransformBody<const MoveOnly, const MoveOnly>
    {
        size_t ConcurrencyLimit() const override { return 1; }
        float MaxDutyCycle() const override { return 1.0; }

        MoveOnly Process(const MoveOnly& val) override {
            EXPECT_TRUE(val);
            return MoveOnly{};
        }
    };

    struct MoveMultiTransform : public MultiTransformBody<MoveOnly, MoveOnly>
    {
        size_t ConcurrencyLimit() const override { return 1; }
        float MaxDutyCycle() const override { return 1.0; }

        void Process(MoveOnly val) override {
            EXPECT_TRUE(val);
            PushOut(MoveOnly{});
        }
    };

    struct MoveMultiTransformConst : public MultiTransformBody<const MoveOnly, const MoveOnly>
    {
        size_t ConcurrencyLimit() const override { return 1; }
        float MaxDutyCycle() const override { return 1.0; }

        void Process(const MoveOnly& val) override {
            EXPECT_TRUE(val);
            PushOut(MoveOnly{});
        }
    };

    // Check a graph composed of move only types with no const arguments in the graph edges.
    // Without const arguments, a given node cannot have multiple children, as once one
    // child gets moved the result (copy is impossible), it's now invalid and can't be
    // handed to sibling children
    {
        GraphManager<STAGES> graph;
        auto* input = graph.AddNode(std::make_unique<MoveMultiTransform>(), STAGES::ONE);
        auto* stage2 = input->AddNode(std::make_unique<MoveTransform>(), STAGES::TWO);
        // Second child to input will cause an error
        EXPECT_ANY_THROW(input->AddNode(std::make_unique<MoveLeaf>(), STAGES::THREE));
        stage2->AddNode(std::make_unique<MoveLeaf>(), STAGES::THREE);
        // Second child to stage2 will cause an error
        EXPECT_ANY_THROW(stage2->AddNode(std::make_unique<MoveLeaf>(), STAGES::THREE));

        input->ProcessInput(MoveOnly{});
    }

    // Check a graph composed of move only types, now using const arguments for input/output.
    // Now a node can safely have multiple children
    {
        GraphManager<STAGES> graph;
        auto* input = graph.AddNode(std::make_unique<MoveMultiTransformConst>(), STAGES::ONE);
        auto* stage2 = input->AddNode(std::make_unique<MoveTransformConst>(), STAGES::TWO);
        // Second child of input is now valid
        EXPECT_NO_THROW(input->AddNode(std::make_unique<MoveLeafConst>(), STAGES::THREE));
        stage2->AddNode(std::make_unique<MoveLeafConst>(), STAGES::THREE);
        // Second child of stage2 is now valid
        EXPECT_NO_THROW(stage2->AddNode(std::make_unique<MoveLeafConst>(), STAGES::THREE));

        input->ProcessInput(MoveOnly{});
    }
}

// Tried to write a test to make sure the concurrency limits were working,
// but I'm fighting the scheduler too much.  It's hard to construct an
// artificial case that would try to saturate the concurrency limits without
// taking eons to run, and if it doesn't even try to saturate the limits then
// the test is useless
TEST(GraphAPI, DISABLED_ConcurrencyLimits)
{
    struct ConLeaf : LeafBody<size_t>
    {
        ConLeaf(size_t maxConcurrency)
            : maxConcurrency_{maxConcurrency}
        {}

        size_t ConcurrencyLimit() const override { return maxConcurrency_; }
        float MaxDutyCycle() const override { return 1.0; }

        void Process(size_t durationUS) override {
            usleep(durationUS);
        }

    private:
        size_t maxConcurrency_;
    };

    struct ConTransform : TransformBody<size_t, size_t>
    {
        ConTransform(size_t maxConcurrency)
            : maxConcurrency_{maxConcurrency}
        {}

        size_t ConcurrencyLimit() const override { return maxConcurrency_; }
        float MaxDutyCycle() const override { return 1.0; }

        size_t Process(size_t durationUS) override {
            usleep(durationUS);
            return durationUS;
        }

    private:
        size_t maxConcurrency_;
    };

    struct ConMultiTransform : MultiTransformBody<size_t, size_t>
    {
        ConMultiTransform(size_t maxConcurrency)
            : maxConcurrency_{maxConcurrency}
        {}

        size_t ConcurrencyLimit() const override { return maxConcurrency_; }
        float MaxDutyCycle() const override { return 1.0; }

        void Process(size_t durationUS) override {
            usleep(durationUS);
            PushOut(durationUS);
        }

    private:
        size_t maxConcurrency_;
    };

    {
        GraphManager<STAGES> graph;
        auto* input = graph.AddNode(std::make_unique<ConLeaf>(2), STAGES::ONE);

        // Start 10 jobs that just sleep for 10000us.
        for (size_t i = 0; i < 10; ++i)
        {
            input->ProcessInput(10000);
        }
        const auto& reports = graph.SynchronizeAndReport(0);
        ASSERT_EQ(reports.size(), 1);

        const auto& report = reports[0];
        EXPECT_EQ(report.stage, STAGES::ONE);
        EXPECT_NEAR(report.avgOccupancy, 2, .01);
    }

    {
        GraphManager<STAGES> graph;
        auto* input = graph.AddNode(std::make_unique<ConTransform>(3), STAGES::ONE);
        input->AddNode(std::make_unique<ConTransform>(2), STAGES::TWO)
             ->AddNode(std::make_unique<ConLeaf>(3), STAGES::THREE);

        for (size_t i = 0; i < 20; ++i)
        {
            input->ProcessInput(1000000);
        }
        const auto& reports = graph.SynchronizeAndReport(0);
        ASSERT_EQ(reports.size(), 3);

        auto validate = [](auto report) {
            switch(report.stage)
            {
            case STAGES::ONE:
                EXPECT_NEAR(report.avgOccupancy, 3, .01);
                break;
            case STAGES::TWO:
                EXPECT_NEAR(report.avgOccupancy, 2, .01);
                break;
            case STAGES::THREE:
                EXPECT_NEAR(report.avgOccupancy, 1, .01);
                break;
            default:
                ASSERT_TRUE(false);
            }
        };
        validate(reports[0]);
        validate(reports[1]);
    }
}

// Leaf node that doesn't do anything but store the inputs it receives
// into an intermediate vector.  Only upon receiving a flush command
// does it move those values into a "final" location
struct FlushLeaf : LeafBody<const int>
{
    size_t ConcurrencyLimit() const override { return 1; };
    float MaxDutyCycle() const override { return 1; };

    void Process(const int& val) override
    {
        intermediate.push_back(val);
    }

    std::vector<uint32_t> GetFlushTokens() override
    {
        done.resize(intermediate.size());
        std::vector<uint32_t> ret(intermediate.size());
        std::iota(ret.begin(), ret.end(), 0);
        return ret;
    }

    void Flush(uint32_t token) override
    {
        done[token] = intermediate[token];
    }

    std::vector<int> intermediate;
    std::vector<int> done;
};

TEST(GraphAPI, FlushLeaf)
{
    PacBio::Logging::LogSeverityContext lsc(PacBio::Logging::LogLevel::WARN);

    auto leafPtr = std::make_unique<FlushLeaf>();
    auto& leafRef = *leafPtr;
    GraphManager<STAGES> graph;
    auto* inputNode = graph.AddNode(std::move(leafPtr), STAGES::ONE);

    std::array<int, 8> vals {1,3,5,7,9,11,13,15};
    for (auto val : vals) inputNode->ProcessInput(val);

    graph.Synchronize();

    // The initial processing shouldn't have done anything but
    // place all the valuees into the "intermediate" vector.
    ASSERT_EQ(leafRef.intermediate.size(), vals.size());
    ASSERT_EQ(leafRef.done.size(), 0);
    for (size_t i = 0; i < vals.size(); ++i)
    {
        EXPECT_EQ(vals[i], leafRef.intermediate[i]);
    }

    // Flushing the node should result in all the values copied from
    // the intermediate to the "done" vector
    inputNode->FlushNode();
    ASSERT_EQ(leafRef.done.size(), vals.size());
    for (size_t i = 0; i < vals.size(); ++i)
    {
        EXPECT_EQ(vals[i], leafRef.done[i]);
    }
}

// A transform node where every input is effectively duplicated.
// The main process command both forwards the input (because a
// transform must have an input), and also stores it in an
// "intermediate" vector again.  Upon receiving a flush command
// the inputs are both moved to the "done" vector as well as
// pushed downstream a second time
struct FlushTransform : TransformBody<const int, const int>
{
    size_t ConcurrencyLimit() const override { return 1; };
    float MaxDutyCycle() const override { return 1; };

    int Process(const int& val) override
    {
        intermediate.push_back(val);
        return val;
    }

    std::vector<uint32_t> GetFlushTokens() override
    {
        done.resize(intermediate.size());
        std::vector<uint32_t> ret(intermediate.size());
        std::iota(ret.begin(), ret.end(), 0);
        return ret;
    }

    int Flush(uint32_t token) override
    {
        done[token] = intermediate[token];
        return token;
    }

    std::vector<int> intermediate;
    std::vector<int> done;
};

// Since a graph is required to terminate in leaf nodes,
// here we both check that a TransformNode functions properly
// with flushing, as well as that flushing one node also
// flushes it's children
TEST(GraphAPI, FlushTransform)
{
    PacBio::Logging::LogSeverityContext lsc(PacBio::Logging::LogLevel::WARN);

    // We're going to make sure that we can handle multiple children
    // while we are at it.
    auto leafPtr1 = std::make_unique<FlushLeaf>();
    auto leafPtr2 = std::make_unique<FlushLeaf>();
    auto& leafRef1 = *leafPtr1;
    auto& leafRef2 = *leafPtr2;

    auto transPtr = std::make_unique<FlushTransform>();
    auto& transRef = *transPtr;
    GraphManager<STAGES> graph;
    auto* inputNode = graph.AddNode(std::move(transPtr), STAGES::ONE);

    inputNode->AddNode(std::move(leafPtr1), STAGES::TWO);
    inputNode->AddNode(std::move(leafPtr2), STAGES::THREE);

    std::array<int, 8> vals {1,3,5,7,9,11,13,15};
    for (auto val : vals) inputNode->ProcessInput(val);

    graph.Synchronize();

    // At this stage, all of the nodes should have a copy of the data
    // in their "intermediate" vectors.
    ASSERT_EQ(transRef.intermediate.size(), vals.size());
    ASSERT_EQ(transRef.done.size(), 0);
    ASSERT_EQ(leafRef1.intermediate.size(),  vals.size());
    ASSERT_EQ(leafRef1.done.size(),  0);
    ASSERT_EQ(leafRef2.intermediate.size(),  vals.size());
    ASSERT_EQ(leafRef2.done.size(),  0);

    for (size_t i = 0; i < vals.size(); ++i)
    {
        EXPECT_EQ(vals[i], transRef.intermediate[i]);
        EXPECT_EQ(vals[i], leafRef1.intermediate[i]);
        EXPECT_EQ(vals[i], leafRef2.intermediate[i]);
    }

    // Flushing the transform node should do two things:
    // 1. Flushing the transform node directly should both move the
    //    original data to the "done" vector as well as duplicate it
    //    by pushing it downstream again
    // 2. Cause both children nodes to get flushed as well, moving their
    //    data into the done vector as well
    inputNode->FlushNode();
    ASSERT_EQ(transRef.done.size(), vals.size());
    for (size_t i = 0; i < vals.size(); ++i)
    {
        EXPECT_EQ(vals[i], transRef.done[i]);
    }

    ASSERT_EQ(leafRef1.intermediate.size(),  vals.size()*2);
    ASSERT_EQ(leafRef1.done.size(),  vals.size()*2);
    ASSERT_EQ(leafRef2.intermediate.size(),  vals.size()*2);
    ASSERT_EQ(leafRef2.done.size(),  vals.size()*2);
}

// Another transform, but this time since a multi-transform can
// have zero output, we're not going to duplicate the data.  We
// only send it downstream during the flush operation
struct FlushMultiTransform : MultiTransformBody<const int, const int>
{
    size_t ConcurrencyLimit() const override { return 1; };
    float MaxDutyCycle() const override { return 1; };

    void Process(const int& val) override
    {
        intermediate.push_back(val);
    }

    std::vector<uint32_t> GetFlushTokens() override
    {
        done.resize(intermediate.size());
        return {0};
    }

    void Flush(uint32_t token) override
    {
        assert(token == 0);
        done = intermediate;
        for (auto val : done) PushOut(val);
    }

    std::vector<int> intermediate;
    std::vector<int> done;
};

TEST(GraphAPI, FlushMultiTransform)
{
    PacBio::Logging::LogSeverityContext lsc(PacBio::Logging::LogLevel::WARN);

    // Again testing multiple children while we're at it
    auto leafPtr1 = std::make_unique<FlushLeaf>();
    auto leafPtr2 = std::make_unique<FlushLeaf>();
    auto& leafRef1 = *leafPtr1;
    auto& leafRef2 = *leafPtr2;

    auto multiTransPtr = std::make_unique<FlushMultiTransform>();
    auto& multiTransRef = *multiTransPtr;
    GraphManager<STAGES> graph;
    auto* inputNode = graph.AddNode(std::move(multiTransPtr), STAGES::ONE);

    inputNode->AddNode(std::move(leafPtr1), STAGES::TWO);
    inputNode->AddNode(std::move(leafPtr2), STAGES::THREE);

    std::array<int, 8> vals {1,3,5,7,9,11,13,15};
    for (auto val : vals) inputNode->ProcessInput(val);

    graph.Synchronize();

    // This time none of the data should have made it downstream
    // into the children
    ASSERT_EQ(multiTransRef.intermediate.size(), vals.size());
    ASSERT_EQ(multiTransRef.done.size(), 0);
    for (size_t i = 0; i < vals.size(); ++i)
    {
        EXPECT_EQ(vals[i], multiTransRef.intermediate[i]);
    }

    ASSERT_EQ(leafRef1.intermediate.size(), 0);
    ASSERT_EQ(leafRef1.done.size(),  0);
    ASSERT_EQ(leafRef2.intermediate.size(), 0);
    ASSERT_EQ(leafRef2.done.size(),  0);

    // Now flushing will have pushed data downstream
    // to the children, as well as causing the children
    // to flush as well
    inputNode->FlushNode();
    ASSERT_EQ(multiTransRef.done.size(), vals.size());
    for (size_t i = 0; i < vals.size(); ++i)
    {
        EXPECT_EQ(vals[i], multiTransRef.done[i]);
    }

    ASSERT_EQ(leafRef1.intermediate.size(),  vals.size());
    ASSERT_EQ(leafRef1.done.size(),  vals.size());
    ASSERT_EQ(leafRef2.intermediate.size(),  vals.size());
    ASSERT_EQ(leafRef2.done.size(),  vals.size());
}

} //anon
