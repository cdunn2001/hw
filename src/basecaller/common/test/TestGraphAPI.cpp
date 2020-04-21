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

    graph.Flush();
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

    graph.Flush();
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

    graph.Flush();
    EXPECT_EQ(0, result);

    inputNode->ProcessInput(vals[7]);
    graph.Flush();

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
        const auto& reports = graph.FlushAndReport(0);
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
        const auto& reports = graph.FlushAndReport(0);
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

} //anon
