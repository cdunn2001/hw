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

#include <cmath>
#include <functional>

#include <boost/numeric/conversion/cast.hpp>
#include <gtest/gtest.h>

#include <common/LaneArray.h>
#include <common/simd/ArrayUnion.h>
#include <type_traits>

using namespace PacBio::Mongo;
using namespace PacBio::Cuda::Utility;

using boost::numeric_cast;

// Could potentially make this publicly visible.  However
// CudaArray is quite naturally used in places in structures
// which are dynamically allocated and intentionally layed out
// to be an aligned layout, as they are intended for upload to
// the GPU.  As of now there doesn't seem to be any real need
// for this to have broader visibility.
template <typename T, size_t Len>
struct alignas(64) AlignedCudaArray : public CudaArray<T, Len>
{
    AlignedCudaArray() = default;
    AlignedCudaArray(const AlignedCudaArray&) = default;
    AlignedCudaArray(AlignedCudaArray&&) = default;
    AlignedCudaArray& operator=(const AlignedCudaArray&) = default;
    AlignedCudaArray& operator=(AlignedCudaArray&&) = default;

    using CudaArray<T, Len>::CudaArray;
    AlignedCudaArray(const CudaArray<T, Len>& o)
        : CudaArray<T, Len>(o)
    {}
};

namespace {

// Helper to generate a CudaArray with a uniform value
template <typename T, typename U, size_t Len = laneSize>
AlignedCudaArray<T, Len> ConstantCudaArray(U val)
{
    AlignedCudaArray<T, Len> ret;
    for (size_t i = 0; i < Len; ++i)
    {
        ret[i] = val;
    }
    return ret;
}

// Helper to generate a LaneArray with a uniform value
template <typename T, typename U, size_t Len = laneSize>
LaneArray<T, Len> ConstantLaneArray(U val)
{
    return ConstantCudaArray(val);
}

// Helper to generate a CudaArray that monotonically
// increases.  Must specify the starting value and increment
template <typename T, typename U, size_t Len = laneSize>
AlignedCudaArray<T, Len> IncreasingCudaArray(U startVal, U increment)
{
    AlignedCudaArray<T, Len> ret;
    for (size_t i = 0; i < Len; ++i)
    {
        ret[i] = startVal + i * increment;
    }
    return ret;
}

// Helper to generate a LaneArray that monotonically
// increases.  Must specify the starting value and increment
template <typename T, typename U, size_t Len = laneSize>
LaneArray<T, Len> IncreasingLaneArray(U val, U increment)
{
    return LaneArray<T, Len>(IncreasingCudaArray<T, U, Len>(val, increment));
}

template <size_t Len = laneSize>
AlignedCudaArray<bool, Len> AlternatingBools()
{
    AlignedCudaArray<bool, Len> ret;
    for (size_t i = 0; i < Len; ++i)
    {
        ret[i] = (i % 2 == 0);
    }
    return ret;
}

}

// This test passes merely by compiling.  LaneArrays are large enough that we don't want to zero
// out the memory unless the user specifically asks for it.  This is consistent with how our
// m512 types have always worked.
TEST(BaseArray, TrivialDefaultConstruction)
{
    static_assert(std::is_trivially_default_constructible<LaneArray<float, laneSize>>::value, "Failed trivial test");
    static_assert(std::is_trivially_default_constructible<LaneArray<int32_t, laneSize>>::value, "Failed trivial test");
    static_assert(std::is_trivially_default_constructible<LaneArray<int16_t, laneSize>>::value, "Failed trivial test");
    static_assert(std::is_trivially_default_constructible<LaneArray<uint32_t, laneSize>>::value, "Failed trivial test");
    static_assert(std::is_trivially_default_constructible<LaneArray<uint16_t, laneSize>>::value, "Failed trivial test");
    static_assert(std::is_trivially_default_constructible<LaneMask<laneSize>>::value, "Failed trivial test");
}

// We don't want any extra padding or anything laying around
TEST(BaseArray, Size)
{
    static_assert(sizeof(LaneArray<float, laneSize>) == laneSize*sizeof(float),  "Failed size test");
    static_assert(sizeof(LaneArray<int32_t, laneSize>) == laneSize*sizeof(int32_t),  "Failed size test");
    static_assert(sizeof(LaneArray<int16_t, laneSize>) == laneSize*sizeof(int16_t),  "Failed size test");
    static_assert(sizeof(LaneArray<uint32_t, laneSize>) == laneSize*sizeof(uint32_t),  "Failed size test");
    static_assert(sizeof(LaneArray<uint16_t, laneSize>) == laneSize*sizeof(uint16_t),  "Failed size test");
}

TEST(BaseArray, UniformConstruction)
{
    // Accepts a LaneArray and the value it was constructed
    // with, ensuring that each element is correct
    auto Validate = [](const auto& array, auto val)
    {
        auto check = MakeUnion(array);
        bool allEq = true;
        for (const auto& v : check)
        {
            allEq &= (v == val);
        }
        return allEq;
    };

    LaneArray<float, laneSize> arr1(12.5f);
    EXPECT_TRUE(Validate(arr1, 12.5f));

    LaneArray<int32_t, laneSize> arr2(-13);
    EXPECT_TRUE(Validate(arr2, -13));

    LaneArray<uint32_t, laneSize> arr3(8u);
    EXPECT_TRUE(Validate(arr3, 8u));

    LaneArray<int16_t, laneSize> arr4(-9);
    EXPECT_TRUE(Validate(arr4, -9));

    LaneArray<uint16_t, laneSize> arr5(7u);
    EXPECT_TRUE(Validate(arr5, 7u));

    LaneMask<laneSize> arr6(true);
    for(size_t i = 0; i < laneSize; ++i) EXPECT_TRUE(arr6[i]);
}

// Make sure that we can construct and assign from a CudaArray
TEST(BaseArray, ArrayConstruction)
{
    auto Validate = [](const auto& laneArray, const auto& cudaArray)
    {
        auto check = MakeUnion(laneArray);
        bool allEq = true;
        for (size_t i = 0; i < laneSize; ++i)
        {
            allEq &= (check[i] == cudaArray[i]);
        }
        return allEq;
    };

    auto c1 = IncreasingCudaArray<float>(1.3f, 1.1f);
    LaneArray<float, laneSize> arr1(c1);
    EXPECT_TRUE(Validate(arr1, c1));
    c1 = IncreasingCudaArray<float>(2.3f, 2.1f);
    arr1 = c1;
    EXPECT_TRUE(Validate(arr1, c1));

    auto c2 = IncreasingCudaArray<int32_t>(-5, 2);
    LaneArray<int32_t, laneSize> arr2(c2);
    EXPECT_TRUE(Validate(arr2, c2));
    c2 = IncreasingCudaArray<int32_t>(-10, 3);
    arr2 = c2;
    EXPECT_TRUE(Validate(arr2, c2));

    auto c3 = IncreasingCudaArray<uint32_t>(7, 8);
    LaneArray<uint32_t, laneSize> arr3(c3);
    EXPECT_TRUE(Validate(arr3, c3));
    c3 = IncreasingCudaArray<uint32_t>(2, 5);
    arr3 = c3;
    EXPECT_TRUE(Validate(arr3, c3));

    auto c4 = IncreasingCudaArray<int16_t>(-33, 5);
    LaneArray<int16_t, laneSize> arr4(c4);
    EXPECT_TRUE(Validate(arr4, c4));
    c4 = IncreasingCudaArray<int16_t>(-22, 9);
    arr4 = c4;
    EXPECT_TRUE(Validate(arr4, c4));

    auto c5 = IncreasingCudaArray<uint16_t>(3, 6);
    LaneArray<uint16_t, laneSize> arr5(c5);
    EXPECT_TRUE(Validate(arr5, c5));
    c5 = IncreasingCudaArray<uint16_t>(4, 11);
    arr5 = c5;
    EXPECT_TRUE(Validate(arr5, c5));

    auto c6 = AlternatingBools();
    LaneMask<laneSize> arr6(c6);
    for (size_t i = 0; i < laneSize; ++i)
    {
        EXPECT_TRUE(c6[i] == arr6[i]);
    }
    for (auto& val : c6) val = true;
    arr6 = c6;
    EXPECT_TRUE(all(arr6));
}

// We also want to convert back to cuda array
TEST(BaseArray, ToCudaArray)
{
    auto Validate = [](const auto& cudaArr, const auto& laneArr)
    {
        auto expected = MakeUnion(laneArr);
        bool allEq = true;
        for (size_t i = 0; i < laneSize; ++i)
        {
            allEq &= (cudaArr[i] == expected[i]);
        }
        return allEq;
    };

    auto l1 = IncreasingLaneArray<float>(1.3f, 1.1f);
    CudaArray<float, laneSize> arr1 = l1;
    EXPECT_TRUE(Validate(arr1, l1));

    auto l2 = IncreasingLaneArray<int32_t>(1, 2);
    CudaArray<int32_t, laneSize> arr2 = l2;
    EXPECT_TRUE(Validate(arr2, l2));

    auto l3 = IncreasingLaneArray<uint32_t>(1, 3);
    CudaArray<uint32_t, laneSize> arr3 = l3;
    EXPECT_TRUE(Validate(arr3, l3));

    auto l4 = IncreasingLaneArray<int16_t>(4, 5);
    CudaArray<int16_t, laneSize> arr4 = l4;
    EXPECT_TRUE(Validate(arr4, l4));

    auto l5 = IncreasingLaneArray<uint16_t>(5, 5);
    CudaArray<uint16_t, laneSize> arr5 = l5;
    EXPECT_TRUE(Validate(arr5, l5));
}

// Finally also need to be able to construct from raw pointer
TEST(BaseArray, PointerConstruction)
{
    auto Validate = [](const auto& laneArray, const auto& cudaArray)
    {
        auto check = MakeUnion(laneArray);
        bool allEq = true;
        for (size_t i = 0; i < laneSize; ++i)
        {
            allEq &= (check[i] == cudaArray[i]);
        }
        return allEq;
    };

    auto c1 = IncreasingCudaArray<float>(1.3f, 1.1f);
    LaneArray<float, laneSize> arr1(MemoryRange<float, laneSize>(c1.data()));
    EXPECT_TRUE(Validate(arr1, c1));

    auto c2 = IncreasingCudaArray<int32_t>(-22, 2);
    LaneArray<int32_t, laneSize> arr2(MemoryRange<int32_t, laneSize>(c2.data()));
    EXPECT_TRUE(Validate(arr2, c2));

    auto c3 = IncreasingCudaArray<uint32_t>(-22, 2);
    LaneArray<uint32_t, laneSize> arr3(MemoryRange<uint32_t, laneSize>(c3.data()));
    EXPECT_TRUE(Validate(arr3, c3));

    auto c4 = IncreasingCudaArray<int16_t>(-3, 5);
    LaneArray<int16_t, laneSize> arr4(MemoryRange<int16_t, laneSize>(c4.data()));
    EXPECT_TRUE(Validate(arr4, c4));

    auto c5 = IncreasingCudaArray<uint16_t>(3, 5);
    LaneArray<uint16_t, laneSize> arr5(MemoryRange<uint16_t, laneSize>(c5.data()));
    EXPECT_TRUE(Validate(arr5, c5));
}

// Testing next that logical operations work, as they'll
// be useful to use in subsequent tests
TEST(BoolArray, LogicalOps)
{
    LaneMask<laneSize> mask1(true);
    EXPECT_TRUE(all(mask1));
    EXPECT_TRUE(any(mask1));
    EXPECT_FALSE(none(mask1));

    mask1 = LaneMask<laneSize>(false);
    EXPECT_FALSE(all(mask1));
    EXPECT_FALSE(any(mask1));
    EXPECT_TRUE(none(mask1));

    // Setting up two patterns, which will allow
    // bitwise operations to go through all combinations
    // of states.
    AlignedCudaArray<bool, laneSize> pattern1;
    AlignedCudaArray<bool, laneSize> pattern2;
    for (size_t i = 0; i < laneSize; i+=4)
    {
        pattern1[i+0] = true;
        pattern1[i+1] = false;
        pattern1[i+2] = true;
        pattern1[i+3] = false;

        pattern2[i+0] = true;
        pattern2[i+1] = true;
        pattern2[i+2] = false;
        pattern2[i+3] = false;
    }

    mask1 = LaneMask<laneSize>(pattern1);
    EXPECT_FALSE(all(mask1));
    EXPECT_TRUE(any(mask1));
    EXPECT_FALSE(none(mask1));

    mask1 = LaneMask<laneSize>(pattern1);
    auto mask2 = LaneMask<laneSize>(pattern2);

    auto mask3 = mask1 | mask2;
    for (size_t i = 0; i < laneSize; ++i)
    {
        EXPECT_TRUE(mask3[i] == (mask1[i] | mask2[i])) << i;
    }

    mask3 = mask1 & mask2;
    for (size_t i = 0; i < laneSize; ++i)
    {
        EXPECT_TRUE(mask3[i] == (mask1[i] & mask2[i])) << i;
    }

    mask3 = mask1 ^ mask2;
    for (size_t i = 0; i < laneSize; ++i)
    {
        EXPECT_TRUE(mask3[i] == (mask1[i] ^ mask2[i])) << i;
    }

    mask3 = !mask1;
    for (size_t i = 0; i < laneSize; ++i)
    {
        EXPECT_TRUE(mask3[i] == (!mask1[i])) << i;
    }

    mask3 = mask1 | mask2;
    mask1 |= mask2;
    for (size_t i = 0; i < laneSize; ++i)
    {
        EXPECT_TRUE(mask3[i] == mask1[i]) << i;
    }

    mask1 = LaneMask<laneSize>(pattern1);
    mask3 = mask1 & mask2;
    mask1 &= mask2;
    for (size_t i = 0; i < laneSize; ++i)
    {
        EXPECT_TRUE(mask3[i] == mask1[i]) << i;
    }
}

// Setting up a testing harness, that allows us to have
// a pair of LaneArrays and a pair of scalar values
// for each type.  Will be used with a typed test to
// programatically exercise all variations of scalar/vector
// arguments to binary operators
template <typename T>
struct LinearArray
{
    T initial;
    T stride;
};
template <typename T>
struct Params
{
    LinearArray<T> vec1;
    LinearArray<T> vec2;
    T constant1;
    T constant2;
};
template<class T>
struct LaneArrayHomogeneousTypes: public ::testing::Test {
    using type = T;

    static Params<T> testingParams;
};

// Define an overload set for validation.  Probably could have abstracted it out,
// but it felt faster and clearer to just have distinct vec-vec, scalar-vec and
// vec-scalar versions.
template <typename Op, typename Result, typename Arg1, typename Arg2, size_t Len>
bool ValidateOp(const Result& actual,
                const LaneArray<Arg1, Len>& left,
                const LaneArray<Arg2, Len>& right)
{
    Op op{};
    auto a = actual.ToArray();
    auto l = left.ToArray();
    auto r = right.ToArray();
    bool correct = true;
    for (size_t i = 0; i < Len; ++i)
    {
        correct &= (a[i] == op(l[i], r[i]));
    }
    return correct;
}
template <typename Op, typename Result, typename Arg1, typename Arg2, size_t Len>
bool ValidateOp(const Result& actual,
                      const Arg1& left,
                      const LaneArray<Arg2, Len>& right)
{
    Op op{};
    auto a = actual.ToArray();
    auto l = left;
    auto r = right.ToArray();
    bool correct = true;
    for (size_t i = 0; i < Len; ++i)
    {
        correct &= (a[i] == op(l, r[i]));
    }
    return correct;
}
template <typename Op, typename Result, typename Arg1, typename Arg2, size_t Len>
bool ValidateOp(const Result& actual,
                const LaneArray<Arg1, Len>& left,
                const Arg2& right)
{
    Op op{};
    auto a = actual.ToArray();
    auto l = left.ToArray();
    auto r = right;
    bool correct = true;
    for (size_t i = 0; i < Len; ++i)
    {
        correct &= (a[i] == op(l[i], r));
    }
    return correct;
}

// Set up the actual parameters for each type.
template<> Params<float> LaneArrayHomogeneousTypes<float>::testingParams{
    LinearArray<float>{13.5f, 1.5f},
    LinearArray<float>{9.25f, 1.75f},
        4.f,
        6.5f};
template<> Params<int32_t> LaneArrayHomogeneousTypes<int32_t>::testingParams{
    LinearArray<int32_t>{-270, 91},
    LinearArray<int32_t>{17, 5},
        2,
        12};
template<> Params<int16_t> LaneArrayHomogeneousTypes<int16_t>::testingParams{
    LinearArray<int16_t>{-10, 3},
    LinearArray<int16_t>{-20, 9},
        -4,
        18};
template<> Params<uint32_t> LaneArrayHomogeneousTypes<uint32_t>::testingParams{
    LinearArray<uint32_t>{3, 91},
    LinearArray<uint32_t>{17, 22},
        3,
        7};
template<> Params<uint16_t> LaneArrayHomogeneousTypes<uint16_t>::testingParams{
    LinearArray<uint16_t>{10, 3},
    LinearArray<uint16_t>{20, 9},
        4,
        18};

using ArrTypes = ::testing::Types<int16_t, int32_t, float, uint16_t, uint32_t>;
TYPED_TEST_SUITE(LaneArrayHomogeneousTypes, ArrTypes);

// Test that goes through and checks all arithmetic binary operations
TYPED_TEST(LaneArrayHomogeneousTypes, Arithmetic)
{
    auto& params = this->testingParams;
    using T = typename TestFixture::type;

    const auto v1 = IncreasingLaneArray<T>(params.vec1.initial, params.vec1.stride);
    const auto v2 = IncreasingLaneArray<T>(params.vec2.initial, params.vec2.stride);

    {
        // Check that Blend works for this type
        LaneMask<laneSize> mask(AlternatingBools());
        auto blendResult = MakeUnion(Blend(mask, v1, v2));
        auto tmp1 = MakeUnion(v1);
        auto tmp2 = MakeUnion(v2);
        for (size_t i = 0; i < laneSize; ++i)
        {
            if (i % 2 == 0)
                EXPECT_EQ(blendResult[i], tmp1[i]);
            else
                EXPECT_EQ(blendResult[i], tmp2[i]);
        }

        // While we're at it, check inc and pow2.
        EXPECT_TRUE(all(inc(v1, mask) == Blend(mask, v1+static_cast<T>(1), v1)));
        EXPECT_TRUE(all(pow2(v1) == v1*v1));
    }

    EXPECT_TRUE(ValidateOp<std::plus<T>>(v1+v2, v1, v2));
    EXPECT_TRUE(ValidateOp<std::plus<T>>(params.constant1+v2, params.constant1, v2));
    EXPECT_TRUE(ValidateOp<std::plus<T>>(v1+params.constant2, v1, params.constant2));

    EXPECT_TRUE(ValidateOp<std::minus<T>>(v1-v2, v1, v2));
    EXPECT_TRUE(ValidateOp<std::minus<T>>(params.constant1-v2, params.constant1, v2));
    EXPECT_TRUE(ValidateOp<std::minus<T>>(v1-params.constant2, v1, params.constant2));

    EXPECT_TRUE(ValidateOp<std::multiplies<T>>(v1*v2, v1, v2));
    EXPECT_TRUE(ValidateOp<std::multiplies<T>>(params.constant1*v2, params.constant1, v2));
    EXPECT_TRUE(ValidateOp<std::multiplies<T>>(v1*params.constant2, v1, params.constant2));

    EXPECT_TRUE(ValidateOp<std::divides<T>>(v1/v2, v1, v2));
    EXPECT_TRUE(ValidateOp<std::divides<T>>(params.constant1/v2, params.constant1, v2));
    EXPECT_TRUE(ValidateOp<std::divides<T>>(v1/params.constant2, v1, params.constant2));

    // Doing compound operations requires jumping through a couple extra hoops.
    auto v3 = v1;
    EXPECT_TRUE(ValidateOp<std::plus<T>>(v3+=v2, v1, v2));
    v3 = v1;
    EXPECT_TRUE(ValidateOp<std::plus<T>>(v3+=params.constant1, v1, params.constant1));

    v3 = v1;
    EXPECT_TRUE(ValidateOp<std::minus<T>>(v3-=v2, v1, v2));
    v3 = v1;
    EXPECT_TRUE(ValidateOp<std::minus<T>>(v3-=params.constant1, v1, params.constant1));

    v3 = v1;
    EXPECT_TRUE(ValidateOp<std::multiplies<T>>(v3*=v2, v1, v2));
    v3 = v1;
    EXPECT_TRUE(ValidateOp<std::multiplies<T>>(v3*=params.constant1, v1, params.constant1));

    v3 = v1;
    EXPECT_TRUE(ValidateOp<std::divides<T>>(v3/=v2, v1, v2));
    v3 = v1;
    EXPECT_TRUE(ValidateOp<std::divides<T>>(v3/=params.constant1, v1, params.constant1));
}

// Similar to last test, now focusing on comparison binary operations
TYPED_TEST(LaneArrayHomogeneousTypes, Comparisons)
{
    auto& params = this->testingParams;
    using T = typename TestFixture::type;

    const auto v1 = IncreasingLaneArray<T>(params.vec1.initial, params.vec1.stride);
    const auto v2 = IncreasingLaneArray<T>(params.vec2.initial, params.vec2.stride);

    EXPECT_TRUE(ValidateOp<std::less<T>>(v1 < v2, v1, v2));
    EXPECT_TRUE(ValidateOp<std::less<T>>(v1 < v1, v1, v1));
    EXPECT_TRUE(ValidateOp<std::less<T>>(params.constant1 < v2, params.constant1, v2));
    EXPECT_TRUE(ValidateOp<std::less<T>>(v1 < params.constant2, v1, params.constant2));

    EXPECT_TRUE(ValidateOp<std::less_equal<T>>(v1 <= v2, v1, v2));
    EXPECT_TRUE(ValidateOp<std::less_equal<T>>(v1 <= v1, v1, v1));
    EXPECT_TRUE(ValidateOp<std::less_equal<T>>(params.constant1 <= v2, params.constant1, v2));
    EXPECT_TRUE(ValidateOp<std::less_equal<T>>(v1 <= params.constant2, v1, params.constant2));

    EXPECT_TRUE(ValidateOp<std::greater<T>>(v1 > v2, v1, v2));
    EXPECT_TRUE(ValidateOp<std::greater<T>>(v1 > v1, v1, v1));
    EXPECT_TRUE(ValidateOp<std::greater<T>>(params.constant1 > v2, params.constant1, v2));
    EXPECT_TRUE(ValidateOp<std::greater<T>>(v1 > params.constant2, v1, params.constant2));

    EXPECT_TRUE(ValidateOp<std::greater_equal<T>>(v1 >= v2, v1, v2));
    EXPECT_TRUE(ValidateOp<std::greater_equal<T>>(v1 >= v1, v1, v1));
    EXPECT_TRUE(ValidateOp<std::greater_equal<T>>(params.constant1 >= v2, params.constant1, v2));
    EXPECT_TRUE(ValidateOp<std::greater_equal<T>>(v1 >= params.constant2, v1, params.constant2));

    EXPECT_TRUE(ValidateOp<std::equal_to<T>>(v1 == v2, v1, v2));
    EXPECT_TRUE(ValidateOp<std::equal_to<T>>(v1 == v1, v1, v1));
    EXPECT_TRUE(ValidateOp<std::equal_to<T>>(params.constant1 == v2, params.constant1, v2));
    EXPECT_TRUE(ValidateOp<std::equal_to<T>>(v1 == params.constant2, v1, params.constant2));

    EXPECT_TRUE(ValidateOp<std::not_equal_to<T>>(v1 != v2, v1, v2));
    EXPECT_TRUE(ValidateOp<std::not_equal_to<T>>(v1 != v1, v1, v1));
    EXPECT_TRUE(ValidateOp<std::not_equal_to<T>>(params.constant1 != v2, params.constant1, v2));
    EXPECT_TRUE(ValidateOp<std::not_equal_to<T>>(v1 != params.constant2, v1, params.constant2));

    struct Min { T operator()(const T& l,const T&r) { return std::min(l,r);}};
    EXPECT_TRUE(ValidateOp<Min>(min(v1, v2), v1, v2));
    EXPECT_TRUE(ValidateOp<Min>(min(v1, v1), v1, v1));
    EXPECT_TRUE(ValidateOp<Min>(min(params.constant1, v2), params.constant1, v2));
    EXPECT_TRUE(ValidateOp<Min>(min(v1, params.constant2), v1, params.constant2));

    struct Max { T operator()(const T& l,const T&r) { return std::max(l,r);}};
    EXPECT_TRUE(ValidateOp<Max>(max(v1, v2), v1, v2));
    EXPECT_TRUE(ValidateOp<Max>(max(v1, v1), v1, v1));
    EXPECT_TRUE(ValidateOp<Max>(max(params.constant1, v2), params.constant1, v2));
    EXPECT_TRUE(ValidateOp<Max>(max(v1, params.constant2), v1, params.constant2));
}

// Test operations unique to LaneArray<float>
TEST(LaneArray, FloatOps)
{
    auto cudaArr = IncreasingCudaArray<float>(-234.5f, 8.9f);
    LaneArray<float, laneSize> laneArr(cudaArr);

    auto result = erfc(laneArr).ToArray();
    for (size_t i = 0; i < laneSize; ++i)
    {
        EXPECT_FLOAT_EQ(result[i], std::erfc(cudaArr[i])) << i;
    }

    auto result2 = floorCastInt(laneArr).ToArray();
    for (size_t i = 0; i < laneSize; ++i)
    {
        EXPECT_EQ(result2[i], std::floor(cudaArr[i])) << i;
    }

    const auto result3 = roundCastInt(laneArr).ToArray();
    for (size_t i = 0; i < laneSize; ++i)
    {
        EXPECT_EQ(result3[i], std::rint(cudaArr[i]))
            << "  i is " << i << '\n'
            << "  cudaArr[i] is " << cudaArr[i];
    }

    const auto resultAbs = abs(laneArr).ToArray();
    for (size_t i = 0; i < laneSize; ++i)
    {
        EXPECT_EQ(resultAbs[i], std::abs(cudaArr[i]))
            << "  i is " << i << '\n'
            << "  cudaArr[i] is " << cudaArr[i];
    }

    // Get rid of negatives so we can take the sqrt and log
    cudaArr = IncreasingCudaArray<float>(2.9f, 13.9f);
    laneArr = LaneArray<float, laneSize>(cudaArr);

    result = sqrt(laneArr).ToArray();
    for (size_t i = 0; i < laneSize; ++i)
    {
        EXPECT_FLOAT_EQ(result[i], std::sqrt(cudaArr[i])) << i;
    }


    result = log(laneArr).ToArray();
    for (size_t i = 0; i < laneSize; ++i)
    {
        EXPECT_FLOAT_EQ(result[i], std::log(cudaArr[i])) << i;
    }

    result = log2(laneArr).ToArray();
    for (size_t i = 0; i < laneSize; ++i)
    {
        EXPECT_FLOAT_EQ(result[i], std::log2(cudaArr[i])) << i;
    }

    // Lower the dynamic range so we can take the exp
    // SSE and std:exp diverge a bit toward the extreme
    // end of the range
    cudaArr = IncreasingCudaArray<float>(-12.34f, .74f);
    laneArr = LaneArray<float, laneSize>(cudaArr);

    result = exp(laneArr).ToArray();
    for (size_t i = 0; i < laneSize; ++i)
    {
        EXPECT_FLOAT_EQ(result[i], std::exp(cudaArr[i])) << i;
    }

    result = exp2(laneArr).ToArray();
    for (size_t i = 0; i < laneSize; ++i)
    {
        EXPECT_FLOAT_EQ(result[i], std::exp2(cudaArr[i])) << i;
    }

    // manually inject some nan and inf
    for (size_t i = 0; i < laneSize; ++i)
    {
        if (i % 3 == 1) cudaArr[i] = std::numeric_limits<float>::infinity();
        if (i % 3 == 2) cudaArr[i] = std::numeric_limits<float>::quiet_NaN();
    }
    laneArr = LaneArray<float, laneSize>(cudaArr);
    auto finiteMask = isfinite(laneArr);
    auto nanMask = isnan(laneArr);
    for (size_t i = 0; i < laneSize; ++i)
    {
        if (i % 3 == 0)
        {
            EXPECT_TRUE(finiteMask[i]) << i;
            EXPECT_FALSE(nanMask[i]) << i;
        }
        if (i % 3 == 1)
        {
            EXPECT_FALSE(finiteMask[i]) << i;
            EXPECT_FALSE(nanMask[i]) << i;
        }
        if (i % 3 == 2)
        {
            EXPECT_FALSE(finiteMask[i]) << i;
            EXPECT_TRUE(nanMask[i]) << i;
        }
    }
}

TEST(LaneArray, IntOps)
{
    auto a = IncreasingLaneArray<int>(12, 23);
    auto b = IncreasingLaneArray<int>(3, 15);
    const auto c = IncreasingLaneArray<int>(-12, 1);

    EXPECT_TRUE(ValidateOp<std::bit_or<int>>(a | b, a, b));
    EXPECT_TRUE(ValidateOp<std::modulus<int>>(a % b, a, b));

    {   // Test for abs.
        const auto result = abs(c);
        const auto cArr = c.ToArray();
        const auto rArr = result.ToArray();
        for (unsigned int i = 0; i < cArr.size(); ++i)
        {
            EXPECT_EQ(std::abs(cArr[i]), rArr[i]);
        }
    }
}

// Our unsigned int class supports various bitwise operations,
// right now primarily to support bit packing operations in
// the FrameLabeler
TEST(LaneArray, UIntOps)
{
    using Arr = LaneArray<uint32_t>;

    //----------First bitwise Or
    //
    // b has a subset of the bits of a
    Arr a = 7;
    Arr b = 4;
    EXPECT_TRUE(all((a|a) == a));
    EXPECT_TRUE(all((a|b) == a));
    EXPECT_TRUE(all((b|a) == a));

    // a and b have a disjoint set of bits,
    // or is the same as addition
    b = 24;
    EXPECT_TRUE(all((a|b) == a+b));
    EXPECT_TRUE(all((b|a) == a+b));

    // a and b overlap in the 3rd bit.
    b = 20;
    EXPECT_TRUE(all((a|b) == a+b-4u));
    EXPECT_TRUE(all((b|a) == a+b-4u));

    auto c = a;
    c |= b;
    EXPECT_TRUE(all(c == a+b-4u));

    //----------Now bitwise And
    //
    // b has a subset of the bits of a
    a = 7;
    b = 4;
    EXPECT_TRUE(all((a&a) == a));
    EXPECT_TRUE(all((a&b) == b));
    EXPECT_TRUE(all((b&a) == b));

    // a and b have a disjoint set of bits,
    // or is the same as addition
    b = 24;
    EXPECT_TRUE(all((a&b) == 0u));
    EXPECT_TRUE(all((b&a) == 0u));

    // a and b overlap in the 3rd bit.
    b = 20;
    EXPECT_TRUE(all((a&b) == 4u));
    EXPECT_TRUE(all((b&a) == 4u));

    c = a;
    c &= b;
    EXPECT_TRUE(all(c == 4u));

    //----------Next scalar bit shifts
    a = 123;
    b = a << 3;
    EXPECT_TRUE(all(b == a*8u));
    b = a >> 3;
    EXPECT_TRUE(all(b == a/8u));

    //----------Finally vector bit shifts
    // gets values 0-63
    auto shift = IncreasingLaneArray<uint32_t>(0, 1);
    // Effectively %32, so that none of our shifts go past
    // 32 bits.
    shift &= 31;
    ArrayUnion<Arr> u = 1 << shift;
    for (size_t i = 0; i < laneSize; ++i)
    {
        EXPECT_EQ(u[i], 1 << (i%32));
    }
    u = std::numeric_limits<uint32_t>::max() >> shift;
    for (size_t i = 0; i < laneSize; ++i)
    {
        EXPECT_EQ(u[i], std::numeric_limits<uint32_t>::max() >> (i%32));
    }
}

// LaneArray tries to mimic scalars in that you can seamlessly
// do mixed-type operations.  Make sure the resulting type
// is as expected.
// Note: Only checking multiplication, because all operations
//       use the same mechanism to determine their common type,
//       and because it would be hard to make this generic.
TEST(LaneArray, OperationResultTypes)
{
    // Leaving these uninitialized.  I really only care
    // what the type system is producing
    LaneArray<float, laneSize> fltArr;
    LaneArray<int32_t, laneSize> intArr;
    LaneArray<uint32_t, laneSize> uintArr;
    LaneArray<int16_t, laneSize> shortArr;
    LaneArray<uint16_t, laneSize> ushortArr;

    auto CheckType = [](auto&& result, auto&& expected)
    {
        return std::is_same<std::decay_t<decltype(result)>, std::decay_t<decltype(expected)>>::value;
    };

    // Anything times a float is a float.
    EXPECT_TRUE(CheckType(fltArr*fltArr,      LaneArray<float, laneSize>{}));
    EXPECT_TRUE(CheckType(intArr*fltArr,      LaneArray<float, laneSize>{}));
    EXPECT_TRUE(CheckType(fltArr*intArr,      LaneArray<float, laneSize>{}));
    EXPECT_TRUE(CheckType(uintArr*fltArr,     LaneArray<float, laneSize>{}));
    EXPECT_TRUE(CheckType(fltArr*uintArr,     LaneArray<float, laneSize>{}));
    EXPECT_TRUE(CheckType(shortArr*fltArr,    LaneArray<float, laneSize>{}));
    EXPECT_TRUE(CheckType(fltArr*shortArr,    LaneArray<float, laneSize>{}));
    EXPECT_TRUE(CheckType(ushortArr*fltArr,   LaneArray<float, laneSize>{}));
    EXPECT_TRUE(CheckType(fltArr*ushortArr,   LaneArray<float, laneSize>{}));
    EXPECT_TRUE(CheckType(1.0f*fltArr,        LaneArray<float, laneSize>{}));
    EXPECT_TRUE(CheckType(fltArr*1.0f,        LaneArray<float, laneSize>{}));
    EXPECT_TRUE(CheckType(1*fltArr,           LaneArray<float, laneSize>{}));
    EXPECT_TRUE(CheckType(fltArr*1,           LaneArray<float, laneSize>{}));
    EXPECT_TRUE(CheckType(1u*fltArr,          LaneArray<float, laneSize>{}));
    EXPECT_TRUE(CheckType(fltArr*1u,          LaneArray<float, laneSize>{}));
    EXPECT_TRUE(CheckType((int16_t)1*fltArr,  LaneArray<float, laneSize>{}));
    EXPECT_TRUE(CheckType(fltArr*(int16_t)1,  LaneArray<float, laneSize>{}));
    EXPECT_TRUE(CheckType((uint16_t)1*fltArr, LaneArray<float, laneSize>{}));
    EXPECT_TRUE(CheckType(fltArr*(uint16_t)1, LaneArray<float, laneSize>{}));

    // Next in priority is 32bit integers.  No mixed signed operations are
    // allowed so that reduces the combinations we need to check
    EXPECT_TRUE(CheckType(intArr*intArr,       LaneArray<int32_t, laneSize>{}));
    EXPECT_TRUE(CheckType(shortArr*intArr,     LaneArray<int32_t, laneSize>{}));
    EXPECT_TRUE(CheckType(intArr*shortArr,     LaneArray<int32_t, laneSize>{}));
    EXPECT_TRUE(CheckType(1*intArr,            LaneArray<int32_t, laneSize>{}));
    EXPECT_TRUE(CheckType(intArr*1,            LaneArray<int32_t, laneSize>{}));
    EXPECT_TRUE(CheckType((int16_t)1*intArr,   LaneArray<int32_t, laneSize>{}));
    EXPECT_TRUE(CheckType(intArr*(int16_t)1,   LaneArray<int32_t, laneSize>{}));

    EXPECT_TRUE(CheckType(uintArr*uintArr,     LaneArray<uint32_t, laneSize>{}));
    EXPECT_TRUE(CheckType(ushortArr*uintArr,   LaneArray<uint32_t, laneSize>{}));
    EXPECT_TRUE(CheckType(uintArr*ushortArr,   LaneArray<uint32_t, laneSize>{}));
    EXPECT_TRUE(CheckType(1u*uintArr,          LaneArray<uint32_t, laneSize>{}));
    EXPECT_TRUE(CheckType(uintArr*1u,          LaneArray<uint32_t, laneSize>{}));
    EXPECT_TRUE(CheckType((uint16_t)1*uintArr, LaneArray<uint32_t, laneSize>{}));
    EXPECT_TRUE(CheckType(uintArr*(uint16_t)1, LaneArray<uint32_t, laneSize>{}));

    // Last in priority are the 16 bit types
    EXPECT_TRUE(CheckType(shortArr*shortArr,   LaneArray<int16_t, laneSize>{}));
    EXPECT_TRUE(CheckType(int16_t(1)*shortArr, LaneArray<int16_t, laneSize>{}));
    EXPECT_TRUE(CheckType(shortArr*int16_t(1), LaneArray<int16_t, laneSize>{}));

    EXPECT_TRUE(CheckType(ushortArr*ushortArr,   LaneArray<uint16_t, laneSize>{}));
    EXPECT_TRUE(CheckType(uint16_t(1)*ushortArr, LaneArray<uint16_t, laneSize>{}));
    EXPECT_TRUE(CheckType(ushortArr*uint16_t(1), LaneArray<uint16_t, laneSize>{}));

    // This is the exception.  Don't let a scalar int promote a shortArray
    // It's too hard to type short literals, and even if you have a 16
    // bit type, it's horribly easy to promote it to 32 bits. Without this
    // we'll constantly be doing unecessary (and expensive) promotions
    // from a short array to an int array
    EXPECT_TRUE(CheckType(1*shortArr,   LaneArray<int16_t, laneSize>{}));
    EXPECT_TRUE(CheckType(shortArr*1,   LaneArray<int16_t, laneSize>{}));
    EXPECT_TRUE(CheckType(1u*ushortArr, LaneArray<uint16_t, laneSize>{}));
    EXPECT_TRUE(CheckType(ushortArr*1u, LaneArray<uint16_t, laneSize>{}));

    // Float scalars do always cause promotions still
    EXPECT_TRUE(CheckType(1.f*intArr,   LaneArray<float, laneSize>{}));
    EXPECT_TRUE(CheckType(intArr*1.f,   LaneArray<float, laneSize>{}));
    EXPECT_TRUE(CheckType(1.f*shortArr, LaneArray<float, laneSize>{}));
    EXPECT_TRUE(CheckType(shortArr*1.f, LaneArray<float, laneSize>{}));

    EXPECT_TRUE(CheckType(1.f*uintArr,   LaneArray<float, laneSize>{}));
    EXPECT_TRUE(CheckType(uintArr*1.f,   LaneArray<float, laneSize>{}));
    EXPECT_TRUE(CheckType(1.f*ushortArr, LaneArray<float, laneSize>{}));
    EXPECT_TRUE(CheckType(ushortArr*1.f, LaneArray<float, laneSize>{}));
}

// As long as we're just doing 2-s compliment and normal unsigned rollover,
// there's really not much difference between signed and unsigned arithmetic.
// A large portion of the potentially unsigned SIMD intrinsics don't even
// exist (e.g. _mm512_add_epu32 doesn't exist, you're supposed to use
// _mm512_add_epi32).  All we really need to check is comparisons, division,
// and maybe a few conversions.
TEST(LaneArray, SignedVsUnsignedIntegers)
{
    LaneArray<int16_t, laneSize> shortSignedZero{0};
    LaneArray<int16_t, laneSize> shortSignedNeg1{static_cast<int16_t>(-1)};
    LaneArray<int16_t, laneSize> shortSignedNeg2{static_cast<int16_t>(-2)};

    LaneArray<int32_t, laneSize> intSignedZero{0};
    LaneArray<int32_t, laneSize> intSignedNeg1{static_cast<int32_t>(-1)};
    LaneArray<int32_t, laneSize> intSignedNeg2{static_cast<int32_t>(-2)};

    // Force "rollover" with negatives, just to verify that the same byte
    // pattern has different signed/unsigned results for some key operations
    LaneArray<uint16_t, laneSize> shortUnsignedZero{0};
    LaneArray<uint16_t, laneSize> shortUnsignedNeg1{static_cast<uint16_t>(-1)};
    LaneArray<uint16_t, laneSize> shortUnsignedNeg2{static_cast<uint16_t>(-2)};

    LaneArray<uint32_t, laneSize> intUnsignedZero{0};
    LaneArray<uint32_t, laneSize> intUnsignedNeg1{static_cast<uint32_t>(-1)};
    LaneArray<uint32_t, laneSize> intUnsignedNeg2{static_cast<uint32_t>(-2)};

    // Check comparisons.  They are all implemented indpenedantly
    // (e.g. > isn't the negation of >=), so we should check them all
    EXPECT_TRUE(all(intSignedNeg1  <  intSignedZero));
    EXPECT_TRUE(all(intSignedNeg1  <= intSignedZero));
    EXPECT_TRUE(none(intSignedNeg1 >  intSignedZero));
    EXPECT_TRUE(none(intSignedNeg1 >= intSignedZero));

    EXPECT_TRUE(all(shortSignedNeg1  <  shortSignedZero));
    EXPECT_TRUE(all(shortSignedNeg1  <= shortSignedZero));
    EXPECT_TRUE(none(shortSignedNeg1 >  shortSignedZero));
    EXPECT_TRUE(none(shortSignedNeg1 >= shortSignedZero));

    EXPECT_TRUE(none(shortUnsignedNeg1 <  shortUnsignedZero));
    EXPECT_TRUE(none(shortUnsignedNeg1 <= shortUnsignedZero));
    EXPECT_TRUE(all(shortUnsignedNeg1  >  shortUnsignedZero));
    EXPECT_TRUE(all(shortUnsignedNeg1  >= shortUnsignedZero));

    EXPECT_TRUE(none(intUnsignedNeg1 <  intUnsignedZero));
    EXPECT_TRUE(none(intUnsignedNeg1 <= intUnsignedZero));
    EXPECT_TRUE(all(intUnsignedNeg1  >  intUnsignedZero));
    EXPECT_TRUE(all(intUnsignedNeg1  >= intUnsignedZero));

    // Now check division
    EXPECT_TRUE(all(shortSignedNeg2    / shortSignedNeg1   == static_cast<int16_t>(2)));
    EXPECT_TRUE(all(intSignedNeg2      / intSignedNeg1     == static_cast<int32_t>(2)));
    EXPECT_TRUE(none(shortUnsignedNeg2 / shortUnsignedNeg1 == static_cast<uint16_t>(2)));
    EXPECT_TRUE(none(intUnsignedNeg2   / intUnsignedNeg1   == static_cast<uint32_t>(2)));

    // Make sure we have expected sign when going to a wider signed type like float.
    EXPECT_TRUE(all(LaneArray<float, laneSize>(shortUnsignedNeg2) > 0.0f));
    EXPECT_TRUE(all(LaneArray<float, laneSize>(shortSignedNeg2) < 0.0f));
    EXPECT_TRUE(all(LaneArray<float, laneSize>(intUnsignedNeg2) > 0.0f));
    EXPECT_TRUE(all(LaneArray<float, laneSize>(intSignedNeg2) < 0.0f));
}

// How do you write a unit test that checks to make sure certain code
// *doesn't* compile?  Magic, that's how.  (Really by forcing dependant
// contexts and relying on SFINAE)

template <typename T> struct sink { using type = void; };
template <typename T> using sink_t = typename sink<T>::type;

// Template and specialization that accepts a function type and two
// argument types.  This works from a combination of template
// specialization and SFINAE in dependant contexts.  If
// F<T1, T2> is a valid expression, then as `sink_t` always
// returns a `void` type, `CheckCompiles<F, T1, T2, void>`
// inherits from true.  If `F<T1, T2>` is *not* a valid expression,
// then SFINAE simply kicks in and disqualifies the specialization,
// casuing `CheckCompiles<F, T1, T2, Anything>` to inherit from false.
// In the general template list Result has a default of `void`
// specifically to match what sink_t produces.  If they were not
// the same this would still work, but you'd have to actually type
// CheckCompiles<A, B, C, void> instead of just CheckCompiles<A, B, C>
template <template <typename, typename> class F, typename T1, typename T2, typename Result = void>
struct CheckCompiles : public std::false_type {};
template <template <typename, typename> class F, typename T1, typename T2>
struct CheckCompiles<F, T1, T2, sink_t<F<T1, T2>>> : public std::true_type {};

// Set up aliases for decltype expressions, so that we can plug them
// into the above test.  It's a bit annoying to type them all out,
// but at least it's just mindless boilerplate.
struct MulOp {
template <typename T1, typename T2>
using type = decltype(std::declval<const T1&>() * std::declval<const T2&>());
};
struct DivOp {
template <typename T1, typename T2>
using type = decltype(std::declval<const T1&>() / std::declval<const T2&>());
};
struct SubOp {
template <typename T1, typename T2>
using type = decltype(std::declval<const T1&>() - std::declval<const T2&>());
};
struct AddOp {
template <typename T1, typename T2>
using type = decltype(std::declval<const T1&>() + std::declval<const T2&>());
};
struct MulEqOp {
template <typename T1, typename T2>
using type = decltype(std::declval<T1&>() *= std::declval<const T2&>());
};
struct DivEqOp {
template <typename T1, typename T2>
using type = decltype(std::declval<T1&>() /= std::declval<const T2&>());
};
struct SubEqOp {
template <typename T1, typename T2>
using type = decltype(std::declval<T1&>() -= std::declval<const T2&>());
};
struct AddEqOp {
template <typename T1, typename T2>
using type = decltype(std::declval<T1&>() += std::declval<const T2&>());
};
struct GtOp {
template <typename T1, typename T2>
using type = decltype(std::declval<const T1&>() > std::declval<const T2&>());
};
struct LtOp {
template <typename T1, typename T2>
using type = decltype(std::declval<const T1&>() < std::declval<const T2&>());
};
struct GtEqOp {
template <typename T1, typename T2>
using type = decltype(std::declval<const T1&>() >= std::declval<const T2&>());
};
struct LtEqOp {
template <typename T1, typename T2>
using type = decltype(std::declval<const T1&>() <= std::declval<const T2&>());
};
struct EqOp {
template <typename T1, typename T2>
using type = decltype(std::declval<const T1&>() == std::declval<const T2&>());
};
struct NeqOp {
template <typename T1, typename T2>
using type = decltype(std::declval<const T1&>() != std::declval<const T2&>());
};

template<class T>
struct LaneArrayBinaryOps : public ::testing::Test {
    using type = T;
};
using BinaryTypes = ::testing::Types<MulOp, DivOp, AddOp, SubOp,
                                     LtOp, GtOp, LtEqOp, GtEqOp, EqOp, NeqOp>;
TYPED_TEST_SUITE(LaneArrayBinaryOps, BinaryTypes);

// Test all binary ops.  Either the left or the right argument
// should be allowed to both be a different type as well as
// potentially a scalar.
TYPED_TEST(LaneArrayBinaryOps, Valid)
{
    using Op = typename TestFixture::type;


    // Helper function to do the scalar/array combinations.  t1 and t2
    // should just be scalar types, and this function will promote
    // that to an array as necessary
    auto PermuteScalarVector = [](bool expectedToCompile, auto&& t1, auto&& t2)
    {
        using T1 = std::decay_t<decltype(t1)>;
        using T2 = std::decay_t<decltype(t2)>;
        bool success = true;
        success &= (!expectedToCompile) xor CheckCompiles<Op::template type, LaneArray<T1, laneSize>, LaneArray<T2, laneSize>>::value;
        success &= (!expectedToCompile) xor CheckCompiles<Op::template type, T1, LaneArray<T2, laneSize>>::value;
        success &= (!expectedToCompile) xor CheckCompiles<Op::template type, LaneArray<T1, laneSize>, T2>::value;
        return success;
    };
    // One more lambda to switch left and right arguments
    auto TestAllCombinations = [&](bool expectedToCompile, auto&& t1, auto&& t2)
    {
        return PermuteScalarVector(expectedToCompile, t1, t2)
             & PermuteScalarVector(expectedToCompile, t2, t1);
    };

    // Now with the above lambdas, we only have to specify two types,
    // and we'll make sure the operation compiles (or not) as
    // expected regardless of scalar/vector and left/right
    // permutations.

    // Floats can operate with anything.
    EXPECT_TRUE(TestAllCombinations(true, float{}, float{}));
    EXPECT_TRUE(TestAllCombinations(true, float{}, int32_t{}));
    EXPECT_TRUE(TestAllCombinations(true, float{}, uint32_t{}));
    EXPECT_TRUE(TestAllCombinations(true, float{}, int16_t{}));
    EXPECT_TRUE(TestAllCombinations(true, float{}, uint16_t{}));

    // integers can only operate on things with the samed signedness.
    // This is to force users to do an explicit cast if they
    // want to mix signed and unsigned variables.
    EXPECT_TRUE(TestAllCombinations(true, int32_t{}, int32_t{}));
    EXPECT_TRUE(TestAllCombinations(true, int32_t{}, int16_t{}));

    EXPECT_TRUE(TestAllCombinations(true, uint32_t{}, uint32_t{}));
    EXPECT_TRUE(TestAllCombinations(true, uint32_t{}, uint16_t{}));

    EXPECT_TRUE(TestAllCombinations(true, uint16_t{}, uint16_t{}));
    EXPECT_TRUE(TestAllCombinations(true, int16_t{}, int16_t{}));

    // Now the expected failures.  These combinations should
    // cause a compilation error in normal use, as they mix sign.
    EXPECT_TRUE(TestAllCombinations(false, int32_t{}, uint32_t{}));
    EXPECT_TRUE(TestAllCombinations(false, int32_t{}, uint16_t{}));

    EXPECT_TRUE(TestAllCombinations(false, int16_t{}, uint16_t{}));
}

template<class T>
struct LaneArrayCompoundOps : public ::testing::Test {
    using type = T;
};
using BinaryTypesCompound = ::testing::Types<MulEqOp, DivEqOp, AddEqOp, SubEqOp>;

TYPED_TEST_SUITE(LaneArrayCompoundOps, BinaryTypesCompound);

// Compound operations need their own test, since they are a
// little more constrained.  For instance the lhs cannot be
// a scalar, and the return type must match the lhs (e.g.
// LaneArray<int> += 1.0f cannot work)
TYPED_TEST(LaneArrayCompoundOps, Valid)
{
    using Op = typename TestFixture::type;

    // Helper function to do the scalar/vector combinatorics
    // Only have to worry about thr rhs, as the left is always
    // an array.
    auto TestVecScalarCombinations = [](bool expectedToCompile, auto&& t1, auto&& t2)
    {
        using T1 = std::decay_t<decltype(t1)>;
        using T2 = std::decay_t<decltype(t2)>;
        bool success = true;
        success &= (!expectedToCompile) xor CheckCompiles<Op::template type, LaneArray<T1, laneSize>, LaneArray<T2, laneSize>>::value;
        success &= (!expectedToCompile) xor CheckCompiles<Op::template type, LaneArray<T1, laneSize>, T2>::value;
        return success;
    };

    // Everything promotes to float, so all type combinations will work
    EXPECT_TRUE(TestVecScalarCombinations(true, float{}, float{}));
    EXPECT_TRUE(TestVecScalarCombinations(true, float{}, int32_t{}));
    EXPECT_TRUE(TestVecScalarCombinations(true, float{}, uint32_t{}));
    EXPECT_TRUE(TestVecScalarCombinations(true, float{}, int16_t{}));
    EXPECT_TRUE(TestVecScalarCombinations(true, float{}, uint16_t{}));

    // Integrals will work with anything as long as the rhs is the
    // same sign and same or lesser width
    EXPECT_TRUE(TestVecScalarCombinations(true, int32_t{}, int32_t{}));
    EXPECT_TRUE(TestVecScalarCombinations(true, int32_t{}, int16_t{}));

    EXPECT_TRUE(TestVecScalarCombinations(true, uint32_t{}, uint32_t{}));
    EXPECT_TRUE(TestVecScalarCombinations(true, uint32_t{}, uint16_t{}));

    EXPECT_TRUE(TestVecScalarCombinations(true, uint16_t{}, uint16_t{}));
    EXPECT_TRUE(TestVecScalarCombinations(true, int16_t{}, int16_t{}));

    // Now the expected failures.  These combinations should
    // cause a compilation error

    // First signed/unsigned mismatches
    EXPECT_TRUE(TestVecScalarCombinations(false, int32_t{}, uint32_t{}));
    EXPECT_TRUE(TestVecScalarCombinations(false, int32_t{}, uint16_t{}));
    EXPECT_TRUE(TestVecScalarCombinations(false, int16_t{}, uint16_t{}));

    // These two should fail regardless because of signed/unsigned mismatch
    EXPECT_TRUE(TestVecScalarCombinations(false, int16_t{}, uint32_t{}));
    EXPECT_TRUE(TestVecScalarCombinations(false, uint16_t{}, int32_t{}));

    // Now integers where rhs is a wider type is a special case.  If the
    // rhs is a scalar we permit it as a convenience.  If it's a vector
    // then not.  Unfortunately we still need to set up a mini lambda
    // instead of doing this inline, to as commas in the template list
    // muck up the macro
    auto TestSingle = [](auto&& t1, auto&& t2)
    {
        return CheckCompiles<Op::template type,
                             std::decay_t<decltype(t1)>,
                             std::decay_t<decltype(t2)>>::value;
    };
    EXPECT_TRUE(TestSingle(LaneArray<int16_t>{}, int32_t{}));
    EXPECT_TRUE(TestSingle(LaneArray<uint16_t>{}, uint32_t{}));
    EXPECT_FALSE(TestSingle(LaneArray<int16_t>{}, LaneArray<int32_t>{}));
    EXPECT_FALSE(TestSingle(LaneArray<uint16_t>{}, LaneArray<uint32_t>{}));

    // And a float cannot be on the right unless it's also on the left
    EXPECT_TRUE(TestVecScalarCombinations(false, int16_t{}, float{}));
    EXPECT_TRUE(TestVecScalarCombinations(false, int16_t{}, float{}));
    EXPECT_TRUE(TestVecScalarCombinations(false, uint16_t{}, float{}));
    EXPECT_TRUE(TestVecScalarCombinations(false, uint16_t{}, float{}));

}

// TODO: Comment below indicates that conversion from LaneArray<float> to
// LaneArray<Integer> should perform truncation (round to zero or round to
// -Infinity), conversions to signed integers actually round to nearest. The
// primary perpetrators of this atrocity are `operator m512i` and `operator
// m512ui` members of `m512f`.

// Make sure that we can convert as expected between LaneArrays of
// different types.  This should mimic scalar conversions as much
// as possible.  The most notible exception is that float -> int
// conversions behave differently if the destination type cannot
// represent the source value.  SIMD types basically return the
// max int rather than doing some form of truncation
TEST(LaneArray, Conversions)
{
    // Takes an input type and converts it to all valid output types,
    // making sure that the results are as expected
    auto cvtToAll = [&](const auto& in)
    {
        bool allPass = true;
        bool pass = true;

        // Does an element by element comparison between the original
        // and the conversion.  Returns true only if all of them are
        // true, but will also trigger a gtest value on individual
        // elements to help diagnose error instances
        auto test = [](const auto& convArr, const auto& origArr)
        {
            using conv_t = PacBio::Simd::ScalarType<std::decay_t<decltype(convArr)>>;
            using orig_t = PacBio::Simd::ScalarType<std::decay_t<decltype(origArr)>>;
            bool success = true;

            const auto& convData = convArr.ToArray();
            const auto& origData = origArr.ToArray();

            constexpr bool isFloatToInteger = std::is_same_v<orig_t, float> && std::is_integral_v<conv_t>;
            for (size_t i = 0; i < origData.size(); ++i)
            {
                // Need special handling of float-to-integer conversions.
                if constexpr (isFloatToInteger)
                {
                    // If the input value is out of range of the int, then skip that
                    // element.  SIMD instructions handle that differently from rint,
                    // and since some of our SSE "instructions" are emulated due to missing
                    // intrinsics, we won't even have parity between SSE and AVX512 code
                    // paths.
                    const bool outRange = origData[i] < static_cast<float>(std::numeric_limits<conv_t>::min())
                                          || origData[i] > static_cast<float>(std::numeric_limits<conv_t>::max());
                    if (outRange) continue;

                    const auto odi = origData[i];
                    const auto expect = numeric_cast<conv_t>(
                            std::is_signed_v<conv_t> ? std::rint(odi) : odi);
                    EXPECT_EQ(expect, convData[i])
                        << "  i is " << i
                        << ", origData[i] is " << origData[i];
                    success &= (expect == convData[i]);
                }
                else
                {
                    // All other cases should be handled by static_cast just fine
                    EXPECT_EQ(static_cast<conv_t>(origData[i]), convData[i])
                        << i << ":  " << origData[i];
                    success &= (static_cast<conv_t>(origData[i]) == convData[i]);
                }
            }
            return success;
        };

        // Take the input and convert it to all of the output types.
        // Again return true only if everyone succeeds, but also
        // cause a gtest failure for any individual errors.
        pass = test(LaneArray<float>(in), in);
        EXPECT_TRUE(pass) << "float conv failed";
        allPass &= pass;

        pass = test(LaneArray<int32_t>(in), in);
        EXPECT_TRUE(pass) << "int32_t conv failed";
        allPass &= pass;

        pass = test(LaneArray<uint32_t>(in), in);
        EXPECT_TRUE(pass) << "uint32_t conv failed";
        allPass &= pass;

        pass = test(LaneArray<int16_t>(in), in);
        EXPECT_TRUE(pass) << "int16_t conv failed";
        allPass &= pass;

        pass = test(LaneArray<uint16_t>(in), in);
        EXPECT_TRUE(pass) << "uint16_t conv failed";
        allPass &= pass;

        return allPass;
    };

    // Now that the testing harness is set up, hand pick
    // a few different values for each of our types.  We
    // intentionally want some of the input to be out of
    // bounds for some of the output.  We'll also be
    // replicating these out to fill a LaneArray, so
    // choosing an odd number (5) so that it doesn't tile
    // evenly into any SIMD variables.  This could
    // potentially flush out errors where 16 <-> 32 bit
    // conversions swap the high/low halves or something.
    std::array<float, 9> fltPattern = {
        12.1f, 12.5f, 12.9f,
        -12.1f, -12.5f, -12.9f,
        2.f*std::numeric_limits<int16_t>::max(),
        2.f*std::numeric_limits<int16_t>::min(),
        1.5f*std::numeric_limits<int32_t>::max()};

    std::array<int32_t, 5> intPattern = {
        15, -15,
        2*std::numeric_limits<int16_t>::max(),
        2*std::numeric_limits<int16_t>::min(),
        std::numeric_limits<int32_t>::max()};

    std::array<uint32_t, 5> uintPattern = {
        15, 31,
        2*std::numeric_limits<int16_t>::max(),
        3*std::numeric_limits<int16_t>::max(),
        std::numeric_limits<uint32_t>::max()};

    std::array<int16_t, 5> shortPattern = {
        1, -7, 59, -13241,
        std::numeric_limits<int16_t>::max()};

    std::array<uint16_t, 5> ushortPattern = {
        1, 7, 59, 13241,
        std::numeric_limits<uint16_t>::max()};

    AlignedCudaArray<float, laneSize>    fltData;
    AlignedCudaArray<int32_t, laneSize>  intData;
    AlignedCudaArray<uint32_t, laneSize> uintData;
    AlignedCudaArray<int16_t, laneSize>  shortData;
    AlignedCudaArray<uint16_t, laneSize> ushortData;

    for (size_t i = 0; i < laneSize; ++i)
    {
        fltData[i] = fltPattern[i % fltPattern.size()];
        intData[i] = intPattern[i%5];
        uintData[i] = uintPattern[i%5];
        shortData[i] = shortPattern[i%5];
        ushortData[i] = ushortPattern[i%5];
    }

    EXPECT_TRUE(cvtToAll(LaneArray<float, laneSize>(fltData)));
    EXPECT_TRUE(cvtToAll(LaneArray<int32_t, laneSize>(intData)));
    EXPECT_TRUE(cvtToAll(LaneArray<uint32_t, laneSize>(uintData)));
    EXPECT_TRUE(cvtToAll(LaneArray<int16_t, laneSize>(shortData)));
    EXPECT_TRUE(cvtToAll(LaneArray<uint16_t, laneSize>(ushortData)));
}

// The above has a lot of "procedurally generated" tests, which
// really are the only way to be evenly remotely exhaustive with
// the large number of overloads available to LaneArray operations.
// Still, there's always the chance that bugs in the generic testing
// code cause testing gaps.  What exists already helped me find and
// fix a number of bugs in LaneArray so I'm relatively confidential
// in them as being robust, but still here I'll put a small
// smattering of miscelanous hand-rolled tests.  Nothing nearly as
// exhaustive, but something hopefully immune to any systematic
// flaws that may otherwise be caused by bugs in abstracted and
// generic testing code.
TEST(LaneArray, Misc)
{
    LaneArray<int> a{1};
    a += 3;
    EXPECT_TRUE(all(a == 4));

    auto b = (a+1) / 2.0f;
    static_assert(std::is_same<decltype(b), LaneArray<float>>::value,
                  "expected promotion to float");
    EXPECT_TRUE(all(b == 2.5f));

    AlignedCudaArray<float, 64> bArr = b.ToArray();
    for (size_t i = 0; i < 32; ++i)
    {
        bArr[i] *= -1;
    }

    b = LaneArray<float>(bArr);
    EXPECT_TRUE(any(b >= 0));
    EXPECT_FALSE(all(b >= 0));

    b = Blend(b < 0, 0.0f, b);
    EXPECT_TRUE(any(b >= 0));
    EXPECT_TRUE(all(b >= 0));

    LaneArray<int16_t> c{1};

    auto d = c / 2.0f;
    static_assert(std::is_same<decltype(d), LaneArray<float>>::value,
                  "expected promotion to float");

    auto e = c + LaneArray<int32_t>{1};
    EXPECT_TRUE(all(e == 2));
    static_assert(std::is_same<decltype(e), LaneArray<int32_t>>::value,
                  "expected promotion to int");

    auto f = c + 1;
    EXPECT_TRUE(all(f == 2));
    static_assert(std::is_same<decltype(f), LaneArray<int16_t>>::value,
                  "expected NO promotion to int");

    //                        4 * (2.5 + 1) / 0.5 + 7 = 35
    //                   or   4 * (0 + 1) / 0.5 + 7   = 15
    auto complexExpression = (a * (b + c) / d) + 7;
    static_assert(std::is_same<decltype(complexExpression), LaneArray<float>>::value,
                  "expected float");

    auto mask1 = (complexExpression == 35);
    auto mask2 = (complexExpression == 15);
    EXPECT_TRUE(any(mask1));
    EXPECT_TRUE(any(mask2));
    EXPECT_FALSE(all(mask1));
    EXPECT_FALSE(all(mask2));
    EXPECT_TRUE(all(mask1 ^ mask2));
}
