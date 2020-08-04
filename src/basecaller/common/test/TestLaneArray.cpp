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

#include <functional>

#include <gtest/gtest.h>

#include <common/LaneArray.h>
#include <common/simd/ArrayUnion.h>

using namespace PacBio::Mongo;
using namespace PacBio::Simd;
using namespace PacBio::Cuda::Utility;

namespace {

template <typename T, typename U, size_t Len = laneSize>
CudaArray<T, Len> ConstantCudaArray(U val)
{
    CudaArray<T, Len> ret;
    for (size_t i = 0; i < Len; ++i)
    {
        ret[i] = val;
    }
    return ret;
}

template <typename T, typename U, size_t Len = laneSize>
LaneArray<T, Len> ConstantLaneArray(U val)
{
    return ConstantCudaArray(val);
}

template <typename T, typename U, size_t Len = laneSize>
CudaArray<T, Len> IncreasingCudaArray(U startVal, U increment)
{
    CudaArray<T, Len> ret;
    for (size_t i = 0; i < Len; ++i)
    {
        ret[i] = startVal + i * increment;
    }
    return ret;
}

template <typename T, typename U, size_t Len = laneSize>
LaneArray<T, Len> IncreasingLaneArray(U val, U increment)
{
    return LaneArray<T, Len>(IncreasingCudaArray<T, U, Len>(val, increment));
}

template <size_t Len = laneSize>
CudaArray<bool, Len> AlternatingBools()
{
    CudaArray<bool, Len> ret;
    for (size_t i = 0; i < Len; ++i)
    {
        ret[i] = (i % 2 == 0);
    }
    return ret;
}

}


TEST(BaseArray, TrivialDefaultConstruction)
{
    static_assert(std::is_trivially_default_constructible<LaneArray<float, laneSize>>::value, "Failed trivial test");
    static_assert(std::is_trivially_default_constructible<LaneArray<int, laneSize>>::value, "Failed trivial test");
    static_assert(std::is_trivially_default_constructible<LaneArray<short, laneSize>>::value, "Failed trivial test");
    static_assert(std::is_trivially_default_constructible<LaneMask<laneSize>>::value, "Failed trivial test");
}

TEST(BaseArray, UniformConstruction)
{
    auto Validate = [](const auto& array, auto val)
    {
        auto check = MakeUnion(array);
        for (const auto& v : check) EXPECT_EQ(v, val);
    };

    LaneArray<float, laneSize> arr1(12.5f);
    Validate(arr1, 12.5f);

    LaneArray<int, laneSize> arr2(-13);
    Validate(arr2, -13);

    LaneArray<short, laneSize> arr3(-9);
    Validate(arr3, -9);

    LaneMask<laneSize> arr4(true);
    for(size_t i = 0; i < laneSize; ++i) EXPECT_TRUE(arr4[i]);
}

TEST(BaseArray, ArrayConstruction)
{
    auto Validate = [](const auto& laneArray, const auto& cudaArray)
    {
        auto check = MakeUnion(laneArray);
        for (size_t i = 0; i < laneSize; ++i)
        {
            EXPECT_EQ(check[i], cudaArray[i]);
        }
    };

    auto c1 = IncreasingCudaArray<float>(1.3f, 1.1f);
    LaneArray<float, laneSize> arr1(c1);
    Validate(arr1, c1);

    auto c2 = IncreasingCudaArray<int>(1, 2);
    LaneArray<int, laneSize> arr2(c2);
    Validate(arr2, c2);

    auto c3 = IncreasingCudaArray<short>(3, 5);
    LaneArray<short, laneSize> arr3(c3);
    Validate(arr3, c3);

    auto c4 = AlternatingBools();
    LaneMask<laneSize> arr4(c4);
    for (size_t i = 0; i < laneSize; ++i)
    {
        EXPECT_TRUE(c4[i] == arr4[i]);
    }
}

TEST(BaseArray, PointerConstruction)
{
    auto Validate = [](const auto& laneArray, const auto& cudaArray)
    {
        auto check = MakeUnion(laneArray);
        for (size_t i = 0; i < laneSize; ++i)
        {
            EXPECT_EQ(check[i], cudaArray[i]);
        }
    };

    alignas(64) auto c1 = IncreasingCudaArray<float>(1.3f, 1.1f);
    LaneArray<float, laneSize> arr1(PtrView<float, laneSize>(c1.data()));
    Validate(arr1, c1);

    alignas(64) auto c2 = IncreasingCudaArray<int>(1, 2);
    LaneArray<int, laneSize> arr2(PtrView<int, laneSize>(c2.data()));
    Validate(arr2, c2);

    alignas(64) auto c3 = IncreasingCudaArray<short>(3, 5);
    LaneArray<short, laneSize> arr3(PtrView<short, laneSize>(c3.data()));
    Validate(arr3, c3);
}

TEST(BaseArray, ToCudaArray)
{
    auto Validate = [](const auto& cudaArr, const auto& laneArr)
    {
        auto expected = MakeUnion(laneArr);
        for (size_t i = 0; i < laneSize; ++i)
        {
            EXPECT_EQ(cudaArr[i], expected[i]);
        }
    };

    alignas(64) auto l1 = IncreasingLaneArray<float>(1.3f, 1.1f);
    CudaArray<float, laneSize> arr1 = l1;
    Validate(arr1, l1);

    alignas(64) auto l2 = IncreasingLaneArray<int>(1, 2);
    CudaArray<int, laneSize> arr2 = l2;
    Validate(arr2, l2);

    alignas(64) auto l3 = IncreasingLaneArray<short>(3, 5);
    CudaArray<short, laneSize> arr3 = l3;
    Validate(arr3, l3);
}

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

    CudaArray<bool, laneSize> pattern1;
    CudaArray<bool, laneSize> pattern2;
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
        EXPECT_TRUE(mask3[i] == (mask1[i] | mask2[i]));
    }

    mask3 = mask1 & mask2;
    for (size_t i = 0; i < laneSize; ++i)
    {
        EXPECT_TRUE(mask3[i] == (mask1[i] & mask2[i]));
    }

    mask3 = !mask1;
    for (size_t i = 0; i < laneSize; ++i)
    {
        EXPECT_TRUE(mask3[i] == (!mask1[i]));
    }

    mask3 = mask1 | mask2;
    mask1 |= mask2;
    for (size_t i = 0; i < laneSize; ++i)
    {
        EXPECT_TRUE(mask3[i] == mask1[i]);
    }

    mask1 = LaneMask<laneSize>(pattern1);
    mask3 = mask1 & mask2;
    mask1 &= mask2;
    for (size_t i = 0; i < laneSize; ++i)
    {
        EXPECT_TRUE(mask3[i] == mask1[i]);
    }
}

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

    static Params<T> params_;
};

template <typename Op, typename Result, typename Arg1, typename Arg2, size_t Len>
bool ValidateOp(const Result& actual,
                      const LaneArray<Arg1, Len>& left,
                      const LaneArray<Arg2, Len>& right)
{
    Op op{};
    auto a = static_cast<CudaArray<ScalarType<Result>, Len>>(actual);
    auto l = MakeUnion(left);
    auto r = MakeUnion(right);
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
    auto a = static_cast<CudaArray<ScalarType<Result>, Len>>(actual);
    auto l = left;
    auto r = MakeUnion(right);
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
    auto a = static_cast<CudaArray<ScalarType<Result>, Len>>(actual);
    auto l = MakeUnion(left);
    auto r = right;
    bool correct = true;
    for (size_t i = 0; i < Len; ++i)
    {
        correct &= (a[i] == op(l[i], r));
    }
    return correct;
}

template<> Params<float> LaneArrayHomogeneousTypes<float>::params_{
    LinearArray<float>{13.5f, 1.5f},
    LinearArray<float>{9.25f, 1.75f},
        4.f,
        6.5f};
template<> Params<int> LaneArrayHomogeneousTypes<int>::params_{
    LinearArray<int>{-270, 91},
    LinearArray<int>{17, 5},
        2,
        12};
template<> Params<short> LaneArrayHomogeneousTypes<short>::params_{
    LinearArray<short>{-10, 3},
    LinearArray<short>{-20, 9},
        -4,
        18};

using ArrTypes = ::testing::Types<short, int, float>;
TYPED_TEST_SUITE(LaneArrayHomogeneousTypes, ArrTypes);

TYPED_TEST(LaneArrayHomogeneousTypes, Arithmetic)
{
    auto& params = this->params_;
    using T = typename TestFixture::type;

    const auto v1 = IncreasingLaneArray<T>(params.vec1.initial, params.vec1.stride);
    const auto v2 = IncreasingLaneArray<T>(params.vec2.initial, params.vec2.stride);

    EXPECT_TRUE(all(-v1 == IncreasingLaneArray<T>(-params.vec1.initial, -params.vec1.stride)));

    {
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

        EXPECT_TRUE(all(inc(v1, mask) == Blend(mask, v1+static_cast<T>(1), v1)));
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

TYPED_TEST(LaneArrayHomogeneousTypes, Comparisons)
{
    auto& params = this->params_;
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

TEST(LaneArray, FloatOps)
{
    auto cudaArr = IncreasingCudaArray<float>(-40.2f, 13.9f);
    LaneArray<float, laneSize> laneArr(cudaArr);

    auto result = CudaArray<float, laneSize>(erfc(laneArr));
    for (size_t i = 0; i < laneSize; ++i)
    {
        EXPECT_FLOAT_EQ(result[i], std::erfc(cudaArr[i]));
    }

    auto result2 = CudaArray<int, laneSize>(floorCastInt(laneArr));
    for (size_t i = 0; i < laneSize; ++i)
    {
        EXPECT_EQ(result2[i], std::floor(cudaArr[i]));
    }

    // Get rid of negatives so we can take the sqrt and log
    cudaArr = IncreasingCudaArray<float>(2.9f, 13.9f);
    laneArr = LaneArray<float, laneSize>(cudaArr);

    result = CudaArray<float, laneSize>(sqrt(laneArr));
    for (size_t i = 0; i < laneSize; ++i)
    {
        EXPECT_FLOAT_EQ(result[i], std::sqrt(cudaArr[i]));
    }


    result = CudaArray<float, laneSize>(log(laneArr));
    for (size_t i = 0; i < laneSize; ++i)
    {
        EXPECT_FLOAT_EQ(result[i], std::log(cudaArr[i]));
    }

    result = CudaArray<float, laneSize>(log2(laneArr));
    for (size_t i = 0; i < laneSize; ++i)
    {
        EXPECT_FLOAT_EQ(result[i], std::log2(cudaArr[i]));
    }

    // Lower the dynamic range so we can take the exp
    // SSE and std:exp diverge a bit toward the extreme
    // end of the range
    cudaArr = IncreasingCudaArray<float>(-12.34f, .74f);
    laneArr = LaneArray<float, laneSize>(cudaArr);

    result = CudaArray<float, laneSize>(exp(laneArr));
    for (size_t i = 0; i < laneSize; ++i)
    {
        EXPECT_FLOAT_EQ(result[i], std::exp(cudaArr[i]));
    }

    result = CudaArray<float, laneSize>(exp2(laneArr));
    for (size_t i = 0; i < laneSize; ++i)
    {
        EXPECT_FLOAT_EQ(result[i], std::exp2(cudaArr[i]));
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
            EXPECT_TRUE(finiteMask[i]);
            EXPECT_FALSE(nanMask[i]);
        }
        if (i % 3 == 1)
        {
            EXPECT_FALSE(finiteMask[i]);
            EXPECT_FALSE(nanMask[i]);
        }
        if (i % 3 == 2)
        {
            EXPECT_FALSE(finiteMask[i]);
            EXPECT_TRUE(nanMask[i]);
        }
    }
}

TEST(LaneArray, IntOps)
{
    auto a = IncreasingLaneArray<int>(12, 23);
    auto b = IncreasingLaneArray<int>(3, 15);

    EXPECT_TRUE(ValidateOp<std::bit_or<int>>(a | b, a, b));
}

TEST(LaneArray, ValidConversions)
{
    LaneArray<float, laneSize> fltArr;
    LaneArray<int, laneSize> intArr;
    LaneArray<short, laneSize> shortArr;

    auto CheckType = [](auto&& result, auto&& expected)
    {
        return std::is_same<std::decay_t<decltype(result)>, std::decay_t<decltype(result)>>::value;
    };

    EXPECT_TRUE(CheckType(fltArr*fltArr,   LaneArray<float, laneSize>{}));
    EXPECT_TRUE(CheckType(intArr*fltArr,   LaneArray<float, laneSize>{}));
    EXPECT_TRUE(CheckType(fltArr*intArr,   LaneArray<float, laneSize>{}));
    EXPECT_TRUE(CheckType(shortArr*fltArr, LaneArray<float, laneSize>{}));
    EXPECT_TRUE(CheckType(fltArr*shortArr, LaneArray<float, laneSize>{}));
    EXPECT_TRUE(CheckType(1.0f*fltArr,     LaneArray<float, laneSize>{}));
    EXPECT_TRUE(CheckType(fltArr*1.0f,     LaneArray<float, laneSize>{}));
    EXPECT_TRUE(CheckType(1*fltArr,        LaneArray<float, laneSize>{}));
    EXPECT_TRUE(CheckType(fltArr*1,        LaneArray<float, laneSize>{}));
    EXPECT_TRUE(CheckType((short)1*fltArr, LaneArray<float, laneSize>{}));
    EXPECT_TRUE(CheckType(fltArr*(short)1, LaneArray<float, laneSize>{}));

    EXPECT_TRUE(CheckType(intArr*intArr,   LaneArray<int, laneSize>{}));
    EXPECT_TRUE(CheckType(shortArr*intArr, LaneArray<int, laneSize>{}));
    EXPECT_TRUE(CheckType(intArr*shortArr, LaneArray<int, laneSize>{}));
    EXPECT_TRUE(CheckType(1*intArr,        LaneArray<int, laneSize>{}));
    EXPECT_TRUE(CheckType(intArr*1,        LaneArray<int, laneSize>{}));
    EXPECT_TRUE(CheckType((short)1*intArr, LaneArray<int, laneSize>{}));
    EXPECT_TRUE(CheckType(intArr*(short)1, LaneArray<int, laneSize>{}));

    EXPECT_TRUE(CheckType(shortArr*shortArr, LaneArray<short, laneSize>{}));
    EXPECT_TRUE(CheckType(short(1)*shortArr, LaneArray<short, laneSize>{}));
    EXPECT_TRUE(CheckType(shortArr*short(1), LaneArray<short, laneSize>{}));

    // This is the exception.  Don't let a scalar int promote a shortArray
    EXPECT_TRUE(CheckType(1*shortArr, LaneArray<short, laneSize>{}));
    EXPECT_TRUE(CheckType(shortArr*1, LaneArray<short, laneSize>{}));

    // Float scalars do always cause promotions still
    EXPECT_TRUE(CheckType(1.f*intArr,   LaneArray<float, laneSize>{}));
    EXPECT_TRUE(CheckType(intArr*1.f,   LaneArray<float, laneSize>{}));
    EXPECT_TRUE(CheckType(1.f*shortArr, LaneArray<float, laneSize>{}));
    EXPECT_TRUE(CheckType(shortArr*1.f, LaneArray<float, laneSize>{}));
}
