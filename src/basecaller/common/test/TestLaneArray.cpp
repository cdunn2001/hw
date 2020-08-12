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
template<> Params<uint32_t> LaneArrayHomogeneousTypes<uint32_t>::params_{
    LinearArray<uint32_t>{3, 91},
    LinearArray<uint32_t>{17, 22},
        3,
        7};
template<> Params<uint16_t> LaneArrayHomogeneousTypes<uint16_t>::params_{
    LinearArray<uint16_t>{10, 3},
    LinearArray<uint16_t>{20, 9},
        4,
        18};

using ArrTypes = ::testing::Types<short, int, float, uint16_t, uint32_t>;
TYPED_TEST_SUITE(LaneArrayHomogeneousTypes, ArrTypes);

TYPED_TEST(LaneArrayHomogeneousTypes, Arithmetic)
{
    auto& params = this->params_;
    using T = typename TestFixture::type;

    const auto v1 = IncreasingLaneArray<T>(params.vec1.initial, params.vec1.stride);
    const auto v2 = IncreasingLaneArray<T>(params.vec2.initial, params.vec2.stride);

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

TEST(LaneArray, OperationResultTypes)
{
    LaneArray<float, laneSize> fltArr;
    LaneArray<int32_t, laneSize> intArr;
    LaneArray<uint32_t, laneSize> uintArr;
    LaneArray<int16_t, laneSize> shortArr;
    LaneArray<uint16_t, laneSize> ushortArr;

    auto CheckType = [](auto&& result, auto&& expected)
    {
        return std::is_same<std::decay_t<decltype(result)>, std::decay_t<decltype(expected)>>::value;
    };

    EXPECT_TRUE(CheckType(fltArr*fltArr,   LaneArray<float, laneSize>{}));
    EXPECT_TRUE(CheckType(intArr*fltArr,   LaneArray<float, laneSize>{}));
    EXPECT_TRUE(CheckType(fltArr*intArr,   LaneArray<float, laneSize>{}));
    EXPECT_TRUE(CheckType(uintArr*fltArr,   LaneArray<float, laneSize>{}));
    EXPECT_TRUE(CheckType(fltArr*uintArr,   LaneArray<float, laneSize>{}));
    EXPECT_TRUE(CheckType(shortArr*fltArr, LaneArray<float, laneSize>{}));
    EXPECT_TRUE(CheckType(fltArr*shortArr, LaneArray<float, laneSize>{}));
    EXPECT_TRUE(CheckType(ushortArr*fltArr, LaneArray<float, laneSize>{}));
    EXPECT_TRUE(CheckType(fltArr*ushortArr, LaneArray<float, laneSize>{}));
    EXPECT_TRUE(CheckType(1.0f*fltArr,     LaneArray<float, laneSize>{}));
    EXPECT_TRUE(CheckType(fltArr*1.0f,     LaneArray<float, laneSize>{}));
    EXPECT_TRUE(CheckType(1*fltArr,        LaneArray<float, laneSize>{}));
    EXPECT_TRUE(CheckType(fltArr*1,        LaneArray<float, laneSize>{}));
    EXPECT_TRUE(CheckType(1u*fltArr,        LaneArray<float, laneSize>{}));
    EXPECT_TRUE(CheckType(fltArr*1u,        LaneArray<float, laneSize>{}));
    EXPECT_TRUE(CheckType((int16_t)1*fltArr, LaneArray<float, laneSize>{}));
    EXPECT_TRUE(CheckType(fltArr*(int16_t)1, LaneArray<float, laneSize>{}));
    EXPECT_TRUE(CheckType((uint16_t)1*fltArr, LaneArray<float, laneSize>{}));
    EXPECT_TRUE(CheckType(fltArr*(uint16_t)1, LaneArray<float, laneSize>{}));

    EXPECT_TRUE(CheckType(intArr*intArr,   LaneArray<int, laneSize>{}));
    EXPECT_TRUE(CheckType(shortArr*intArr, LaneArray<int, laneSize>{}));
    EXPECT_TRUE(CheckType(intArr*shortArr, LaneArray<int, laneSize>{}));
    EXPECT_TRUE(CheckType(1*intArr,        LaneArray<int, laneSize>{}));
    EXPECT_TRUE(CheckType(intArr*1,        LaneArray<int, laneSize>{}));
    EXPECT_TRUE(CheckType((short)1*intArr, LaneArray<int, laneSize>{}));
    EXPECT_TRUE(CheckType(intArr*(short)1, LaneArray<int, laneSize>{}));

    EXPECT_TRUE(CheckType(uintArr*uintArr,   LaneArray<uint32_t, laneSize>{}));
    EXPECT_TRUE(CheckType(ushortArr*uintArr, LaneArray<uint32_t, laneSize>{}));
    EXPECT_TRUE(CheckType(uintArr*ushortArr, LaneArray<uint32_t, laneSize>{}));
    EXPECT_TRUE(CheckType(1u*uintArr,        LaneArray<uint32_t, laneSize>{}));
    EXPECT_TRUE(CheckType(uintArr*1u,        LaneArray<uint32_t, laneSize>{}));
    EXPECT_TRUE(CheckType((unsigned short)1*uintArr, LaneArray<uint32_t, laneSize>{}));
    EXPECT_TRUE(CheckType(uintArr*(unsigned short)1, LaneArray<uint32_t, laneSize>{}));

    EXPECT_TRUE(CheckType(shortArr*shortArr, LaneArray<short, laneSize>{}));
    EXPECT_TRUE(CheckType(short(1)*shortArr, LaneArray<short, laneSize>{}));
    EXPECT_TRUE(CheckType(shortArr*short(1), LaneArray<short, laneSize>{}));

    EXPECT_TRUE(CheckType(ushortArr*ushortArr, LaneArray<uint16_t, laneSize>{}));
    EXPECT_TRUE(CheckType(uint16_t(1)*ushortArr, LaneArray<uint16_t, laneSize>{}));
    EXPECT_TRUE(CheckType(ushortArr*uint16_t(1), LaneArray<uint16_t, laneSize>{}));

    // This is the exception.  Don't let a scalar int promote a shortArray
    EXPECT_TRUE(CheckType(1*shortArr, LaneArray<short, laneSize>{}));
    EXPECT_TRUE(CheckType(shortArr*1, LaneArray<short, laneSize>{}));
    EXPECT_TRUE(CheckType(1u*ushortArr, LaneArray<unsigned short, laneSize>{}));
    EXPECT_TRUE(CheckType(ushortArr*1u, LaneArray<unsigned short, laneSize>{}));

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
    EXPECT_TRUE(all(intSignedNeg1 < intSignedZero));
    EXPECT_TRUE(all(intSignedNeg1 <= intSignedZero));
    EXPECT_TRUE(none(intSignedNeg1 > intSignedZero));
    EXPECT_TRUE(none(intSignedNeg1 >= intSignedZero));

    EXPECT_TRUE(all(shortSignedNeg1 < shortSignedZero));
    EXPECT_TRUE(all(shortSignedNeg1 <= shortSignedZero));
    EXPECT_TRUE(none(shortSignedNeg1 > shortSignedZero));
    EXPECT_TRUE(none(shortSignedNeg1 >= shortSignedZero));

    EXPECT_TRUE(none(shortUnsignedNeg1 < shortUnsignedZero));
    EXPECT_TRUE(none(shortUnsignedNeg1 <= shortUnsignedZero));
    EXPECT_TRUE(all(shortUnsignedNeg1 > shortUnsignedZero));
    EXPECT_TRUE(all(shortUnsignedNeg1 >= shortUnsignedZero));

    EXPECT_TRUE(none(intUnsignedNeg1 < intUnsignedZero));
    EXPECT_TRUE(none(intUnsignedNeg1 <= intUnsignedZero));
    EXPECT_TRUE(all(intUnsignedNeg1 > intUnsignedZero));
    EXPECT_TRUE(all(intUnsignedNeg1 >= intUnsignedZero));

    // Now check division
    EXPECT_TRUE(all(shortSignedNeg2 / shortSignedNeg1 == static_cast<int16_t>(2)));
    EXPECT_TRUE(all(intSignedNeg2 / intSignedNeg1 == static_cast<int32_t>(2)));
    EXPECT_TRUE(none(shortUnsignedNeg2 / shortUnsignedNeg1 == static_cast<uint16_t>(2)));
    EXPECT_TRUE(none(intUnsignedNeg2 / intUnsignedNeg1 == static_cast<uint32_t>(2)));

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

template <template <typename, typename> class F, typename T1, typename T2, typename Result = void>
struct CheckCompiles : public std::false_type {};
template <template <typename, typename> class F, typename T1, typename T2>
struct CheckCompiles<F, T1, T2, sink_t<F<T1, T2>>> : public std::true_type {};

template <typename T1, typename T2>
using Multiply = decltype(std::declval<const T1&>() * std::declval<const T2&>());

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

TYPED_TEST(LaneArrayBinaryOps, Valid)
{
    using Op = typename TestFixture::type;


    // Helper function to do the scalar/vector combinatorics
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
    auto TestAllCombinations = [&](bool expectedToCompile, auto&& t1, auto&& t2)
    {
        return PermuteScalarVector(expectedToCompile, t1, t2)
             & PermuteScalarVector(expectedToCompile, t2, t1);
    };

    EXPECT_TRUE(TestAllCombinations(true, float{}, float{}));
    EXPECT_TRUE(TestAllCombinations(true, float{}, int32_t{}));
    EXPECT_TRUE(TestAllCombinations(true, float{}, uint32_t{}));
    EXPECT_TRUE(TestAllCombinations(true, float{}, int16_t{}));
    EXPECT_TRUE(TestAllCombinations(true, float{}, uint16_t{}));

    EXPECT_TRUE(TestAllCombinations(true, int32_t{}, int32_t{}));
    EXPECT_TRUE(TestAllCombinations(true, int32_t{}, int16_t{}));

    EXPECT_TRUE(TestAllCombinations(true, uint32_t{}, uint32_t{}));
    EXPECT_TRUE(TestAllCombinations(true, uint32_t{}, uint16_t{}));

    EXPECT_TRUE(TestAllCombinations(true, uint16_t{}, uint16_t{}));
    EXPECT_TRUE(TestAllCombinations(true, int16_t{}, int16_t{}));

    // Now the expected failures.  These combinations should
    // cause a compilation error
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

TYPED_TEST(LaneArrayCompoundOps, Valid)
{
    using Op = typename TestFixture::type;

    // Helper function to do the scalar/vector combinatorics
    auto TestVecScalarCombinations = [](bool expectedToCompile, auto&& t1, auto&& t2)
    {
        using T1 = std::decay_t<decltype(t1)>;
        using T2 = std::decay_t<decltype(t2)>;
        bool success = true;
        success &= (!expectedToCompile) xor CheckCompiles<Op::template type, LaneArray<T1, laneSize>, LaneArray<T2, laneSize>>::value;
        success &= (!expectedToCompile) xor CheckCompiles<Op::template type, LaneArray<T1, laneSize>, T2>::value;
        return success;
    };

    EXPECT_TRUE(TestVecScalarCombinations(true, float{}, float{}));
    EXPECT_TRUE(TestVecScalarCombinations(true, float{}, int32_t{}));
    EXPECT_TRUE(TestVecScalarCombinations(true, float{}, uint32_t{}));
    EXPECT_TRUE(TestVecScalarCombinations(true, float{}, int16_t{}));
    EXPECT_TRUE(TestVecScalarCombinations(true, float{}, uint16_t{}));

    EXPECT_TRUE(TestVecScalarCombinations(true, int32_t{}, int32_t{}));
    EXPECT_TRUE(TestVecScalarCombinations(true, int32_t{}, int16_t{}));

    EXPECT_TRUE(TestVecScalarCombinations(true, uint32_t{}, uint32_t{}));
    EXPECT_TRUE(TestVecScalarCombinations(true, uint32_t{}, uint16_t{}));

    EXPECT_TRUE(TestVecScalarCombinations(true, uint16_t{}, uint16_t{}));
    EXPECT_TRUE(TestVecScalarCombinations(true, int16_t{}, int16_t{}));

    // Now the expected failures.  These combinations should
    // cause a compilation error
    EXPECT_TRUE(TestVecScalarCombinations(false, int32_t{}, uint32_t{}));
    EXPECT_TRUE(TestVecScalarCombinations(false, int32_t{}, uint16_t{}));

    EXPECT_TRUE(TestVecScalarCombinations(false, int16_t{}, uint16_t{}));
}

TEST(LaneArray, Conversions)
{
    auto cvtToAll = [&](const auto& in)
    {
        bool allPass = true;
        bool pass = true;

        auto test = [](const auto& convArr, const auto& origArr)
        {
            using conv_t = ScalarType<std::decay_t<decltype(convArr)>>;
            using orig_t = ScalarType<std::decay_t<decltype(origArr)>>;
            bool success = true;

            const auto& convData = convArr.ToArray();
            const auto& origData = origArr.ToArray();
            // Make sure it's representble.
            const bool isFloat = std::is_same<orig_t, float>::value;
            for (size_t i = 0; i < origData.size(); ++i)
            {
                const bool outRange = static_cast<float>(origData[i]) < std::numeric_limits<conv_t>::lowest()
                                   || static_cast<float>(origData[i]) > std::numeric_limits<conv_t>::max();
                if (isFloat && outRange)
                    continue;
                else
                {
                    bool elemTest = convData[i] == static_cast<conv_t>(origData[i]);
                    EXPECT_TRUE(elemTest) << i << ": "
                                          << convData[i] << " "
                                          << static_cast<conv_t>(origData[i])
                                          << " " << origData[i];
                    success &= elemTest;
                }

            }
            return success;
        };

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

    std::array<float, 5> fltPattern = {
        12.5f, -12.5f,
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

    CudaArray<float, laneSize>    fltData;
    CudaArray<int32_t, laneSize>  intData;
    CudaArray<uint32_t, laneSize> uintData;
    CudaArray<int16_t, laneSize>  shortData;
    CudaArray<uint16_t, laneSize> ushortData;

    for (size_t i = 0; i < laneSize; ++i)
    {
        fltData[i] = fltPattern[i%5];
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
