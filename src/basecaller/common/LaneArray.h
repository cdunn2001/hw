// Copyright (c) 2019, Pacific Biosciences of California, Inc.
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

#ifndef mongo_common_LaneArray_H_
#define mongo_common_LaneArray_H_

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <array>

#include <common/MongoConstants.h>

#include <common/cuda/utility/CudaArray.h>
#include <common/simd/m512b.h>
#include <common/simd/m512f.h>
#include <common/simd/m512i.h>
#include <common/simd/m512s.h>
#include <common/simd/SimdConvTraits.h>
#include <common/simd/SimdTypeTraits.h>

namespace PacBio {
namespace Mongo {

// TODO fix namespaces?
using namespace Simd;

template <typename T>
struct vec_type;

// TODO remove unsigned hack/lie
template<> struct vec_type<float> { using type = m512f; };
template<> struct vec_type<int>   { using type = m512i; };
template<> struct vec_type<unsigned int>   { using type = m512i; };
template<> struct vec_type<bool>  { using type = m512b; };
template<> struct vec_type<short> { using type = m512s; };
template<> struct vec_type<unsigned short> { using type = m512s; };

template <typename T>
using vec_type_t = typename vec_type<T>::type;

// TODO clean up these Len usages?
template <typename T, size_t Len>
struct vec_count
{
    static_assert(Len % 16 == 0, "Invalid lenth");
    static constexpr size_t value = Len / 16;
};
template <size_t Len>
struct vec_count<short, Len>
{
    static_assert(Len % 32 == 0, "Invalid lenth");
    static constexpr size_t value = Len / 32;
};

template <typename T, size_t ScalarCount = laneSize>
class LaneArray;

struct Noop {};

template <typename T>
struct magic
{
    // Don't let integral scalars affect the return type.  This is primarily
    // to combat things like LaneArray<short> + int = LaneArray<int>, as
    // it's painfully difficult to type short literals
    using typee = std::conditional_t<std::is_integral<T>::value, Noop, ScalarType<T>>;
    // intel is stupid and requires this...  type eventually gets used in
    // `std::common_type`, and intel barfs if certain eigen types make it there
    // This is just a crude filter to replace any non-arithmetic types with void,
    // which for our purposes is just as good
    using typee2 = std::conditional_t<std::is_arithmetic<typee>::value
                                      || std::is_same<Noop, typee>::value,
                                      typee, void>;
    // Filter out double
    // temporarily force signed
    using type = std::conditional_t<!std::is_same<typee2, double>::value, typee2, void>;
};
template <typename T>
struct magic<ArrayUnion<T>>
{
    using type = typename magic<T>::type;
};

template <typename T>
using magic_t = typename magic<T>::type;

// TODO replace Noop with int8_t?
//template <typename T1, typename T2,
//          typename T11 = std::conditional_t<std::is_same<Noop,magic_t<T1>>::value,
//                                            magic_t<T2>,
//                                            magic_t<T1>
//                                            >,
//          typename T22 = std::conditional_t<std::is_same<Noop,magic_t<T2>>::value,
//                                            magic_t<T1>,
//                                            magic_t<T2>
//                                            >
//          >
//using magic2 = std::common_type_t<T11, T22>;

template <typename...Ts>
struct magic22
{
    static constexpr size_t IdxOfDefault()
    {
        bool isNoop[sizeof...(Ts)] {std::is_same<Noop, magic_t<Ts>>::value...};
        size_t ret = 0;
        for (size_t i = 0; i < sizeof...(Ts); ++i)
        {
            if (!isNoop[i]) ret = i;
        }
        return ret;
    }

    using DefType = magic_t<std::tuple_element_t<IdxOfDefault(), std::tuple<std::decay_t<Ts>...>>>;

    template <typename ...Us>
    auto Helper() ->
    std::common_type_t<std::conditional_t<std::is_same<magic_t<Us>, Noop>::value,
                                                       DefType,
                                                       magic_t<Us>>...>;
    //template <typename... Us>
    //static constexpr auto Helper2() ->

    //template <typename T = std::tuple<Ts...>>
    //static constexpr auto Helper()
    //{

    //}
};

template <typename...Ts>
using magic2 = decltype(std::declval<magic22<Ts...>>().template Helper<Ts...>());
//using magic2 = typename magic22<Ts...>::template type<Ts...>;

//template <typename T1, typename T2, typename T3, typename T4 = std::enable_if_t<std::is_same<ScalarType<T2>, magic2<T1, T2>>::value>>
//struct magic44 {
//    using type = magic2<T1, T2>;
//};

template <typename T1, typename T2, bool b>
struct magic44 {};

template <typename T1, typename T2>
struct magic44<T1, T2, true> {
    using type = magic2<T1, T2>;
};

template <typename T1, typename T2, typename T3>
using magic4 = typename magic44<T1, T2, std::is_same<ScalarType<T3>, magic2<T1, T2>>::value>::type;

template <typename T1, typename T2, typename T3, typename Ret>
using magic5 = std::enable_if_t<std::is_same<ScalarType<T3>, magic2<T1, T2>>::value, Ret>;

template <typename T> struct IsLaneArray
{ static constexpr bool value = false; };
template <typename T, size_t N> struct IsLaneArray<LaneArray<T, N>>
{ static constexpr bool value = true; };
template <typename T> struct IsLaneArray<ArrayUnion<T>>
{ static constexpr bool value = true; };

template <
    typename Arg1, typename Arg2, typename T, typename RetRequest = void,  // public API
    // SFINAE check to see if there is a sensible common type
    typename common = magic2<Arg1, Arg2>,
    //
    bool SingleLaneArray = IsLaneArray<Arg1>::value xor IsLaneArray<Arg2>::value,
    bool IsCommon = std::is_same<ScalarType<T>, common>::value,
    size_t VecLen = std::max((uint16_t)SimdTypeTraits<Arg1>::width, (uint16_t)SimdTypeTraits<Arg2>::width),
    typename Ret = std::conditional_t<std::is_same<void, RetRequest>::value, LaneArray<common, VecLen>, RetRequest>
    >
using SmartReturn = std::enable_if_t<SingleLaneArray || IsCommon, Ret>;

// No member for the default type.  We're going to use SFINAE
// to disable functions where mixed type arguments make
// no sense
template <typename T1, typename T2, typename ScalarMul = std::common_type_t<magic_t<T1>, magic_t<T2>>>
struct MixedType {
private:
    static constexpr auto N = std::max((uint16_t)SimdTypeTraits<T1>::width, (uint16_t)SimdTypeTraits<T2>::width);
public:
    using ArithmeticType = LaneArray<ScalarMul, N>;
    // TODO this won't work
    //using CompareType = LaneArray
};

//template <typename T1, typename T2>
//struct MixedType<ArrayUnion<T1>, ArrayUnion<T2>>
//{
//    using ArithmeticType = typename MixedType<T1, T2>::ArithmeticType;
//};
//template <typename T1, typename T2>
//struct MixedType<ArrayUnion<T1>, T2>
//{
//    using ArithmeticType = typename MixedType<T1, T2>::ArithmeticType;
//};
//template <typename T1, typename T2>
//struct MixedType<T1, ArrayUnion<T2>>
//{
//    using ArithmeticType = typename MixedType<T1, T2>::ArithmeticType;
//};

//template <size_t N>
//struct MixedType<LaneArray<float, N>,   LaneArray<float,N>>   { using type = LaneArray<float, N>; };
//template <size_t N>
//struct MixedType<float,                 LaneArray<float,N>>   { using type = LaneArray<float, N>; };
//template <size_t N>
//struct MixedType<LaneArray<float, N>,   float>                { using type = LaneArray<float, N>; };
//template <size_t N>
//struct MixedType<int32_t,               LaneArray<float,N>> { using type = LaneArray<float, N>; };
//template <size_t N>
//struct MixedType<LaneArray<float, N>, int32_t>              { using type = LaneArray<float, N>; };
//template <size_t N>
//struct MixedType<int16_t,               LaneArray<float,N>> { using type = LaneArray<float, N>; };
//template <size_t N>
//struct MixedType<LaneArray<int16_t, N>, int16_t>              { using type = LaneArray<int16_t, N>; };

template <typename T>
struct PairRef
{
    PairRef(T& t1, T& t2) : first(t1), second(t2) {}
    PairRef(const std::pair<T, T>& p) : first(p.first), second(p.second) {}

    template <typename Other,
              bool IsConst = std::is_const<T>::value,
              bool isPair = std::is_same<Other, std::pair<T,T>>::value,
              bool isRef = std::is_base_of<PairRef<const T>, Other>::value,
              std::enable_if_t<!IsConst && (isPair || isRef),int> = 0>
    PairRef& operator=(const Other& other)
    {
        first = other.first;
        second = other.second;
        return *this;
    }

    T& first;
    T& second;
};


// TODO move?
// TODO fix attrocious naming
inline m512s Blend(const PairRef<const m512b>& b, const m512s& l, const m512s& r)
{
    // TODO fix this redirection?
    return Blend(b.first, b.second, l, r);
}

template <bool...bs>
struct bool_pack {};

template <bool...bs>
using all_true = std::is_same<bool_pack<true, bs...>, bool_pack<bs..., true>>;

template <typename T>
struct len_trait;
template <typename T>
struct len_trait
{
    static constexpr size_t SimdCount = 1;
    static constexpr size_t ScalarCount = 1;
    static constexpr size_t SimdWidth = 1;
};

template <typename T>
constexpr size_t rename_me()
{
    constexpr auto tmp = len_trait<T>::SimdCount;
    return tmp == 1 ? std::numeric_limits<size_t>::max() : tmp;
}

template <typename Ret, typename...Args>
static constexpr size_t MinSimdCount()
{
    constexpr auto ret = std::min({rename_me<Ret>(), rename_me<Args>()...});
    static_assert(ret == 2 || ret == 4, "");
    return ret;
}
template <typename...Args>
static constexpr size_t MaxSimdCount()
{
    static_assert(sizeof...(Args) > 0,"");
    return std::max({(size_t)len_trait<Args>::SimdCount...});
}

// TODO clean/rethink?
template <typename T, size_t Len>
struct PtrView
{
    explicit PtrView(const T* data)
        : data_{data}
    {}

    const T*operator[](size_t idx) const { return data_ + idx; }
private:
    const T* data_;
};

template <typename T, size_t SimdCount_, typename Child>
class alignas(T) BaseArray
{
private:
    // Need to put static asserts checking the Child class within a function.
    // At the class scope, Child<VIn> is not yet a valid type.
    static constexpr void Valid()
    {
        static_assert(std::is_base_of<BaseArray, Child>::value,
                     "Improper use of BaseArray class");
        static_assert(sizeof(BaseArray) == sizeof(Child),
                     "Children of BaseArray cannot add their own members");
    }

public:
    static constexpr size_t SimdCount = SimdCount_;
    static constexpr size_t SimdWidth = SimdTypeTraits<T>::width;
    static constexpr size_t ScalarCount = SimdCount * SimdWidth;
    using SimdType = T;

    // TODO worry about alignment
    // Can't use this ctor for bools, as they are not bitwise equivalant
    template <typename U = T,
              bool isBool = std::is_same<ScalarType<U>, bool>::value,
              std::enable_if_t<!isBool, int> = 0>
    explicit BaseArray(const Cuda::Utility::CudaArray<ScalarType<T>, ScalarCount>& data)
    {
        constexpr auto width = SimdTypeTraits<T>::width;
        auto* dat = data.data();
        for (auto& d : data_)
        {
            d = T(dat);
            dat += width;
        }
    }

    template <typename U,
        typename Scalar = ScalarType<T>,
        typename dummy1 = std::enable_if_t<std::is_constructible<Scalar, U>::value>,
        typename dummy2 = std::enable_if_t<!std::is_same<Scalar, U>::value>>
    explicit BaseArray(const Cuda::Utility::CudaArray<U, ScalarCount>& data)
    {
        static constexpr auto width = SimdTypeTraits<T>::width;
        size_t inOffset = 0;
        for (auto& d : data_)
        {
            UnionConv<T> tmp;
            for (size_t i = 0; i < width; ++i)
            {
                tmp[i] = ScalarType<T>(data[inOffset + i]);
            }
            d = tmp;
            inOffset += width;
        }
    }

    template <typename U, std::enable_if_t<std::is_convertible<U, ScalarType<T>>::value, int> = 0>
    BaseArray(U&& val)
    {
        for (auto& d : data_)
        {
            d = ScalarType<T>(val);
        }
    }

    template <typename U = ScalarType<T>,
        std::enable_if_t<!std::is_same<bool, U>::value, int> = 0
        >
    BaseArray(PtrView<U, ScalarCount> dat)
    {
        assert(static_cast<size_t>(dat[0]) % 64 == 0);
        for (size_t i = 0; i < SimdCount; ++i)
        {
            data_[i] = T(dat[i*SimdTypeTraits<T>::width]);
        }
    }

    // Intentionally not initializing our data
    // Would just kill this off, but it's unfortunately
    // required by Eigen at least.
    BaseArray() = default;

    // TODO rename/comment anonymous size_t
    template <size_t N, size_t, typename WorkingType, typename U,
              std::enable_if_t<N == len_trait<std::decay_t<U>>::SimdCount, int> = 0>
    static decltype(auto) Access(U&& val, size_t idx)
    {
        return (val.data()[idx]);
    }
    //template <size_t N, size_t, typename WorkingType, typename U,
    //          typename foo = std::conditional_t<std::is_same<WorkingType, Noop>::value,
    //                                            ScalarType<std::decay_t<U>>,
    //                                            WorkingType>,
    //          std::enable_if_t<std::is_same<typename std::decay_t<U>::SimdType, vec_type_t<foo>>::value, int> = 0,
    //          std::enable_if_t<N == len_trait<std::decay_t<U>>::SimdCount, int> = 0>
    //static decltype(auto) Access(U&& val, size_t idx)
    //{
    //    return (val.data()[idx]);
    //}

    //template <size_t N, size_t, typename WorkingType, typename U,
    //          typename foo = std::conditional_t<std::is_same<WorkingType, Noop>::value,
    //                                            ScalarType<std::decay_t<U>>,
    //                                            WorkingType>,
    //          std::enable_if_t<!std::is_same<typename std::decay_t<U>::SimdType, vec_type_t<foo>>::value, int> = 0,
    //          std::enable_if_t<N == len_trait<std::decay_t<U>>::SimdCount, int> = 0>
    //static decltype(auto) Access(U&& val, size_t idx)
    //{
    //    return WorkingType(val.data()[idx]);
    //}

    template <size_t N, size_t, typename WorkingType, typename U,
        std::enable_if_t<N == 2*len_trait<std::decay_t<U>>::SimdCount && (N > 2), int> = 0>
    static auto Access(U&& val, size_t idx)
    {
        if (idx%2 == 0) return Low<WorkingType>(val.data()[idx/2]);
        else return High<WorkingType>(val.data()[idx/2]);
    }

    template <size_t N, size_t, typename WorkingType, typename U,
              std::enable_if_t<2*N == len_trait<std::decay_t<U>>::SimdCount, int> = 0>
    static auto Access(U&& val, size_t idx)
    {
        // Ignoring workingtype.  TODO are there any good static asserts to put here?
        using SimdType = typename std::decay_t<U>::SimdType;
        using Ref_t = std::conditional_t<std::is_const<std::remove_reference_t<U>>::value, const SimdType, SimdType>;
        return PairRef<Ref_t>{val.data()[idx*2], val.data()[idx*2+1]};
    }

    //template <size_t N, size_t, typename WorkingType, typename U,
    //          std::enable_if_t<1 == len_trait<U>::SimdCount, int> = 0>
    //static decltype(auto) Access(std::remove_const_t<U>& val, size_t)
    //{
    //    return (val);
    //}

    template <size_t N, size_t ScalarStride, typename WorkingType, typename  U,
              std::enable_if_t<1 == len_trait<std::decay_t<U>>::SimdCount, int> = 0>
    static auto Access(U&& val, size_t)
    {
        return static_cast<vec_type_t<WorkingType>>(val);
    }

    template <size_t N, size_t N2, typename WorkingType, typename U>
    static decltype(auto) Access(const ArrayUnion<U>& t, size_t idx)
    {
        return Access<N,N2, WorkingType>(t.Simd(), idx);
    }

    template <typename F, typename...Args, std::enable_if_t<sizeof...(Args) != 0, int> = 0 >
    BaseArray(F&& f, const Args&... args)
    {
        static_assert(sizeof...(Args) > 0, "");
        static constexpr auto loopMax = std::min(MaxSimdCount<Args...>(), Child::SimdCount);
        static constexpr auto ScalarStride = ScalarCount / loopMax;
        using WorkingType = magic2<Args...>;//std::common_type_t<ScalarType<std::decay_t<Args>>...>;
        for (size_t i = 0; i < loopMax; ++i)
        {
            Access<loopMax, ScalarStride, Noop>(static_cast<Child&>(*this), i) = f(Access<loopMax, ScalarStride, WorkingType>(args, i)...);
        }
    }

    template <typename F, typename...Args>
    Child& Update(F&& f, const Args&... args)
    {
        static constexpr auto loopMax = MaxSimdCount<Child, Args...>();
        static constexpr auto ScalarStride = ScalarCount / loopMax;
        using WorkingType = ScalarType<Child>;
        for (size_t i = 0; i < loopMax; ++i)
        {
            f(Access<loopMax, ScalarStride, WorkingType>(static_cast<Child&>(*this), i), Access<loopMax, ScalarStride, WorkingType>(args, i)...);
        }
        return static_cast<Child&>(*this);
    }

    template <typename F, typename Ret, typename...Args>
    static Ret Reduce(F&& f, Ret initial, const Args&... args)
    {
        static constexpr auto loopMax = MaxSimdCount<Child, Args...>();
        static constexpr auto ScalarStride = ScalarCount / loopMax;
        using WorkingType = ScalarType<Child>;

        Ret ret = initial;
        for (size_t i = 0; i < loopMax; ++i)
        {
            f(ret, Access<loopMax, ScalarStride, WorkingType>(args, i)...);
        }
        return ret;
    }

    // TODO clean and check assembly
    // TODO remove implicit?
    template <
        typename U = ScalarType<T>,
        std::enable_if_t<!std::is_same<bool, U>::value, int> = 0>
    operator Cuda::Utility::CudaArray<ScalarType<T>, ScalarCount>() const
    {
        Cuda::Utility::CudaArray<ScalarType<T>, ScalarCount> ret;
        static_assert(sizeof(decltype(ret)) == sizeof(BaseArray), "");
        std::memcpy(&ret, this, sizeof(ret));
        return ret;
    }

    // TODO also SFINAE to check if convertible?
    template <typename U, std::enable_if_t<!std::is_same<U, ScalarType<T>>::value, int> = 0>
    operator Cuda::Utility::CudaArray<U, ScalarCount>() const
    {
        Cuda::Utility::CudaArray<U, ScalarCount> ret;
        auto tmp = MakeUnion(static_cast<const Child&>(*this));
        for (size_t i = 0; i < ScalarCount; ++i)
        {
            // TODO worry about narrowing Conversions
            ret[i] = U(tmp[i]);
        }
        return ret;
    }

    const std::array<T, SimdCount>& data() const
    {
        return data_;
    }
    std::array<T, SimdCount>& data()
    {
        return data_;
    }

protected:
    std::array<T, SimdCount> data_;
};

template<size_t ScalarCount_ = laneSize>
class LaneMask : public BaseArray<m512b, ScalarCount_/16, LaneMask<ScalarCount_>>
{
    static_assert(ScalarCount_ % 16 == 0, "");
    using Base = BaseArray<m512b, ScalarCount_/16, LaneMask<ScalarCount_>>;
public:
    using Base::Base;
    static constexpr auto SimdCount = Base::SimdCount;
    static constexpr auto SimdWidth = Base::SimdWidth;
    static constexpr auto ScalarCount = Base::ScalarCount;
    using SimdType = typename Base::SimdType;
    using Base::Update;
    using Base::Reduce;
    using Base::data;

    LaneMask() = default;
    LaneMask(const Cuda::Utility::CudaArray<bool, ScalarCount_>& arr)
    {
        for (size_t i = 0; i < SimdCount; ++i)
        {
            auto start = i * SimdWidth;
            data()[i] = m512b(arr[start+0],  arr[start+1],  arr[start+2],  arr[start+3],
                              arr[start+4],  arr[start+5],  arr[start+6],  arr[start+7],
                              arr[start+8],  arr[start+9],  arr[start+10], arr[start+11],
                              arr[start+12], arr[start+13], arr[start+14], arr[start+15]);
        }
    }

    explicit operator Cuda::Utility::CudaArray<bool, ScalarCount_>() const
    {
        Cuda::Utility::CudaArray<bool, ScalarCount_> ret;
        for (size_t i = 0; i < ScalarCount_; ++i)
        {
            ret[i] = (*this)[i];
        }
        return ret;
    }

public:
    friend bool all(const LaneMask& m)
    {
        return Reduce([](auto&& l, auto&& r) { l &= all(r); }, true, m);
    }

    friend bool any(const LaneMask& m)
    {
        return Reduce([](auto&& l, auto&& r) { l |= any(r); }, false, m);
    }

    friend bool none(const LaneMask& m)
    {
        return Reduce([](auto&& l, auto&& r) { l &= none(r); }, true, m);
    }

    friend LaneMask operator| (const LaneMask& l, const LaneMask& r)
    {
        return LaneMask(
            [](auto&& l2, auto&& r2){ return l2 | r2; },
            l, r);
    }

    friend LaneMask operator! (const LaneMask& m)
    {
        return LaneMask(
            [](auto&& m2){ return !m2; },
            m);
    }

    friend LaneMask operator& (const LaneMask& l, const LaneMask& r)
    {
        return LaneMask(
            [](auto&& l2, auto&& r2){ return l2 & r2; },
            l, r);
    }

    LaneMask& operator &= (const LaneMask& o)
    {
        return Update([](auto&& l, auto&& r) { l &= r; }, o);
    }

    LaneMask& operator |= (const LaneMask& o)
    {
        return Update([](auto&& l, auto&& r) { l |= r; }, o);
    }

    bool operator[](size_t idx) const
    {
        // TODO make m512b into T?
        static constexpr auto width = SimdTypeTraits<m512b>::width;
        // TODO validate idx
        return this->data_[idx / width][idx%width];
    }
};

template <typename T, size_t SimdCount_, typename Child>
class ArithmeticArray : public BaseArray<T, SimdCount_, Child>
{
    static_assert(std::is_arithmetic<ScalarType<T>>::value, "Arithmetic array requires an arithematic POD or simd type");
protected:
    using Base = BaseArray<T, SimdCount_, Child>;
    using Base::data_;
public:
    using Base::data;
    using Base::Base;
    using Base::Update;
    using Base::Reduce;
    static constexpr auto SimdCount = Base::SimdCount;
    static constexpr auto ScalarCount = Base::ScalarCount;

    template <typename Other, typename common = magic4<Child, Other, Child>>
    Child& operator+=(const Other& other)
    {
        return Update([](auto&& l, auto&& r) { l += r; }, other);
    }
    template <typename Other, typename common = magic4<Child, Other, Child>>
    Child& operator-=(const Other& other)
    {
        return Update([](auto&& l, auto&& r) { l -= r; }, other);
    }
    template <typename Other, typename common = magic4<Child, Other, Child>>
    Child& operator/=(const Other& other)
    {
        return Update([](auto&& l, auto&& r) { l /= r; }, other);
    }
    template <typename Other, typename common = magic4<Child, Other, Child>>
    Child& operator*=(const Other& other)
    {
        return Update([](auto&& l, auto&& r) { l *= r; }, other);
    }

    friend Child operator -(const Child& c)
    {
        return Child(
            [](auto&& d){ return -d;},
            c);
    }

    template <typename T1, typename T2>
    friend auto operator -(const T1& l, const T2& r) -> SmartReturn<T1, T2, Child>
    {
        return SmartReturn<T1, T2, Child>(
            [](auto&& l2, auto&& r2){ return l2 - r2;},
            l, r);
    }

    template <typename T1, typename T2>
    friend auto operator *(const T1& l, const T2& r) -> SmartReturn<T1, T2, Child>
    {
        return SmartReturn<T1, T2, Child>(
            [](auto&& l2, auto&& r2){ return l2 * r2;},
            l, r);
    }

    template <typename T1, typename T2>
    friend auto operator /(const T1& l, const T2& r) -> SmartReturn<T1, T2, Child>
    {
        return SmartReturn<T1, T2, Child>(
            [](auto&& l2, auto&& r2){ return l2 / r2;},
            l, r);
    }

    template <typename T1, typename T2>
    friend auto operator +(const T1& l, const T2& r) -> SmartReturn<T1, T2, Child>
    {
        return SmartReturn<T1, T2, Child>(
            [](auto&& l2, auto&& r2){ return l2 + r2;},
            l, r);
    }

    template <typename T1, typename T2>
    friend auto operator >=(const T1& l, const T2& r) -> SmartReturn<T1, T2, Child, LaneMask<ScalarCount>>
    {
        return LaneMask<ScalarCount>(
            [](auto&& l2, auto&& r2){ return l2 >= r2;},
            l, r);
    }

    template <typename T1, typename T2>
    friend auto operator >(const T1& l, const T2& r) -> SmartReturn<T1, T2, Child, LaneMask<ScalarCount>>
    {
        return LaneMask<ScalarCount>(
            [](auto&& l2, auto&& r2){ return l2 > r2;},
            l, r);
    }

    template <typename T1, typename T2>
    friend auto operator <=(const T1& l, const T2& r) -> SmartReturn<T1, T2, Child, LaneMask<ScalarCount>>
    {
        return LaneMask<ScalarCount>(
            [](auto&& l2, auto&& r2){ return l2 <= r2;},
            l, r);
    }

    template <typename T1, typename T2>
    friend auto operator <(const T1& l, const T2& r) -> SmartReturn<T1, T2, Child, LaneMask<ScalarCount>>
    {
        return LaneMask<ScalarCount>(
            [](auto&& l2, auto&& r2){ return l2 < r2;},
            l, r);
    }

    template <typename T1, typename T2>
    friend auto operator ==(const T1& l, const T2& r) -> SmartReturn<T1, T2, Child, LaneMask<ScalarCount>>
    {
        return LaneMask<ScalarCount>(
            [](auto&& l2, auto&& r2){ return l2 == r2;},
            l, r);
    }

    template <typename T1, typename T2>
    friend auto operator !=(const T1& l, const T2& r) -> SmartReturn<T1, T2, Child, LaneMask<ScalarCount>>
    {
        return LaneMask<ScalarCount>(
            [](auto&& l2, auto&& r2){ return l2 != r2;},
            l, r);
    }

    friend Child min(const Child& l, const Child& r)
    {
        return Child(
            [](auto&& l2, auto&& r2){ return min(l2, r2);},
            l, r);
    }
    friend Child max(const Child& l, const Child& r)
    {
        return Child(
            [](auto&& l2, auto&& r2){ return max(l2, r2);},
            l, r);
    }

    friend ScalarType<T> reduceMax(const Child& c)
    {
        auto init = std::numeric_limits<ScalarType<T>>::lowest();
        return Reduce([](auto&& l, auto&& r) { l = std::max(l, reduceMax(r)); }, init, c);
    }

    friend ScalarType<T> reduceMin(const Child& c)
    {
        auto init = std::numeric_limits<ScalarType<T>>::max();
        return Reduce([](auto&& l, auto&& r) { l = std::min(l, reduceMin(r)); }, init, c);
    }

    friend Child Blend(const LaneMask<ScalarCount>& b, const Child& c1, const Child& c2)
    {
        return Child(
            [](auto&& b2, auto&& l, auto&& r){ return Blend(b2, l, r); },
            b, c1, c2);
    }

    friend Child inc(const Child& in, const LaneMask<ScalarCount>& mask)
    {
        return Blend(mask, in + static_cast<ScalarType<T>>(1), in);
    }

    struct minOp
    {
        Child operator()(const Child& a, const Child& b)
        { return min(a, b); }
    };

    struct maxOp
    {
        Child operator()(const Child& a, const Child& b)
        { return max(a, b); }
    };
};

template <typename T, size_t ScalarCount, template <typename, size_t> class Child>
using ArithmeticBase = ArithmeticArray<vec_type_t<T>, vec_count<T, ScalarCount>::value, Child<T, ScalarCount>>;

template <size_t ScalarCount>
class LaneArray<float, ScalarCount> : public ArithmeticBase<float, ScalarCount, LaneArray>
{
    static_assert(ScalarCount == 64, "");
    using Base = ArithmeticBase<float, ScalarCount, LaneArray>;
public:
    using Base::Base;

    friend LaneMask<ScalarCount> isnan(const LaneArray& c)
    {
        return LaneMask<ScalarCount>(
            [](auto&& d){ return isnan(d); },
            c);
    }

    friend LaneMask<ScalarCount> isfinite(const LaneArray& c)
    {
        return LaneMask<ScalarCount>(
            [](auto&& d){ return isfinite(d); },
            c);
    }

    friend LaneArray erfc(const LaneArray& in)
    {
        return LaneArray(
            [](auto&& in2){ return erfc(in2); },
            in);
    }

    friend LaneArray log(const LaneArray<float, ScalarCount>& in)
    {
        return LaneArray(
            [](auto&& in2){ return log(in2); },
            in);
    }

    friend LaneArray log2(const LaneArray& in)
    {
        return LaneArray(
            [](auto&& in2){ return log2(in2); },
            in);
    }

    friend LaneArray exp(const LaneArray& in)
    {
        return LaneArray(
            [](auto&& in2){ return exp(in2); },
            in);
    }

    friend LaneArray exp2(const LaneArray& in)
    {
        return LaneArray(
            [](auto&& in2){ return exp2(in2); },
            in);
    }

    friend LaneArray sqrt(const LaneArray& in)
    {
        return LaneArray(
            [](auto&& in2){ return sqrt(in2); },
            in);
    }

    friend LaneArray<int, ScalarCount> floorCastInt(const LaneArray& in)
    {
        return LaneArray<int, ScalarCount>(
            [](auto&& in2){ return floorCastInt(in2); },
            in);
    }
};

template <size_t ScalarCount>
class LaneArray<int32_t, ScalarCount> : public ArithmeticBase<int32_t, ScalarCount, LaneArray>
{
    using Base = ArithmeticBase<int32_t, ScalarCount, LaneArray>;
public:
    using Base::Base;

    friend LaneArray operator|(const LaneArray& l, const LaneArray& r)
    {
        return LaneArray(
            [](auto&& l2, auto&& r2) { return l2 | r2; },
            l, r);
    }

};

template <size_t ScalarCount>
class LaneArray<int16_t, ScalarCount> : public ArithmeticBase<int16_t, ScalarCount, LaneArray>
{
    using Base = ArithmeticBase<int16_t, ScalarCount, LaneArray>;
public:
    using Base::Base;
};

template <size_t Len>
LaneArray<float, Len> AsFloat(const LaneArray<short, Len>& in)
{
    return LaneArray<float, Len>(
        [](auto&& in2) { return std::make_pair(LowFloats(in2), HighFloats(in2)); },
        in);
}

template <size_t Len>
LaneArray<int, Len> AsInt(const LaneArray<short, Len>& in)
{
    return LaneArray<int, Len>(
        [](auto&& in2) { return std::make_pair(LowInts(in2), HighInts(in2)); },
        in);
}

// TODO clean/kill?
template <size_t Len>
LaneArray<float, Len> AsFloat(const ArrayUnion<LaneArray<short, Len>>& in)
{
    return AsFloat(in.Simd());
}

template <size_t Len>
LaneArray<float, Len> AsFloat(const LaneArray<int, Len>& in)
{
    return LaneArray<float, Len>(
        [](auto&& in2) { return in2.AsFloat(); },
        in);
}

template <size_t Len>
LaneArray<short, Len> AsShort(const LaneArray<float, Len>& in)
{
    return LaneArray<short, Len>(
        [](auto&& in2) { return m512s(in2.first, in2.second); },
        in);
}

template <typename T, size_t Len, typename Child>
struct len_trait<BaseArray<T, Len, Child>>
{
    static constexpr size_t SimdCount = BaseArray<T, Len, Child>::SimdCount;
    static constexpr size_t ScalarCount = BaseArray<T, Len, Child>::ScalarCount;
    static constexpr size_t SimdWidth = BaseArray<T, Len, Child>::SimdWidth;
};
template <typename T, size_t Len>
struct len_trait<LaneArray<T, Len>>
{
    static constexpr size_t SimdCount = LaneArray<T,Len>::SimdCount;
    static constexpr size_t ScalarCount = LaneArray<T,Len>::ScalarCount;
    static constexpr size_t SimdWidth = LaneArray<T,Len>::SimdWidth;
};
template <size_t Len>
struct len_trait<LaneMask<Len>>
{
    static constexpr size_t SimdCount = LaneMask<Len>::SimdCount;
    static constexpr size_t ScalarCount = LaneMask<Len>::ScalarCount;
    static constexpr size_t SimdWidth = LaneMask<Len>::SimdWidth;
};

template <typename T>
struct len_trait<ArrayUnion<T>>
{
    static constexpr size_t SimdCount = len_trait<T>::SimdCount;
    static constexpr size_t ScalarCount = len_trait<T>::ScalarCount;
    static constexpr size_t SimdWidth = len_trait<T>::SimdWidth;
};

}}      // namespace PacBio::Mongo


// TODO: Seems like we might have a namespace wrinkle to iron out.
namespace PacBio {
namespace Simd {

template <typename T, size_t N>
struct SimdConvTraits<Mongo::LaneArray<T,N>>
{
    typedef Mongo::LaneMask<N> bool_conv;
    typedef Mongo::LaneArray<float,N> float_conv;
    typedef Mongo::LaneArray<int,N> index_conv;
    typedef Mongo::LaneArray<short,N> pixel_conv;
    typedef ArrayUnion<Mongo::LaneArray<T,N>> union_conv;
};

template <typename T, size_t N>
struct SimdTypeTraits<Mongo::LaneArray<T,N>>
{
    typedef T scalar_type;
    static const uint16_t width = N;
};

template <size_t N>
struct SimdTypeTraits<Mongo::LaneMask<N>>
{
    typedef bool scalar_type;
    static const uint16_t width = N;
};

}}   // namespace PacBio::Simd

#endif  // mongo_common_LaneArray_H_
