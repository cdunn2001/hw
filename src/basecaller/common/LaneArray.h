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


// TODO move?
// TODO fix attrocious naming
inline m512s Blend(const std::pair<const m512b&, const m512b&>& b, const m512s& l, const m512s& r)
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
//{
//    static constexpr size_t SimdCount = 1;
//    static constexpr size_t ScalarCount = 1;
//    static constexpr size_t SimdWidth = 1;
//};

template <typename Ret, typename...Args>
static constexpr size_t MinSimdCount()
{
    constexpr auto ret = std::min({len_trait<Ret>::SimdCount, len_trait<Args>::SimdCount...});
    static_assert(ret == 2 || ret == 4, "");
    return ret;
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
        typename dummy1 = std::enable_if_t<std::is_constructible<T, U>::value>,
        typename dummy2 = std::enable_if_t<!std::is_same<T, U>::value>>
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

    // TODO check alignment somehow?
    BaseArray(PtrView<ScalarType<T>, ScalarCount> dat)
    {
        for (size_t i = 0; i < SimdCount; ++i)
        {
            data_[i] = T(dat[i*SimdTypeTraits<T>::width]);
        }
    }

    // Intentionally not initializing our data
    // Would just kill this off, but it's unfortunately
    // required by Eigen at least.
    BaseArray() = default;

    template <size_t N, typename U, std::enable_if_t<N == len_trait<std::decay_t<U>>::SimdCount, int> = 0>
    static decltype(auto) Access(U&& val, size_t idx)
    {
        return (val.data()[idx]);
    }

    template <size_t N, typename U, std::enable_if_t<2*N == len_trait<std::decay_t<U>>::SimdCount, int> = 0>
    static auto Access(U&& val, size_t idx)
    {
        using SimdType = typename std::decay_t<U>::SimdType;
        using Ref_t = std::conditional_t<std::is_const<std::remove_reference_t<U>>::value, const SimdType&, SimdType&>;
        return std::pair<Ref_t, Ref_t>{val.data()[idx*2], val.data()[idx*2+1]};
    }

    template <size_t N, typename U, std::enable_if_t<1 == len_trait<std::decay_t<U>>::SimdCount, int> = 0>
    static decltype(auto) Access(U&& val, size_t idx)
    {
        val.fuckit();
        return (val);
    }

    template <typename F, typename...Args, std::enable_if_t<sizeof...(Args) != 0, int> = 0 >
    BaseArray(F&& f, const Args&... args)
    {
        static constexpr auto loopMax = MinSimdCount<Child, Args...>();
        for (size_t i = 0; i < loopMax; ++i)
        {
            Access<loopMax>(*this, i) = f(Access<loopMax>(args, i)...);
        }
    }

    template <typename F, typename...Args>
    Child& Update(F&& f, const Args&... args)
    {
        static constexpr auto loopMax = MinSimdCount<Child, Args...>();
        for (size_t i = 0; i < loopMax; ++i)
        {
            f(Access<loopMax>(*this, i), Access<loopMax>(args, i)...);
        }
        return static_cast<Child&>(*this);
    }

    template <typename F, typename Ret, typename...Args>
    static Ret Reduce(F&& f, Ret initial, const Args&... args)
    {
        Ret ret = initial;
        static constexpr auto loopMax = MinSimdCount<Args...>();
        for (size_t i = 0; i < loopMax; ++i)
        {
            f(ret, Access<loopMax>(args, i)...);
        }
        return ret;
    }

    // TODO clean and check assembly
    // TODO remove implicit?
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

    Child& operator+=(const Child& other)
    {
        return Update([](auto&& l, auto&& r) { l += r; }, other);
    }
    Child& operator-=(const Child& other)
    {
        return Update([](auto&& l, auto&& r) { l -= r; }, other);
    }
    Child& operator/=(const Child& other)
    {
        return Update([](auto&& l, auto&& r) { l /= r; }, other);
    }
    Child& operator*=(const Child& other)
    {
        return Update([](auto&& l, auto&& r) { l *= r; }, other);
    }

    friend Child operator -(const Child& c)
    {
        return Child(
            [](auto&& d){ return -d;},
            c);
    }

    friend Child operator -(const Child& l, const Child& r)
    {
        return Child(
            [](auto&& l2, auto&& r2){ return l2 - r2;},
            l, r);
    }

    friend Child operator *(const Child& l, const Child& r)
    {
        return Child(
            [](auto&& l2, auto&& r2){ return l2 * r2;},
            l, r);
    }

    friend Child operator /(const Child& l, const Child& r)
    {
        return Child(
            [](auto&& l2, auto&& r2){ return l2 / r2;},
            l, r);
    }

    friend Child operator +(const Child& l, const Child& r)
    {
        return Child(
            [](auto&& l2, auto&& r2){ return l2 + r2;},
            l, r);
    }

    friend LaneMask<ScalarCount> operator >=(const Child& l, const Child& r)
    {
        return LaneMask<ScalarCount>(
            [](auto&& l2, auto&& r2){ return l2 >= r2;},
            l, r);
    }

    friend LaneMask<ScalarCount> operator >(const Child& l, const Child& r)
    {
        return LaneMask<ScalarCount>(
            [](auto&& l2, auto&& r2){ return l2 > r2;},
            l, r);
    }

    friend LaneMask<ScalarCount> operator <=(const Child& l, const Child& r)
    {
        return LaneMask<ScalarCount>(
            [](auto&& l2, auto&& r2){ return l2 <= r2;},
            l, r);
    }

    friend LaneMask<ScalarCount> operator <(const Child& l, const Child& r)
    {
        return LaneMask<ScalarCount>(
            [](auto&& l2, auto&& r2){ return l2 < r2;},
            l, r);
    }

    friend LaneMask<ScalarCount> operator ==(const Child& l, const Child& r)
    {
        return LaneMask<ScalarCount>(
            [](auto&& l2, auto&& r2){ return l2 == r2;},
            l, r);
    }

    friend LaneMask<ScalarCount> operator !=(const Child& l, const Child& r)
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

template <typename T>
struct vec_type;

template<> struct vec_type<float> { using type = m512f; };
template<> struct vec_type<int>   { using type = m512i; };
template<> struct vec_type<bool>  { using type = m512b; };
template<> struct vec_type<short> { using type = m512s; };

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

template <typename T, size_t ScalarCount, template <typename, size_t> class Child>
using ArithmeticBase = ArithmeticArray<vec_type_t<T>, vec_count<T, ScalarCount>::value, Child<T, ScalarCount>>;

template <typename T, size_t ScalarCount = laneSize>
class LaneArray;

template <size_t ScalarCount>
class LaneArray<float, ScalarCount> : public ArithmeticBase<float, ScalarCount, LaneArray>
{
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

    friend LaneArray log(const LaneArray& in)
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

    friend LaneArray inc(const LaneArray& in, const LaneMask<ScalarCount>& mask)
    {
        // TODO add scalar arithmetic.  This promotes to a full vector
        return Blend(mask, in + (short)1, in);
    }
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

}}   // namespace PacBio::Simd

#endif  // mongo_common_LaneArray_H_
