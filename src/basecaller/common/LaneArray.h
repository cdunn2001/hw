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

template <typename T>
struct PairAccess_t
{
    using SType = typename std::decay_t<T>::SimdType;
    static constexpr auto SimdCount = std::decay_t<T>::SimdCount/2;
    struct pair {
        SType& first;
        SType& second;
    };
    struct const_pair {
        const SType& first;
        const SType& second;
    };

    //pair operator[](size_t idx)
    //{
    //    auto* ptr = inner_.data();
    //    return pair{ ptr[2*idx], ptr[2*idx+1] };
    //}

    const_pair operator[](size_t idx) const
    {
        auto& ptr = inner_.data();
        return const_pair{ ptr[2*idx], ptr[2*idx+1] };
    }

    auto& data() { return *this; }
    const auto& data() const { return *this; }
    T& inner_;
};

template <typename T>
auto PairAccess(T&& t)
{
    return PairAccess_t<T>{t};
}


template <typename Ret, typename F, size_t...Ids, typename... Args>
Ret Helper2(F&& f, std::index_sequence<Ids...>, Args&&... args)
{
    auto f2 = [&](size_t id, auto&&... args)
    {
        return f(args.data()[id]...);
    };
    return Ret{f2(Ids, args...)...};
}

// TODO make the default constructor noop and a lot of the motivation for
// the compexity goes away.  We'll still need something to handle 32/16
// width simd types, but we don't have to try and construct in place
template <typename Ret, typename F, typename... Args>
Ret Helper3(F&& f, Args&&... args)
{
    //auto f2 = [&](size_t id, auto&&... args)
    //{
    //    return f(args.data()[id]...);
    //};
    //return Ret{f2(Ids, args...)...};
    Ret ret{};
    for (size_t i = 0; i < Ret::SimdCount/2; ++i)
    {
        auto val = f(args.data()[i]...);
        // TODO this assumes pair.  Need to make it automatic,
        // or at least make compiler errors friendly
        ret.data()[2*i] = val.first;
        ret.data()[2*i+1] = val.second;
    }
    return ret;
}

template <bool...bs>
struct bool_pack {};

template <bool...bs>
using all_true = std::is_same<bool_pack<true, bs...>, bool_pack<bs..., true>>;

// TODO this is a mess...!
template<bool identity> struct pass_through;
template <> struct pass_through<true>
{
    template <typename T>
    static decltype(auto) Do(T&& t) { return std::forward<T>(t); }
};
template <> struct pass_through<false>
{
    template <typename T>
    // TODO I think this leaves a danling reference if input is temporary
    static decltype(auto) Do(T&& t) { return PairAccess(std::forward<T>(t)); }
};

template <size_t len, typename T>
decltype(auto) through(T&&t)
{
    static constexpr bool b = (len == std::decay_t<T>::SimdCount);
    return pass_through<b>::Do(std::forward<T>(t));
}

enum wid_types
{
    all_same,
    input_16,
    input_32
};

template <typename Ret, typename...Args>
static constexpr wid_types wid_helper()
{
    if (all_true<(Ret::SimdCount == Args::SimdCount)...>::value) return all_same;
    if (Ret::ScalarCount / Ret::SimdCount == 16) return input_16;
    else return input_32;
}

// TODO naming sucks
template <wid_types>
struct wid_struct_t
{
    template <typename Ret, typename F, typename...Args>
    static Ret Helper(F&& f, Args&&... args)
    {
        static constexpr auto len = Ret::SimdCount;
        return Helper2<Ret>(std::forward<F>(f), std::make_index_sequence<len>{}, through<len>(std::forward<Args>(args))...);
    }
};

template <>
struct wid_struct_t<wid_types::input_16>
{
    template <typename Ret, typename F, typename...Args>
    static Ret Helper(F&& f, Args&&... args)
    {
        static constexpr auto len = Ret::SimdCount/2;
        return Helper3<Ret>(std::forward<F>(f), through<len>(std::forward<Args>(args))...);
    }
};

template <typename Ret, typename F, typename... Args>
Ret Helper(F&& f, Args&&... args)
{
    static constexpr auto t = wid_helper<Ret, std::decay_t<Args>...>();

    return wid_struct_t<t>::template Helper<Ret>(std::forward<F>(f), std::forward<Args>(args)...);
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
    static constexpr size_t ScalarCount = SimdCount * SimdTypeTraits<T>::width;
    explicit BaseArray(const std::array<T, SimdCount>& dat) : data_(dat) {}

    // TODO worry about alignment
    // TODO Make default ctor a noop?
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

    // Constructor that allows conversion (e.g. for promoting array of floats
    // to an array of m512f).
    template <typename U,
              // This prevents U = T, which is necessary if T is already scalar,
              // in which case we'd be duplicating the above constructor
              std::enable_if_t<!std::is_same<T, U>::value, int> = 0,
              // Make sure we can validly create our native scalar type from U
              std::enable_if_t<std::is_constructible<T, U>::value, int> = 0
    >
    explicit BaseArray(const std::array<U, SimdCount>& dat)
    {
        for (size_t i = 0; i < SimdCount; ++i)
        {
            data_[i] = T(dat[i]);
        }
    }

    // Construct off either 1 or N entries
    template <class... Ts,
            // Using SFINAE to make sure each type in Ts is (essentially) T.
            // The tuple is used to unpack all entries in the variadic pack, and
            // enable_if will prevent the function from being valid if the wrong
            // type is slipped in.
            typename dummy = std::tuple<
                std::enable_if_t<std::is_same<std::decay_t<Ts>, T>::value, int>...
            >
    >
    BaseArray(Ts&&... ts) : data_{{std::forward<Ts>(ts)...}}
    {
        static_assert(sizeof...(ts) == SimdCount || sizeof...(ts) == 1, "Incorrect number of arguments");
        if (sizeof...(ts) == 1) data_.fill(data()[0]);
    }

    // Similar to the last ctor, but now allowing type conversion (e.g. promoting from float)
    // TODO I think this can be unified with the above
    // TODO do we need decay?
    //template <class... Ts,
    //          typename dummy1 = std::tuple<
    //              std::enable_if_t<std::is_constructible<T, std::decay_t<Ts>>::value, int>...
    //          >,
    //          typename dummy2 = std::tuple<
    //              std::enable_if_t<!std::is_same<std::decay_t<Ts>, T>::value, int>...
    //          >
    //>
    //BaseArray(Ts&&... ts) : BaseArray(T(std::forward<Ts>(ts))...) {}
    template <typename U, std::enable_if_t<std::is_convertible<U, ScalarType<T>>::value, int> = 0>
    BaseArray(U&& val) : BaseArray(T(val)) {}

    // TODO check alignment somehow?
    BaseArray(PtrView<ScalarType<T>, ScalarCount> dat)
    {
        for (size_t i = 0; i < SimdCount; ++i)
        {
            data_[i] = T(dat[i*SimdTypeTraits<T>::width]);
        }
    }

    // TODO do we want this?
    BaseArray() : BaseArray(T(static_cast<ScalarType<T>>(0))) {}

    // TODO should these be protected only?
    // allows access to underlying array, assuming calling code is explicitly
    // aware of how long the array is.
    operator const std::array<T,SimdCount>&() const
    {
        return data_;
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
    static constexpr auto ScalarCount = Base::ScalarCount;
    using SimdType = m512b;

public:
    friend bool all(const LaneMask& m)
    {
        bool ret = true;
        for (const auto& d : m.data()) ret &= all(d);
        return ret;
    }

    friend bool any(const LaneMask& m)
    {
        bool ret = false;
        for (const auto& d : m.data()) ret |= any(d);
        return ret;
    }

    friend bool none(const LaneMask& m)
    {
        bool ret = true;
        for (const auto& d : m.data()) ret &= none(d);
        return ret;
    }

    friend LaneMask operator| (const LaneMask& l, const LaneMask& r)
    {
        return Helper<LaneMask>(
            [](auto&& l, auto&& r){ return l | r; },
            l, r);
    }

    friend LaneMask operator! (const LaneMask& m)
    {
        return Helper<LaneMask>(
            [](auto&& m){ return !m; },
            m);
    }

    friend LaneMask operator& (const LaneMask& l, const LaneMask& r)
    {
        return Helper<LaneMask>(
            [](auto&& l, auto&& r){ return l & r; },
            l, r);
    }

    LaneMask& operator &= (const LaneMask& o)
    {
        for (size_t i = 0; i < SimdCount; ++i) this->data()[i] &= o.data()[i];
        return *this;
    }

    LaneMask& operator |= (const LaneMask& o)
    {
        for (size_t i = 0; i < SimdCount; ++i) this->data()[i] |= o.data()[i];
        return *this;
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
    static constexpr auto SimdCount = Base::SimdCount;
    static constexpr auto ScalarCount = Base::ScalarCount;

    Child& operator+=(const Child& other)
    {
        for (size_t i = 0; i < SimdCount; ++i)
        {
            data_[i] += other.data()[i];
        }
        return static_cast<Child&>(*this);
    }
    Child& operator-=(const Child& other)
    {
        for (size_t i = 0; i < SimdCount; ++i)
        {
            data_[i] -= other.data()[i];
        }
        return static_cast<Child&>(*this);
    }
    Child& operator/=(const Child& other)
    {
        for (size_t i = 0; i < SimdCount; ++i)
        {
            data_[i] /= other.data()[i];
        }
        return static_cast<Child&>(*this);
    }
    Child& operator*=(const Child& other)
    {
        for (size_t i = 0; i < SimdCount; ++i)
        {
            data_[i] *= other.data()[i];
        }
        return static_cast<Child&>(*this);
    }

    // TODO should only exist for floats...
    friend LaneMask<ScalarCount> isnan(const Child& c)
    {
        return Helper<LaneMask<ScalarCount>>(
            [](auto&& d){ return isnan(d); },
            c);
    }
    // TODO should only exist for floats...
    friend LaneMask<ScalarCount> isfinite(const Child& c)
    {
        return Helper<LaneMask<ScalarCount>>(
            [](auto&& d){ return isfinite(d); },
            c);
    }

    friend Child operator -(const Child& c)
    {
        return Helper<Child>(
            [](auto&& d){ return -d;},
            c);
    }

    friend Child operator -(const Child& l, const Child& r)
    {
        return Helper<Child>(
            [](auto&& l, auto&& r){ return l - r;},
            l, r);
    }

    friend Child operator *(const Child& l, const Child& r)
    {
        return Helper<Child>(
            [](auto&& l, auto&& r){ return l * r;},
            l, r);
    }

    friend Child operator /(const Child& l, const Child& r)
    {
        return Helper<Child>(
            [](auto&& l, auto&& r){ return l / r;},
            l, r);
    }

    friend Child operator +(const Child& l, const Child& r)
    {
        return Helper<Child>(
            [](auto&& l, auto&& r){ return l + r;},
            l, r);
    }

    friend LaneMask<ScalarCount> operator >=(const Child& l, const Child& r)
    {
        return Helper<LaneMask<ScalarCount>>(
            [](auto&& l, auto&& r){ return l >= r;},
            l, r);
    }

    friend LaneMask<ScalarCount> operator >(const Child& l, const Child& r)
    {
        return Helper<LaneMask<ScalarCount>>(
            [](auto&& l, auto&& r){ return l > r;},
            l, r);
    }

    friend LaneMask<ScalarCount> operator <=(const Child& l, const Child& r)
    {
        return Helper<LaneMask<ScalarCount>>(
            [](auto&& l, auto&& r){ return l <= r;},
            l, r);
    }

    friend LaneMask<ScalarCount> operator <(const Child& l, const Child& r)
    {
        return Helper<LaneMask<ScalarCount>>(
            [](auto&& l, auto&& r){ return l < r;},
            l, r);
    }

    friend LaneMask<ScalarCount> operator ==(const Child& l, const Child& r)
    {
        return Helper<LaneMask<ScalarCount>>(
            [](auto&& l, auto&& r){ return l == r;},
            l, r);
    }

    friend LaneMask<ScalarCount> operator !=(const Child& l, const Child& r)
    {
        return Helper<LaneMask<ScalarCount>>(
            [](auto&& l, auto&& r){ return l != r;},
            l, r);
    }

    friend Child min(const Child& l, const Child& r)
    {
        Child ret;
        for (size_t i = 0; i < SimdCount; ++i)
        {
            using std::min;
            ret.data()[i] = min(l.data()[i], r.data()[i]);
        }
        return ret;
    }
    friend Child max(const Child& l, const Child& r)
    {
        Child ret;
        for (size_t i = 0; i < SimdCount; ++i)
        {
            using std::max;
            ret.data()[i] = max(l.data()[i], r.data()[i]);
        }
        return ret;
    }

    friend ScalarType<T> reduceMax(const Child& c)
    {
        auto ret = std::numeric_limits<ScalarType<T>>::lowest();
        for (const auto& d : c.data()) ret = std::max(ret, reduceMax(d));
        return ret;
    }

    friend ScalarType<T> reduceMin(const Child& c)
    {
        auto ret = std::numeric_limits<ScalarType<T>>::max();
        for (const auto& d : c.data()) ret = std::min(ret, reduceMin(d));
        return ret;
    }

    friend Child Blend(const LaneMask<ScalarCount>& b, const Child& c1, const Child& c2)
    {
        return Helper<Child>(
            [](auto&& b, auto&& l, auto&& r){ return Blend(b, l, r); },
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

template <typename T, size_t ScalarCount_ = laneSize>
class LaneArray : public ArithmeticArray<vec_type_t<T>, ScalarCount_/16, LaneArray<T, ScalarCount_>>
{
    static_assert(ScalarCount_ % 16 == 0, "");
    using Base = ArithmeticArray<vec_type_t<T>, ScalarCount_/16, LaneArray<T, ScalarCount_>>;
public:
    using Base::Base;
    static constexpr auto SimdCount = Base::SimdCount;
    static constexpr auto ScalarCount = Base::ScalarCount;
    using SimdType = T;
};

template <size_t ScalarCount_>
class LaneArray<short, ScalarCount_> : public ArithmeticArray<m512s, ScalarCount_/32, LaneArray<short, ScalarCount_>>
{
    static_assert(ScalarCount_ % 32 == 0, "");
    using Base = ArithmeticArray<m512s, ScalarCount_/32, LaneArray<short, ScalarCount_>>;
    using T = short;
public:
    using Base::Base;
    static constexpr auto SimdCount = Base::SimdCount;
    static constexpr auto ScalarCount = Base::ScalarCount;
};

// tODO ??? keep?
template <size_t Len>
LaneArray<float, Len> AsFloat(const LaneArray<short, Len>& in)
{
    LaneArray<float, Len> out;
    // TODO again, clean up Len/Count naming...!!
    for (size_t i = 0; i < LaneArray<short, Len>::SimdCount; ++i)
    {
        out.data()[2*i] = LowFloats(in.data()[i]);
        out.data()[2*i+1] = HighFloats(in.data()[i]);
    }
    return out;
}

template <size_t Len>
LaneArray<int, Len> AsInt(const LaneArray<short, Len>& in)
{
    LaneArray<int, Len> out;
    // TODO again, clean up Len/Count naming...!!
    for (size_t i = 0; i < LaneArray<short, Len>::SimdCount; ++i)
    {
        out.data()[2*i] = LowInts(in.data()[i]);
        out.data()[2*i+1] = HighInts(in.data()[i]);
    }
    return out;
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
    LaneArray<float, Len> out;
    // TODO again, clean up Len/Count naming...!!
    for (size_t i = 0; i < LaneArray<int, Len>::SimdCount; ++i)
    {
        out.data()[i] = in.data()[i].AsFloat();
    }
    return out;
}

template <size_t Len>
LaneArray<short, Len> AsShort(const LaneArray<float, Len>& in)
{
    LaneArray<short, Len> out;
    // TODO again, clean up Len/Count naming...!!
    for (size_t i = 0; i < LaneArray<short, Len>::SimdCount; ++i)
    {
        out.data()[i] = m512s(in.data()[2*i], in.data()[2*i+1]);
    }
    return out;
}

template <size_t ScalarCount>
LaneArray<float, ScalarCount> erfc(const LaneArray<float, ScalarCount>& in)
{
    return Helper<LaneArray<float, ScalarCount>>(
        [](auto&& in){ return erfc(in); },
        in);
}

template <size_t ScalarCount>
LaneArray<float, ScalarCount> log(const LaneArray<float, ScalarCount>& in)
{
    return Helper<LaneArray<float, ScalarCount>>(
        [](auto&& in){ return log(in); },
        in);
}

template <size_t ScalarCount>
LaneArray<float, ScalarCount> log2(const LaneArray<float, ScalarCount>& in)
{
    return Helper<LaneArray<float, ScalarCount>>(
        [](auto&& in){ return log2(in); },
        in);
}

template <size_t ScalarCount>
LaneArray<float, ScalarCount> exp(const LaneArray<float, ScalarCount>& in)
{
    return Helper<LaneArray<float, ScalarCount>>(
        [](auto&& in){ return exp(in); },
        in);
}

template <size_t ScalarCount>
LaneArray<float, ScalarCount> exp2(const LaneArray<float, ScalarCount>& in)
{
    return Helper<LaneArray<float, ScalarCount>>(
        [](auto&& in){ return exp2(in); },
        in);
}

template <size_t ScalarCount>
LaneArray<float, ScalarCount> sqrt(const LaneArray<float, ScalarCount>& in)
{
    return Helper<LaneArray<float, ScalarCount>>(
        [](auto&& in){ return sqrt(in); },
        in);
}

template <size_t ScalarCount>
LaneArray<int, ScalarCount> floorCastInt(const LaneArray<float, ScalarCount>& in)
{
    return Helper<LaneArray<int, ScalarCount>>(
        [](auto&& in){ return floorCastInt(in); },
        in);
}

template <size_t ScalarCount>
LaneArray<short, ScalarCount> inc(const LaneArray<short, ScalarCount>& in, const LaneMask<ScalarCount>& mask)
{
    // TODO add scalar arithmetic.  This promotes to a full vector
    return Blend(mask, in + (short)1, in);
}

// TODO fix attrocious naming
inline m512s Blend(PairAccess_t<const LaneMask<64>&>::const_pair b, const m512s& l, const m512s& r)
{
    // TODO fix this redirection?
    return Blend(b.first, b.second, l, r);
}

template <size_t ScalarCount>
LaneArray<int, ScalarCount> operator|(const LaneArray<int, ScalarCount>& l, const LaneArray<int, ScalarCount>& r)
{
    return Helper<LaneArray<int, ScalarCount>>(
        [](auto&& l, auto&& r) { return l | r; },
        l, r);
}

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
