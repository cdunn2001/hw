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
//
// This defines the CRTP base class that all the LaneArray stuff is based on.
// It is responsible for managing the creation/storage/transformation of
// the data.  In particular it provides:
// - Easy conversions to/from CudaArray
// - Easy conversion to/from other types (e.g. LaneArray<int> -> LaneArray<float>)
// - Loading of data from a memory address
// - Abstracts away most complications arising from mixed width types,
//   e.g. short types that prefer to operate on 32 elements at a time and
//   float types that prefer to operate on 16.
// - Some generic transformation infrastructure that applies an arbitrary
//   functor to any number of LaneArray/scalar arguments.  This will be
//   the building block that allows children to simply define their
//   operator overloads without worrying about type width and/or
//   scalar/vector differences.
//
// - As an aside, the transformation functionality is general enough we're
//   probably not too far away from being able to support "Expression Templates".
//   I didn't go that far because I don't know we have a concrete need, but
//   LaneArrays are large enough that for complex computations it might indeed
//   be worth it to do the entire computation one simd element at a time.

#ifndef mongo_common_simd_BaseArray_H_
#define mongo_common_simd_BaseArray_H_

#include <common/LaneArray_fwd.h>
#include <common/simd/LaneArrayTraits.h>

#include <common/cuda/utility/CudaArray.h>

namespace PacBio {
namespace Simd {

// Helper class to smooth over situations where we're iterating over both
// 32 bit and 16 bit types.  It just sets up a pair of references so
// that you can more naturally associate something like a single
// m512s with a pair of m512i
//
// T can either be const or not, depending on the type of
// reference you want.
template <typename T>
struct PairRef
{
    PairRef(T& t1, T& t2) : first(t1), second(t2) {}
    PairRef(const std::pair<T, T>& p) : first(p.first), second(p.second) {}

    // Allow assignments from other PairRefs or even std::pair, as
    // long as our destination is not itself const.
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

// Describes a range of memory, which is delineated
// by a pointer and a compile time length.  This is used
// to force the calling code to write out the array length
// in the type name should prevent/limit memory overrun
// errors because of length mismatches
template <typename T, size_t Len>
class MemoryRange
{
public:
    explicit MemoryRange(const T* data)
        : data_{data}
    {}

    const T* get() const { return data_; }
private:
    const T* data_;
};

// CRTP base class for all LaneArray objects.  It's responsible for defining all
// storage/construction/conversion logic.
template <typename T, size_t SimdCount_, typename Child>
class alignas(T) BaseArray
{
    // Helper function to find the argument that requires the max
    // number of m512 varibles to store.  e.g. LaneArray<int> needs
    // 4 while LaneArray<short> needs 2.
    // Note: Scalars do come through as arguments, but their `SimdCount`
    //       comes through as 1 so they effectively don't participate
    template <typename...Args>
    static constexpr size_t MaxSimdCount()
    {
        static_assert(sizeof...(Args) > 0,"");
        return std::max({(size_t)len_trait<Args>::SimdCount...});
    }

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
    static constexpr size_t SimdWidth = T::size();
    static constexpr size_t ScalarCount = SimdCount * SimdWidth;
    using SimdType = T;

public: // ctors

    // Intentionally not initializing our data
    // Would just kill this off, but it's unfortunately
    // required by Eigen at least.
    BaseArray() = default;

    // Populate whole array with uniform value.
    template <typename U, std::enable_if_t<std::is_convertible<U, ScalarType<T>>::value, int> = 0>
    BaseArray(U&& val)
    {
        for (auto& d : data_)
        {
            // first convert the input to the expected type
            // then the implicit ctors of the m512 types
            // can take it the rest of the way
            d = static_cast<ScalarType<T>>(val);
        }
    }

    // Explicit conversion from CudaArray.  We will use intrinsics to
    // load the data, so the CudaArray is required to be appropriately
    // aligned.
    // Note: Can't use this ctor for bools, as they are not bitwise equivalant,
    //       neither for SSE nor for AVX512
    template <typename U = T,
              bool isBool = std::is_same<ScalarType<U>, bool>::value,
              std::enable_if_t<!isBool, int> = 0>
    explicit BaseArray(const Cuda::Utility::CudaArray<ScalarType<T>, ScalarCount>& data)
    {
        constexpr auto width = SimdTypeTraits<T>::width;
        auto* dat = data.data();
        assert(reinterpret_cast<size_t>(dat) % alignof(T) == 0);

        for (auto& d : data_)
        {
            d = T(dat);
            dat += width;
        }
    }

    // Explicit conversion from CudaArray, this time also converting from
    // one type to another.
    // Note: We still can't use this ctor for bools, as ArrayUnion is
    //       incompatible (still requires bitwise compatible layout)
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

    // Construct by loading data from an arbitrary memory location.
    // Again it's the calling code's responsibility to ensure
    // appropriate alignment.
    template <typename U = ScalarType<T>,
        std::enable_if_t<!std::is_same<bool, U>::value, int> = 0
        >
    BaseArray(MemoryRange<ScalarType<T>, ScalarCount> dat)
    {
        assert(reinterpret_cast<size_t>(dat.get()) % alignof(T) == 0);
        for (size_t i = 0; i < SimdCount; ++i)
        {
            data_[i] = T(dat.get() + i*SimdTypeTraits<T>::width);
        }
    }

    // Now copy/conversions from other BaseArrays
    BaseArray(const BaseArray&) = default;

    template <typename U, typename UChild>
    BaseArray(const BaseArray<U, SimdCount, UChild>& o)
    {
        for (size_t i = 0; i < SimdCount; ++i)
        {
            data_[i] = T(o.data()[i]);
        }
    }

    // Handles converting to narrower type,
    // e.g. m512i to m512s
    template <typename U, typename UChild>
    BaseArray(const BaseArray<U, 2*SimdCount, UChild>& o)
    {
        for (size_t i = 0; i < SimdCount; ++i)
        {
            data_[i] = T(o.data()[2*i], o.data()[2*i+1]);
        }
    }

    // Handles converting to wider type,
    // e.g. m512s to m512i
    template <typename U, typename UChild>
    BaseArray(const BaseArray<U, SimdCount/2, UChild>& o)
    {
        static_assert(SimdCount % 2 == 0, "");
        for (size_t i = 0; i < SimdCount; i+=2)
        {
            auto tmp = std::pair<T,T>(o.data()[i/2]);
            data_[i] = tmp.first;
            data_[i+1] = tmp.second;
        }
    }

    // Constructs this base array as an arbitrary function of the supplied arguments.
    // This function will automatically walk the supplied input arrays, calling
    // F for each value.  This function does the necessary legwork to smooth over
    // 32/16 bit mismatches as well as scalar/vector mismatches (via the Access function
    // overload set)
    template <typename F, typename...Args, std::enable_if_t<sizeof...(Args) != 0, int> = 0 >
    BaseArray(F&& f, const Args&... args)
    {
        static_assert(sizeof...(Args) > 0, "");
        // WorkingType is mostly used to promote scalar values to an
        // appropriate m512 type, which will give us implicit access
        // to the sensible set of operator overloads
        using WorkingType = CommonArrayType_t<Args...>;

        // Determine how many scalar elements we traverse at once.  If every datatype is
        // the same width we'll just use the native m512 width.  If things are mixed then
        // either someone will be traversed in pairs, or m512s values will be traversed
        // in their low/high values separately.
        static constexpr auto ScalarStride = std::max(SimdTypeTraits<vec_type_t<WorkingType>>::width,
                                                      SimdTypeTraits<SimdType>::width);
        static constexpr auto loopMax = ScalarCount / ScalarStride;
        for (size_t i = 0; i < loopMax; ++i)
        {
            Access<loopMax, void>(static_cast<Child&>(*this), i)
                = f(Access<loopMax, WorkingType>(args, i)...);
        }
    }

 public: // transformation operations

    // Very similar to the last ctor, but we'll update ourself instead
    // of construction.  Useful for things like compound assignment
    // operators
    template <typename F, typename...Args>
    Child& Update(F&& f, const Args&... args)
    {
        static constexpr auto loopMax = MaxSimdCount<Child, Args...>();
        using WorkingType = ScalarType<Child>;
        for (size_t i = 0; i < loopMax; ++i)
        {
            f(Access<loopMax, WorkingType>(static_cast<Child&>(*this), i),
              Access<loopMax, WorkingType>(args, i)...);
        }
        return static_cast<Child&>(*this);
    }

    // One more function for reduction operations (e.g. any/all/reduceMax)
    template <typename F, typename Ret, typename...Args>
    static Ret Reduce(F&& f, Ret initial, const Args&... args)
    {
        static constexpr auto loopMax = MaxSimdCount<Child, Args...>();
        using WorkingType = ScalarType<Child>;

        Ret ret = initial;
        for (size_t i = 0; i < loopMax; ++i)
        {
            f(ret, Access<loopMax, WorkingType>(args, i)...);
        }
        return ret;
    }

 public: // conversions

    // Convert ourselves to a CudaArray.  We're just going to do
    // a memcpy, so no bools allowed
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

    // Convert to CudaArray of a different type.  Again no bools,
    // because we'll be using ArrayUnion which is incompatible
    template <typename U,
              bool different = !std::is_same<U, ScalarType<T>>::value,
              bool canConvert = std::is_convertible<ScalarType<T>, U>::value,
              std::enable_if_t<different && canConvert, int> = 0>
    operator Cuda::Utility::CudaArray<U, ScalarCount>() const
    {
        Cuda::Utility::CudaArray<U, ScalarCount> ret;
        auto tmp = MakeUnion(static_cast<const Child&>(*this));
        for (size_t i = 0; i < ScalarCount; ++i)
        {
            ret[i] = static_cast<U>(tmp[i]);
        }
        return ret;
    }

    Cuda::Utility::CudaArray<ScalarType<T>, ScalarCount> ToArray() const
    {
        return static_cast<const Child&>(*this);
    }

public: // raw simd data access.
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

private: // Access helpers

    // The greatest difficulty in all this LaneArray stuff was abstracting
    // away the differences between 16bit and 32bit types.  I almost wrote
    // some m256s classes so that we always stride by 16 elements, but
    // ultimately did not because mongo does a lot of 16 bit computation and
    // we don't want to cut the throughput.
    //
    // So the only real answer I came up with was to traverse LaneArray objects via
    // a gnarly set of overloads as defined here.  They should automatically
    // access elements by pairs or halves as necessary, but it's admittedly
    // complicated.  It took me a few attempts to get it right...

    // Easiest version, where our native simd width matches our
    // iteration stride.  Just return a reference to the appropriate index
    template <size_t Count, typename WorkingType, typename U,
              std::enable_if_t<Count == len_trait<std::decay_t<U>>::SimdCount, int> = 0>
    static decltype(auto) Access(U&& val, size_t idx)
    {
        return (val.data()[idx]);
    }

    // We're striding over half our native width.  Return either the low
    // or high half as appropriate.
    // Note: we're returning a value, so this only works on the rhs of
    //       an assignment.
    template <size_t Count, typename WorkingType, typename U,
        std::enable_if_t<Count == 2*len_trait<std::decay_t<U>>::SimdCount && (Count > 2), int> = 0>
    static auto Access(U&& val, size_t idx)
    {
        if (idx%2 == 0) return Low<WorkingType>(val.data()[idx/2]);
        else return High<WorkingType>(val.data()[idx/2]);
    }

    // Now the opposite, we need to access two elements at
    // once.  Return a PairRef pointing to the appropriate locations
    template <size_t N, typename WorkingType, typename U,
              std::enable_if_t<2*N == len_trait<std::decay_t<U>>::SimdCount, int> = 0>
    static auto Access(U&& val, size_t idx)
    {
        using SimdType = typename std::decay_t<U>::SimdType;
        using Ref_t = std::conditional_t<std::is_const<std::remove_reference_t<U>>::value, const SimdType, SimdType>;
        return PairRef<Ref_t>{val.data()[idx*2], val.data()[idx*2+1]};
    }

    // This overload wins if we have a scalar value.  Promote it to
    // the common type simd variable.  This is necessary so that
    // something like LaneArray<int> * float sees the right overload
    // set and forces the m512i to convert to an m512f
    template <size_t N, typename WorkingType, typename  U,
              std::enable_if_t<1 == len_trait<std::decay_t<U>>::SimdCount, int> = 0>
    static auto Access(U&& val, size_t)
    {
        return static_cast<vec_type_t<WorkingType>>(val);
    }

    // One last helper in case we're dealing with an ArrayUnion.
    // Just unwrap it and call one of the Access overloads above.
    template <size_t N, typename WorkingType, typename U>
    static decltype(auto) Access(const ArrayUnion<U>& t, size_t idx)
    {
        return Access<N, WorkingType>(t.Simd(), idx);
    }

};

}}

#endif //mongo_common_simd_BaseArray_H_
