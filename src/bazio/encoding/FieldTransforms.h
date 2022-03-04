// Copyright (c) 2021, Pacific Biosciences of California, Inc.
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
/// \file FieldTransforms.h
/// \details Defines the various ways by which an input value can be
///          transformed to an integral type in preparation for
///          serialization.  The result of the transformation will be
///          a uint64_t value, though the storeSigned value will indicate
///          if the bits should be interpreted as a signed value. The
///          currently supported transformations are:
///          * NoOp:             Do nothing, just pass through the value
///          * FixedPoint:       Apply a scale paramter to a floating point value
///                              and then round do the nearest integral value
///          * LossySequelCodec: Applies a lossy and compression, which starts out
///                              counting by 1s, but as we get further from 0 our
///                              stride begins increasing by powers of 2, so that
///                              a wider input range can be mapped to a fewer number
///                              of bits.
///          * DeltaCompression: A stateful transformation that returns the difference
///                              between the current and previously supplied values.
///
///          Beyond that, there is a Transform class, suitable for
///          use in template metaprogramming.  Where the transformations
///          themselves use runtime values to control their behaviour,
///          the Transform class uses only compile time values, and
///          can be used to promote extensive inlining/optimization by
///          the compiler for chosen configurations.
///
///          There is also the MultiTransform class which can be used to chain together
///          multiple Transform types together.  Each constituent transformation is
///          still required to result in a uint64_t.
///
///          Stateful transformations are supported, though it's recommended to avoid
///          that when possible.  If there is a data-dependant value that needs to be
///          recorded/updated then that is fine, but if there is a field parameter it
///          may be better to accept it as an argument to Apply/Revert, so that the
///          compiler can see things better and perhaps optimize more aggressively.


#ifndef PACBIO_BAZIO_ENCODING_FIELD_TRANSFORMS_H
#define PACBIO_BAZIO_ENCODING_FIELD_TRANSFORMS_H

#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <common/cuda/utility/CudaTuple.h>

#include <bazio/encoding/BazioEnableCuda.h>
#include <bazio/encoding/EncodingParams.h>

#include <bazio/encoding/Types.h>

namespace PacBio {
namespace BazIO {

template <typename Base, typename... autoParams>
struct Transform
{
    template <typename T>
    BAZ_CUDA uint64_t Apply(const T& t, StoreSigned storeSigned)
    {
        return b_.Apply(t, storeSigned, autoParams::val...);
    }
    template <typename Ret>
    BAZ_CUDA Ret Revert(uint64_t val, StoreSigned storeSigned)
    {
        return b_.template Revert<Ret>(val, storeSigned, autoParams::val...);
    }
    static std::vector<TransformsParams> Params()
    {
        return std::vector<TransformsParams>{Base::Params(autoParams::val...)};
    }
private:
    Base b_;
};

template <typename Trans1, typename... TransRest>
struct MultiTransform
{
    template <typename T>
    BAZ_CUDA uint64_t Apply(const T& t, StoreSigned storeSigned)
    {
        return rest_.Apply(first_.Apply(t, storeSigned), storeSigned);
    }
    template <typename Ret>
    BAZ_CUDA Ret Revert(uint64_t val, StoreSigned storeSigned)
    {
        return first_.template Revert<Ret>(rest_.template Revert<uint64_t>(val, storeSigned), storeSigned);
    }
    static std::vector<TransformsParams> Params()
    {
        return std::vector<TransformsParams>{ Trans1::Params(), TransRest::Params()... };
    }
private:
    Trans1 first_;
    MultiTransform<TransRest...> rest_;
};

template <typename Trans>
struct MultiTransform<Trans> : public Trans {};

struct NoOp
{
    template <typename T>
    BAZ_CUDA static uint64_t Apply(const T& t, StoreSigned)
    {
        static_assert(std::is_integral<T>::value, "");
        return static_cast<uint64_t>(t);
    }
    template <typename Ret>
    BAZ_CUDA static Ret Revert(uint64_t val, StoreSigned)
    {
        static_assert(std::is_integral<Ret>::value, "");
        return static_cast<Ret>(val);
    }

    static TransformsParams Params()
    {
        TransformsParams tp;
        NoOpTransformParams params;
        tp.params = params;
        return tp;
    }
};

// A simple fixed point representation for floating point input.  It is an error to use this
// function with values that are out of range, due to:
// * Handing in a literal inf/nan input
// * Handing in a scale parameter that causes the value to not fit in the bounds of a
//   (potentially signed) 64 bit integer
// * Handing in a negative value while `storeSigned` is false.
//
// Any violations of these pre-conditions will technically result in undefined behavior, and
// if your input is at risk then you should consider using the alternate FloatFixedCodec
// implementation provided below.
struct FixedPoint
{
    BAZ_CUDA static uint64_t Apply(float f, StoreSigned storeSigned, FixedPointScale scale)
    {
        auto scaled = roundf(f*scale);
        // Check for various flavors of under/over flow
        if (storeSigned)
        {
            assert(nextafterf(static_cast<float>(std::numeric_limits<int64_t>::max()), 0) > scaled);
            assert(nextafterf(static_cast<float>(std::numeric_limits<int64_t>::lowest()), 0) < scaled);
            // Need to cast to a signed type first to make sure we handle negatives correctly.
            // Casting a signed to an unsigned works fine as long as you assume 2s complement
            // (which we do), but casting a float to an int is undefined if the value is out of
            // range.
            return static_cast<uint64_t>(static_cast<int64_t>(scaled));
        } else {
            assert(scaled >= 0);
            assert(nextafterf(static_cast<float>(std::numeric_limits<uint64_t>::max()), 0) > scaled);
            return static_cast<uint64_t>(scaled);
        }
    }

    template <typename Ret>
    BAZ_CUDA static Ret Revert(uint64_t val, StoreSigned storeSigned, FixedPointScale scale)
    {
        static_assert(std::is_same<Ret, float>::value,"");
        if (storeSigned)
            return static_cast<float>(static_cast<int64_t>(val)) / scale;
        else
            return static_cast<float>(val) / scale;
    }

    static TransformsParams Params(FixedPointScale scale)
    {
        TransformsParams tp;
        FixedPointParams params;
        params.scale = scale;
        tp.params = params;
        return tp;
    }
};

// A slightly more robust codec for floats using a fixed point representation but also handling
// special values.  All float inputs are valid.  Inf/Nan values are preserved, and if a fixed
// point representation doesn't fit within the specified number of bytes, it will be stored as
// an infinity.
struct FloatFixedCodec
{
    BAZ_CUDA static uint64_t Apply(float f,
                                   StoreSigned storeSigned,
                                   FixedPointScale scale,
                                   NumBytes bytes)
    {
        assert(bytes <= 8);

        auto scaled = roundf(f*scale);
        // Check if we are in bounds.  We unfortunately have to handle signed and unsigned representations
        // separately in order to correctly handle the sign bit, despite the fact that the below code looks
        // like it's nearly duplicated.
        if (storeSigned)
        {
            const auto max = static_cast<int64_t>((1ull << (8 * bytes - 1)) - 1 - offset);
            // Note: max is about to be cast by float (made explicit for clarity, but it would happen anyway)
            //       It's possible that neither max nor scaled are finite, but that is fine and these
            //       conditionals were crafted with that in mind.  That will cause us to fail
            //       this conditional and fall back to the special case handling below
            if (static_cast<float>(-max) <= scaled && static_cast<float>(max) >= scaled)
            {
                if (scaled >= 0) scaled += offset;
                // Need to cast to a signed type first to make sure we handle negatives correctly.
                // Casting a signed to an unsigned works fine as long as you assume 2s complement
                // (which we do), but casting a float to an int is undefined if the value is out of
                // range.
                return static_cast<uint64_t>(static_cast<int64_t>(scaled));
            }
        } else {
            const auto max = ((1ull << (8 * bytes)) - 1 - offset);
            // Note: max is about to be cast by float (made explicit for clarity, but it would happen anyway)
            //       It's possible that neither max nor scaled are finite, but that is fine and these
            //       conditionals were crafted with that in mind.  That will cause us to fail
            //       this conditional and fall back to the special case handling below
            if (0 <= scaled && static_cast<float>(max) >= scaled)
            {
                scaled += offset;
                return static_cast<uint64_t>(scaled);
            }
        }
        if (std::isnan(f)) return nan;
        if (scaled < 0) return ninf;
        else return inf;
    }

    template <typename Ret>
    BAZ_CUDA static Ret Revert(uint64_t val,
                               StoreSigned storeSigned,
                               FixedPointScale scale,
                               // Only present for symmetry of arguments between serialization and deserialization.
                               // Not actually used in production, but some test code currently needs it
                               NumBytes /*unused*/ = NumBytes{0})
    {
        static_assert(std::is_same<Ret, float>::value,"");

        if (val == nan) return std::numeric_limits<Ret>::quiet_NaN();
        if (val == inf) return std::numeric_limits<Ret>::infinity();
        if (val == ninf) return -std::numeric_limits<Ret>::infinity();

        if (storeSigned)
        {
            auto signedVal = static_cast<int64_t>(val);
            if (signedVal >= 0) signedVal -= offset;
            return static_cast<float>(signedVal) / scale;
        }
        else
        {
            return static_cast<float>(val - offset) / scale;
        }
    }

    static TransformsParams Params(FixedPointScale scale, NumBytes numBytes)
    {
        TransformsParams tp;
        FloatFixedCodecParams params;
        params.scale = scale;
        params.numBytes = numBytes;
        tp.params = params;
        return tp;
    }

    static constexpr uint64_t nan = 0x0;
    static constexpr uint64_t inf = nan+1;
    static constexpr uint32_t ninf = inf+1;
    // Offset positive values need to be shifted by, to make room for
    // the special values just defined
    static constexpr uint32_t offset = ninf+1;
};

// Some algebra to make the below bit twiddles make a bit more sense
//
// Previous incarnations of this codec were described in terms of a
// mantissa and exponent.  This description has problems though, as
// we don't have numbers of the form `a*2^b`, but rather `offset + a*2^b`,
// where the offset is a function of both a and b.
//
// Instead we will describe is as having M bits for counting, with the remaining
// bits used to determine both our stride while counting as well as our starting
// offset. for instance if M=6, we have the following:
//   * Count from 0-63 by 1s
//   * Then from 64 to 190 by 2s,
//   * Then from 192 to 444 by 4s
//   * Then from 448 to 952 by 8s
//   Assuming we only have 8 bits total, then these 4 "groups" use up our remaining two
//   two bits and our value is saturated.
//
// If we generalize this and use G_N for our grouping, M for our number of bits, and F = 2**M,
// we have:
//    * G_0 = 0*F + [0, 1*(F-1)]
//    * G_1 = 1*F + [0, 2*(F-1)]
//    * G_2 = 3*F + [0, 4*(F-1)]
//    * G_3 = 7*F + [0, 8*(F-1)]
//     ...
//    * G_N = (2^N - 1)*F + [0, 2^N * (F-1)]
//
// Now consider any number X belonging to an unknown group
//    * (2^N - 1) * F <= X     <= (2^N - 1) * F + 2^N * (F-1)
//    * 2^N * F       <= X + F <= 2^N * F + 2^N * (F-1)
//
//    As the second term on the right hand side is less than the first,
//    and since we're dealing with powers of two, we can determine which
//    "group" a number belongs to by adding F, then finding the position
//    of the most significant bit, and then subtracting M.
//
//    Similarly, if we clear that high bit, then the bits that are left
//    describe the range [0, 2^N * (F-1)].  If we shift evertyhing N bits
//    to the right, then we've mapped our value into the M bits via
//    truncation.  If we instead add 2^(N-1) before shifting, then we've
//    mapped our value into the M bits via round-to-nearest.
struct LossySequelCodec
{
    BAZ_CUDA static uint64_t Apply(uint64_t t, StoreSigned storeSigned, NumBits bits)
    {
        assert(!storeSigned);

        uint64_t F = 1ul << bits;
        t += F;
#ifdef __CUDA_ARCH__
        auto highBit = 64 - __clzll(t) - 1;
#else
        auto highBit = 64 - __builtin_clzl(t) - 1;
#endif
        int group = highBit - bits;
        t = t ^ (1ul << highBit);
        auto round = (1ul << group) >> 1;
        uint64_t main = (t + round) >> group;
        return main + (group << bits);
    }

    template <typename Ret>
    BAZ_CUDA static Ret Revert(uint64_t t, StoreSigned storeSigned, NumBits bits)
    {
        assert(!storeSigned);

        auto group = (t >> bits);
        auto main = t & ((1ul << bits) - 1);
        uint64_t F = 1ul << bits;
        auto multiplier = 1ul << group;
        auto base = (multiplier - 1) * F;
        return static_cast<Ret>(base + multiplier * main);
    }

    static TransformsParams Params(NumBits bits)
    {
        TransformsParams tp;
        CodecParams params;
        params.numBits = bits;
        tp.params = params;
        return tp;
    }
};

struct DeltaCompression
{
    template <typename T>
    BAZ_CUDA uint64_t Apply(const T& t, StoreSigned)
    {
        static_assert(std::is_integral<T>::value, "");
        auto ut = static_cast<uint64_t>(t);
        auto ret = ut - lastVal;
        lastVal = ut;
        return ret;
    }
    template <typename Ret>
    BAZ_CUDA Ret Revert(uint64_t val, StoreSigned)
    {
        lastVal += val;
        return lastVal;
    }

    static TransformsParams Params()
    {
        TransformsParams tp;
        DeltaCompressionParams params;
        tp.params = params;
        return tp;
    }
private:
    // Storing this as unsigned, but really only so
    // that over/underflow is well defined.  Since we're
    // two's compliment, adding/subtracting are the same
    // operation regardless of sign, and the StoreSigned
    // parameter is already handed in to instruct us if we
    // should interpret the most significant bit as a sign
    // bit or not.
    uint64_t lastVal = 0;
};

}}

#endif //PACBIO_BAZIO_ENCODING_FIELD_TRANSFORMS_H
