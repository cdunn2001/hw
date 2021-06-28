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
/// \file FieldSerializers.h
/// \details Defines the various ways by which a value, after being
///          transformed in some fashion into an integral type, is
///          serialized to the baz file.  For all of these methods
///          you can configure how many bits are used for primary
///          storage, and they differ in how they handle the data
///          when the specified bits are not enough to represent the
///          provided value.  If we need to overflow we can either:
///          * Truncate the value.  This is provided for computational
///            efficiency, it should not be used unless something
///            upstream can guarantee no overflow will occur
///          * Use a pre-specified number of overflow bytes
///          * Use a dynamically determined number of overflow bytes,
///            which will only use 7 bits of storage per byte, but
///            is capable of using fewer bytes overal for most numbers.
///
///          Beyond that, there is a Serialize class, suitable for
///          use in template metaprogramming.  Where the serializers
///          themselves use runtime values to control their behaviour,
///          the Serialize class uses only compile time values, and
///          can be used to promote extensive inlining/optimization by
///          the compiler for chosen configurations.


#ifndef PACBIO_BAZIO_ENCODING_FIELD_SERIALIZERS_H
#define PACBIO_BAZIO_ENCODING_FIELD_SERIALIZERS_H

#include <cassert>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <type_traits>
#include <limits>

#include <bazio/encoding/BazioEnableCuda.h>

#include <bazio/encoding/Types.h>

namespace PacBio {
namespace BazIO {

// Generic Serializer function for template metaprogramming use.
// The first parameter the type of serializer to use, with the
// subsequent parameters to be used as function arguments.
// Invoking things through this class allows the compiler to see
// how the functions are used, and optimize them more aggresively
// than if the below serializers were used directly.
template <typename Base, typename numBits, typename... autoParams>
struct Serialize
{
    static constexpr uint8_t nBits = numBits::val;

    /// \param t The value in question (must be integral type)
    /// \returns The number of overflow bytes that will be required
    ///          to serialize the provided value
    BAZ_CUDA static size_t OverflowBytes(uint64_t val, StoreSigned storeSigned)
    {
        return Base::OverflowBytes(val, storeSigned, numBits::val, autoParams::val...);
    }

    /// Converts the provided value to binary.
    /// \param val The value to be serialized
    /// \param ptr A reference to a pointer for overflow data storage.
    ///            If overflow space is needed, this parameter
    ///            will be incremented to the next unused address
    /// \returns The "main portion", which has a payload of the specified
    ///          number of bits.  Some bits or bit patterns may be
    ///          reserved to indicate that an overflow has happened
    BAZ_CUDA static uint64_t ToBinary(uint64_t val, uint8_t*& ptr, StoreSigned storeSigned)
    {
        return Base::ToBinary(val, ptr, storeSigned, numBits::val, autoParams::val...);
    }

    /// Converts a binary stream to a integral value
    /// \param val The "main portion" of the data being read in.  Only the first
    ///            numBits will be used
    /// \param ptr A reference to a pointer for the overflow data storage.
    ///            If overflow data was read, this parameter will be
    ///            incrememted to the next unread address
    /// \returns The deserialized value
    BAZ_CUDA static uint64_t FromBinary(uint64_t val, uint8_t const*& ptr, StoreSigned storeSigned)
    {
        return Base::FromBinary(val, ptr, storeSigned, numBits::val, autoParams::val...);
    }
};

/// \brief Serializer that just truncates the data if it doesn't fit
///        in the specified bits.  Only suitable for use if you know
///        things won't overflow, and it's main purpose is to just not
///        do any unecessary operations
struct TruncateOverflow
{
    BAZ_CUDA static size_t OverflowBytes(uint64_t, StoreSigned, NumBits)
    {
        return 0;
    }

    BAZ_CUDA static uint64_t ToBinary(uint64_t val, uint8_t*&, StoreSigned, NumBits)
    {
        return val;
    }

    BAZ_CUDA static uint64_t FromBinary(uint64_t val, const uint8_t*&, StoreSigned storeSigned, NumBits numBits)
    {
        if (storeSigned)
        {
            // If we have a signed value and the most significant bit
            // read is one, then we need to pad out sign bits to fill
            // the rest of our wider destination type.
            if (val & (1ul << (numBits-1)))
                val |= (static_cast<uint64_t>(-1) << numBits);
        }
        return val;
    }
};

/// \brief Serializer with a simple overflow scheme.  If the value will not fit
///        in the specified number of bits, then all bits in the "main" portion
///        are set to 1, and the `NumBytes` are used in the overflow section to
///        represent the value.
///        Note: If `NumBytes` is still not enough to represent the value, the
///              data will be truncated.
///        Note: Signifying all bits set to 1 for overflow is a touch awkward for
///              signed types, since the reserved value is equivalent to -1 and
///              in the middle of our range.  However we want the same bit patern
///              for overflows with both signed and unsigned numbers, so that any
///              bugs where the serializer/desierilaizer have a signed/unsigned
///              mismatch will just corrupt the individually read value, and
///              not potentially downstream values because overflow data was not
///              properly consumed (or improperly consumed).
struct SimpleOverflow
{
    // Technically implementation defined until c++20,
    // but I seriously doubt we're going to run on any
    // hardware that is not 2s complement
    static_assert(-2 >> 1 == -1, "");

    BAZ_CUDA static bool NeedsOverflow(uint64_t val, StoreSigned storeSigned, NumBits numBits)
    {
        // We know we're on hardware that's twos compliment, so just cast things
        // to an usigned int.  Now we can add/subtact whatever and have well
        // defined overflow behaviour.
        if (storeSigned)
        {
            // The usual N bit signed integer range is from -2^(N-1) to 2^(N-1)-1.  We're going
            // to reserve the bit pattern of all 1s to indicate overflow, so we
            // need to remove one value from the negative range, -2^(N-1) + 1 to 2^(N-1)-1.
            // If we add 2^(N-1)-1 to our value, then the range is 0 to 2^N-2, and
            // we no longer have to worry about the fact that our unsigned value was
            // really a signed value.
            val += (1ul << (numBits-1)) - 1;
        }
        uint64_t max = (1ul << numBits) - 1;

        if (val >= max) return true;
        else return false;
    }

    BAZ_CUDA static size_t OverflowBytes(uint64_t val, StoreSigned storeSigned, NumBits numBits, NumBytes overflowSize)
    {
        if(NeedsOverflow(val, storeSigned, numBits))
            return overflowSize;
        else
            return 0;
    }

    BAZ_CUDA static uint64_t ToBinary(uint64_t val, uint8_t*& ptr, StoreSigned storeSigned, NumBits numBits, NumBytes overflowSize)
    {
        if (NeedsOverflow(val, storeSigned, numBits))
        {
            memcpy(ptr, &val, overflowSize);
            ptr += overflowSize;
            return (1ul<<numBits)-1;
        }
        else
        {
            if (storeSigned)
            {
                // Add 2(^N-1)-1 to our value (using unsigned types to avoid undefined overflow)
                // so that none of our valid numbers have the reserved bit pattern of all 1s,
                // incidating an overflow.
                auto signedOffset = (1ul << (numBits-1)) - 1;
                val += signedOffset;
            }
            return val;
        }
    }

    BAZ_CUDA static uint64_t FromBinary(uint64_t val, const uint8_t*& ptr, StoreSigned storeSigned, NumBits numBits, NumBytes overflowSize)
    {
        uint64_t ret = val;
        if (ret == (1ul<<numBits)-1)
        {
            std::memcpy(&ret, ptr, overflowSize);
            ptr += overflowSize;
            auto signBit = 8*overflowSize;
            if (storeSigned && (ret & (1ul << (signBit-1))))
                ret |= (static_cast<uint64_t>(-1) << (signBit-1));
        } else if (storeSigned)
        {
            // Subtract the offset we added during serialization
            auto signedOffset = (1ul << (numBits-1)) - 1;
            ret = val - signedOffset;
        }
        return ret;
    }
};

/// \brief Serializer with a compact overflow scheme.  One bit from
///        the main portion is used to specify if the remaining bits
///        are sufficient to hold the data, or if any overflow is required.
///        If any overflow is required, then each overflow byte will have
///        a bit set to indicate if there is another overflow byte that
///        needs to be read in.
struct CompactOverflow
{
    // Technically implementation defined until c++20,
    // but I seriously doubt we're going to run on any
    // hardware that is not 2s complement
    static_assert(-2 >> 1 == -1, "");

    BAZ_CUDA static size_t OverflowBytes(uint64_t t, StoreSigned storeSigned, NumBits numBits)
    {
        // Separating the signed and unsigned paths, because I can't
        // seem to make the bit counting version nearly as fast
        // (there is almost a 2x difference).  I've tried several
        // formulations, and nothing seems to work as well as the
        // loop and shift strategy in the else, but that specifically
        // only works for unsigned integers.  (It won't even work for
        // positive signed integers, as it may truncate off the sign bit)
        if (storeSigned)
        {
#ifdef __CUDA_ARCH__
            auto discard = [&](){ if (static_cast<int64_t>(t) < 0) t = ~t; return __clzll(t) - 1;}();
#else
            auto discard = __builtin_clrsbl(t);
#endif
            auto keep = 64 - discard - (static_cast<int>(numBits)-1);
            return (keep + 6)/7;
        } else {
            t = t >> (numBits-1);
            size_t ret = 0;
            while(t)
            {
                ret++;
                t = t >> 7;
            }
            return ret;
        }
    }

    BAZ_CUDA static uint64_t ToBinary(uint64_t val, uint8_t*& ptr, StoreSigned storeSigned, NumBits numBits)
    {
        auto InitialContinueMask = 1ul << (numBits - 1);
        auto InitialValueMask = InitialContinueMask - 1;

        size_t numOverflow = OverflowBytes(val, storeSigned, numBits);
        uint64_t initialVal = val & InitialValueMask;
        val = val >> (numBits-1);
        if (numOverflow != 0) initialVal |= InitialContinueMask;

        while (numOverflow != 0)
        {
            static constexpr auto overflowContinueMask = 1u << 7;
            static constexpr auto overflowValueMask = overflowContinueMask - 1;
            uint8_t byte = static_cast<uint8_t>(val & overflowValueMask);
            val = val >> 7;
            numOverflow--;
            if (numOverflow > 0)
                byte |= overflowContinueMask;
            *ptr = byte;
            ptr++;
        }
        return initialVal;
    }

    BAZ_CUDA static uint64_t FromBinary(uint64_t val, const uint8_t*& ptr, StoreSigned storeSigned, NumBits numBits)
    {
        auto InitialContinueMask = 1ul << (numBits - 1);
        auto InitialValueMask = InitialContinueMask - 1;

        uint64_t ret = val & InitialValueMask;
        bool cont = val & InitialContinueMask;
        size_t currBits = numBits-1;
        while (cont)
        {
            static constexpr auto overflowContinueMask = 1ul << 7;
            static constexpr auto overflowValueMask = overflowContinueMask - 1;

            uint64_t byte = *ptr;
            ptr++;

            cont = byte & overflowContinueMask;
            ret |= (byte & overflowValueMask) << currBits;
            currBits += 7;
        }
        if (storeSigned && (ret & (1ul << (currBits-1))))
            ret |= static_cast<uint64_t>(-1) << currBits;
        return ret;
    }
};

}}

#endif //PACBIO_BAZIO_ENCODING_FIELD_SERIALIZERS_H
