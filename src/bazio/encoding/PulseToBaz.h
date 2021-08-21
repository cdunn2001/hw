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
/// \File PulseToBaz.h
/// \detail This file is a preliminary stab at writing a generic pulse to baz stream
///         using template metaprogramming.  It is targeted at serializing pulses,
///         where we are generally under stricter real-time constraints and performance
///         matters, though it does have the inverse baz to pulse functionality for
///         test reasons.
///
///         Some of the generic programming done here is perhaps a bit involved, but
///         its justification is best viewed in light of the alternatives:
///             * We could hand-code individual formats like was done in the Sequel code
///               base.  However this is a manual and potentially error prone process.
///               We lived for a long time now with an under-tuned internal mode format
///               that comes with strong ROI constraints in Sequel II, simply because we've
///               never had the time to revisit the code and fix the issues.  Also now
///               that Kestrel is dealing with a much more relaxed sub-byte and cross-byte
///               possibilities, playing with hand rolled code anytime we wish to make a
///               change has a high chance of bugs.
///             * We could write a general parser like this, but based off runtime rather
///               that compile time information, using switches and/or virtual dispatch.
///               This was tried, and at the time of writing this there exists a
///               BazToPulse file that took this approach.  However it was measured to be
///               6-7 times slower than this approach, and that was before all features
///               were added.  Even now that Kestrel is moving to a multi-threaded
///               baz writing process, our throughput requirements are high enough that
///               it doesn't seem prudent to accept an order of magnitude slowdown.
///               * Note: If you want to view the version of BazToPulse being reference,
///                       you will likely have to go back to the git commit that added
///                       this comment.  The current version is just a POC implementation
///                       that goes from pulses to baz and back, but I expect that to change
///                       soon.  The current usecase in PPA is to just convert the byte
///                       steam into vector of vector of ints, not all the way back to
///                       a pulse data structure
///
///          The code here is robust and flexible.  The bit twiddles are written once,
///          so changing what format we choose to serialize shouldn't introduce new bugs.
///          Also since all serialization information is viewable by the compiler, it
///          can do some pretty aggressive inlining and optimization.  I've hand checked
///          some of the generated assembly, and it's pretty close to what you'd get from
///          writing things manually.  Things are also very easy to update.  Changing
///          what information gets serialized is just a matter of tweaking a template
///          parameter.
///
///          There is a fair bit of template machinery in this file that can make it
///          hard to sort out what are the important bits of implementation.  To understand
///          how things function, the most important classes are:
///          * GroupEncoder: Controls the mechanics of serializing a group (fields that take
///                          up an integral number of bytes) to the actual byte stream.
///          * FieldGroupEncoder: Mostly the same, but now knows what field names to
///                               associate with the group.
///          * PulseEncoder: The top level, capable of serializing an entire pulse

#ifndef PACBIO_BAZIO_ENCODING_PULSE_TO_BAZ_H
#define PACBIO_BAZIO_ENCODING_PULSE_TO_BAZ_H

#include <cstddef>
#include <numeric>
#include <utility>

#include <bazio/encoding/BazioEnableCuda.h>
#include <bazio/encoding/EncodingParams.h>
#include <bazio/encoding/FieldNames.h>
#include <bazio/encoding/FieldAccessors.h>
#include <bazio/encoding/FieldSerializers.h>
#include <bazio/encoding/FieldTransforms.h>

#include <common/cuda/utility/CudaTuple.h>

namespace PacBio {
namespace BazIO {

namespace detail {

template <typename T1, typename... Rest>
struct FirstOf { using type = T1; };
template <typename... Ts>
using FirstOf_t = typename FirstOf<Ts...>::type;

template <auto Name1, auto...Names>
struct EnumFromNames
{
    static_assert((true && ... && std::is_same<
                   EnumFromRaw_t<decltype(Name1)>,
                   EnumFromRaw_t<decltype(Names)>
                   >::value),
                  "Error: Received entries from multiple different enums");

    using type = EnumFromRaw_t<decltype(Name1)>;
};
template <auto...Names>
using EnumFromNames_t = typename EnumFromNames<Names...>::type;

// Just a helper struct to record a combination of Transformation and Serialization
// that will end up applied to individual fields.
template <typename StoreSigned, typename Trans, typename Serial>
struct EncodeInfo {};

// Serializes an individual group of fields.  Fields are grouped together such
// that the sum of their individual bits is as close to a multiple of 8 as possible,
// while still being <= 64.  During serialization, first each field writes it's main
// portion, and then after that any necessary overflow information is recorded.
template <typename Idxs, typename...Info> struct GroupEncoder {};
template <size_t... idxs, typename... Signed, typename... Transforms, typename... Serializers>
struct GroupEncoder<std::index_sequence<idxs...>, EncodeInfo<Signed, Transforms, Serializers>...>
{
    // Compute the number of bits we require for this group, not counting
    // any data-dependent overflow bytes
    BAZ_CUDA static constexpr size_t TotalBits()
    {
        return (0 + ... + Serializers::nBits);
    }
    static constexpr size_t numBytes = (TotalBits() + 7) / 8;

    static_assert(numBytes <= 8, "Cannot use more than 64 bit storage");

    // Used to determine how much space would hypthetically be required to store
    // the supplied values.
    template <typename... Ts>
    BAZ_CUDA size_t BytesRequired(const Ts&... ts)
    {
        return (numBytes + ... + Serializers::OverflowBytes(transforms_.template Get<idxs>().Apply(ts, Signed::val), Signed::val));
    }

    // Writes a series of values to a memory location.  The return value is a pointer to the
    // next byte after the current writes.
    template <typename... Ts>
    BAZ_CUDA uint8_t* Encode(uint8_t* ptr, const Ts&... ts)
    {
        static_assert(sizeof...(Ts) == sizeof...(Serializers), "Wrong number of arguments");
        uint64_t data = 0;
        auto ptr2 = ptr+numBytes;

        size_t pos = 0;
        auto Insert = [&](auto val, size_t bits)
        {
            static_assert(std::is_integral<decltype(val)>::value, "");
            data |= (val & ((1 << bits)-1)) << pos;
            pos += bits;
        };

        (Insert(Serializers::ToBinary(transforms_.template Get<idxs>().Apply(ts, Signed::val),
                                      ptr2,
                                      Signed::val),
                Serializers::nBits)
         , ...);

        std::memcpy(ptr, &data, numBytes);

        return ptr2;
    }

    // Reads a series of values from a memory location.  The return value is a pointer to the
    // next byte after the curent reads.
    template <typename... Ts>
    BAZ_CUDA const uint8_t* Decode(const uint8_t* ptr, Ts&... ts)
    {
        uint64_t data = 0;
        std::memcpy(&data, ptr, numBytes);
        ptr += numBytes;

        auto Extract = [&](size_t bits)
        {
            uint64_t val = data & ((1 << bits) - 1);
            data = data >> bits;
            return val;
        };
        ((ts = transforms_.template Get<idxs>().template Revert<Ts>(
                Serializers::FromBinary(
                    Extract(Serializers::nBits),
                    ptr, Signed::val),
                Signed::val))
         , ...);
        return ptr;
    }

    template <typename RawEnum, typename TransformParams, typename SerializerParams>
    static auto FieldParam(RawEnum name, StoreSigned storeSigned,
                           TransformParams transformParams, SerializerParams serializerParams)
    {
        using FieldName = EnumFromRaw_t<RawEnum>;
        FieldParams<FieldName> fp;
        fp.name = name;
        fp.storeSigned = storeSigned;
        fp.transform = transformParams;
        fp.serialize = serializerParams;
        return fp;
    }

    template <typename... Names>
    static auto Params(Names... names)
    {
        using FieldName = EnumFromRaw_t<FirstOf_t<Names...>>;
        static_assert((true && ... && std::is_same<FieldName, EnumFromRaw_t<Names>>::value), "Function called with RawEnums from different SmartEnum classes");

        GroupParams<FieldName> params;
        params.members = {FieldParam(names, Signed::val, Transforms::Params(), Serializers::Params())...};
        params.numBits = {Serializers::nBits...};
        params.totalBits = (0 + ... + Serializers::nBits);
        return params;
    }

    PacBio::Cuda::Utility::CudaTuple<Transforms...> transforms_;
};

// Like an EncodeInfo, but now we add a particular FieldName to the mix.
template <auto name, typename StoreSigned, typename Trans, typename Serial>
struct Field
{
    // Add some validation of the template parameters, since this is something
    // external code will be interacting with.
    template <typename U, typename...params>
    BAZ_CUDA static constexpr bool IsSerializer(Serialize<U, params...>*) { return true; }
    BAZ_CUDA static constexpr bool IsSerializer(...) { return false; }

    template <typename U, typename...params>
    BAZ_CUDA static constexpr bool IsTransformer(Transform<U, params...>*) { return true; }
    BAZ_CUDA static constexpr bool IsTransformer(...) { return false; }

    template <bool b>
    BAZ_CUDA static constexpr bool IsStoreSigned(StoreSigned_t<b>*) { return true; }
    BAZ_CUDA static constexpr bool IsStoreSigned(...) { return false; }

    static_assert(IsSerializer(static_cast<Serial*>(nullptr)), "");
    static_assert(IsTransformer(static_cast<Trans*>(nullptr)), "");
    static_assert(IsStoreSigned(static_cast<StoreSigned*>(nullptr)), "");

    static constexpr auto nBits = Serial::nBits;
};

// A collection of fields that will form a "group"
template <size_t numBits, typename... Fields>
struct FieldGroup {
    static constexpr auto totalBits = numBits;
};

// Just a struct to hold an arbitary list of template parameters.
template <typename...T> struct Pack{};

// This class just wraps the functionality of a GroupEncoder, but now
// associated with a particular list of FieldNames.  This class
// is kept separate from the GroupEncoder mostly to avoid unecessary
// extra template instantiations, since two groups can have the same
// formula for encoding fields, just with a different set of fields.
template <typename GroupEncoder, auto... names>
struct FieldGroupEncoder
{
    using FieldNames = EnumFromNames_t<names...>;
    template <typename P>
    BAZ_CUDA uint8_t* Serialize(const P& pulse, uint8_t* dest)
    {
        return groupEncoder_.Encode(dest, FieldAccessor<P, FieldNames>::template Get<names>(pulse)...);
    }

    template <typename P, size_t...ids>
    BAZ_CUDA const uint8_t* DeserializeHelper(P& pulse, const uint8_t* dest, std::index_sequence<ids...>)
    {
        using Accessor = FieldAccessor<P, FieldNames>;
        PacBio::Cuda::Utility::CudaTuple<typename Accessor::template Type<names>...> vals;
        dest = groupEncoder_.Decode(dest, vals.template Get<ids>()...);
        (Accessor::template Set<names>(pulse, vals.template Get<ids>()) , ...);
        return dest;
    }

    template <typename P>
    BAZ_CUDA const uint8_t* Deserialize(P& pulse, const uint8_t* dest)
    {
        return DeserializeHelper(pulse, dest, std::make_index_sequence<sizeof...(names)>{});
    }

    template <typename P, size_t...ids>
    BAZ_CUDA size_t BytesRequired(const P& pulse)
    {
        return groupEncoder_.BytesRequired(FieldAccessor<P, FieldNames>::template Get<names>(pulse)...);
    }

    static auto Params()
    {
        return GroupEncoder::Params(names...);
    }

    GroupEncoder groupEncoder_;
    // The nominal number of bytes we expect to use to serialize this group,
    // neglecting the effects from any overflow mechanisms
    static constexpr size_t nominalBytes = GroupEncoder::numBytes;
};

template <typename FieldGroup>
struct GroupToEncoder;
template <size_t totalBits, auto... names, typename... Signed, typename... Transforms, typename... Serializers>
struct GroupToEncoder<FieldGroup<totalBits, Field<names, Signed, Transforms, Serializers>...>> {
    using T = FieldGroupEncoder<GroupEncoder<std::make_index_sequence<sizeof...(Transforms)>, EncodeInfo<Signed, Transforms, Serializers>...>, names...>;
};
template <typename Encoding>
using GroupToEncoder_t = typename GroupToEncoder<Encoding>::T;

template <typename... GroupEncoders>
struct PulseEncoder
{
    // The nominal number of bytes we expect to use to serialize this group,
    // neglecting the effects from any overflow mechanisms
    static constexpr size_t nominalBytes = (0 + ... + GroupEncoders::nominalBytes);

    template <typename P>
    BAZ_CUDA uint8_t* Serialize(const P& pulse, uint8_t* dest)
    {
        ((dest = data_.template Get<GroupEncoders>().Serialize(pulse, dest)) , ...);
        return dest;
    }
    template <typename P>
    BAZ_CUDA const uint8_t* Deserialize(P& pulse, const uint8_t* dest)
    {
        ((dest = data_.template Get<GroupEncoders>().Deserialize(pulse, dest)) , ...);
        return dest;
    }
    template <typename P>
    BAZ_CUDA size_t BytesRequired(const P& pulse)
    {
        return (0 + ... + data_.template Get<GroupEncoders>().BytesRequired(pulse));
    }

    BAZ_CUDA void Reset()
    {
        data_ = PacBio::Cuda::Utility::CudaTuple<GroupEncoders...>{};
    }

    static auto Params()
    {
        using type = FirstOf_t<decltype(GroupEncoders::Params())...>;
        static_assert((true && ... && std::is_same<type, decltype(GroupEncoders::Params())>::value),
                      "Function called with RawEnums from different SmartEnum classes");
        return std::vector<type>{GroupEncoders::Params()...};
    }

private:

    PacBio::Cuda::Utility::CudaTuple<GroupEncoders...> data_;
};

template <typename ProcessedPack,
          typename ProgressEncoding,
          typename... Fields>
struct GenerateGroups;

template <typename... ProcessedEncodings,
          typename ProgressEncoding>
struct GenerateGroups<Pack<ProcessedEncodings...>, ProgressEncoding> {
    using T = PulseEncoder<GroupToEncoder_t<ProcessedEncodings>..., GroupToEncoder_t<ProgressEncoding>>;
};

template <typename... ProcessedGroups,
          size_t currBits,
          typename...CurrFields,
          typename Head, typename... Tail>
struct GenerateGroups<Pack<ProcessedGroups...>, FieldGroup<currBits, CurrFields...>, Head, Tail...>
{
    using CurrEncoding = FieldGroup<currBits, CurrFields...>;
    static constexpr auto currNumBits = CurrEncoding::totalBits;
    static constexpr auto nextNumBits = CurrEncoding::totalBits + Head::nBits;
    static constexpr auto evenBytes = currNumBits % 8 == 0 && currNumBits > 0;
    static constexpr auto overflowCapacity = nextNumBits > 64;

    using T = std::conditional_t<evenBytes || overflowCapacity,
                                 typename GenerateGroups<Pack<ProcessedGroups..., CurrEncoding>, FieldGroup<Head::nBits, Head>, Tail...>::T,
                                 typename GenerateGroups<Pack<ProcessedGroups...>, FieldGroup<nextNumBits, CurrFields..., Head>, Tail...>::T>;
};

}

// This is the front-end type to use.  It accepts a flat list of fieldnames and their
// respecive transformations/serializations.  Automatic template machinery
// will automatically transform that flat list into the heirarchical groups.
// Groups will be formed by the smallest number of fields that tally up to
// an integral number of bytes, with the exception that a group cannot be
// larger than 64 bits, and if we have to tie off a ragged group to avoid
// that limitation then we will.
template <auto name, typename Signed, typename Trans, typename Serial>
using Field = detail::Field<name, Signed, Trans, Serial>;
template <typename...Fields>
using PulseToBaz = typename detail::GenerateGroups<detail::Pack<>, detail::FieldGroup<0>, Fields...>::T;

}}

#endif //PACBIO_BAZIO_ENCODING_PULSE_TO_BAZ_H
