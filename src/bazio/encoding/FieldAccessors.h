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
/// \brief Declares a template class, which client code shall specialize
///        in order to inject a mapping between entries in SMART_ENUM
///        and fields in some class/object to be serialized
///

#ifndef PACBIO_BAZIO_ENCODING_FIELD_ACCESSORS_H
#define PACBIO_BAZIO_ENCODING_FIELD_ACCESSORS_H

namespace PacBio::BazIO {

// Create specializations of this class to map entries in an enum
// to accessor functions in Obj.  `FieldNames` is expected to
// be an instance of a SMART_ENUM, and `Obj` can be effectively
// anything, but the intention is to be something like a Pulse
// or Metric object.
//
// At a minimum the specialization must include the function:
//   template <FieldNames::RawEnum Name>
//   static auto Get(const Obj& obj);
//
// If you wish to do deserialization as well as serialization
// (which is primarily for test paths right now) then you also
// need something like:
//
//   template <FieldName::RawEnum Name>
//   using Type = decltype(Get<Name>(std::declval<Obj>()));
//
//   template <FieldName::RawEnum Name>
//   static void Set(Obj& p, Type<Name> val);
//
template <typename Obj, typename FieldNames>
struct FieldAccessor
{
    static_assert(!sizeof(Obj), "Missing specialization for FieldAccessor!");
};

}  // namespace PacBio::BazIO

#endif  // PACBIO_BAZIO_ENCODING_FIELDNAMES_H
