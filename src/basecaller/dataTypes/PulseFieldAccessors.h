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

#ifndef PACBIO_DATA_PULSE_FIELD_ACCESSORS_H
#define PACBIO_DATA_PULSE_FIELD_ACCESSORS_H

#include <common/cuda/CudaFunctionDecorators.h>
#include <bazio/encoding/FieldNames.h>
#include <bazio/encoding/FieldAccessors.h>
#include <dataTypes/Pulse.h>

namespace PacBio::BazIO {

// Specialization class, to plug our Kestrel Pulse data structure
// into the bazio serialization framework
template <>
struct FieldAccessor<Mongo::Data::Pulse, PacketFieldName>
{
    template <PacketFieldName::RawEnum Name>
    CUDA_ENABLED static auto Get(const Mongo::Data::Pulse& p)
    {
        if constexpr (Name == PacketFieldName::Label)
            return static_cast<uint8_t>(p.Label());
        else if constexpr (Name == PacketFieldName::StartFrame)
            return p.Start();
        else if constexpr (Name == PacketFieldName::PulseWidth)
            return p.Width();
        else if constexpr (Name == PacketFieldName::IsBase)
            return !p.IsReject();
        else if constexpr (Name == PacketFieldName::Pkmax)
            return p.MaxSignal();
        else if constexpr (Name == PacketFieldName::Pkmid)
            return p.MidSignal();
        else if constexpr (Name == PacketFieldName::Pkmean)
            return p.MeanSignal();
        else if constexpr (Name == PacketFieldName::Pkvar)
            return p.SignalM2();
        else
            static_assert(Name == PacketFieldName::Label,
                          "PacketFieldName not supported with Mongo::Data::Pulse");
        // NVCC seems to have a diagnostic bug, where it warns about no return statement
        // in a function not returning void.  This builtin helps silence that, even
        // though the constexpr statements themselves should be enough.
        __builtin_unreachable();
    }

    template <PacketFieldName::RawEnum Name>
    using Type = decltype(Get<Name>(std::declval<Mongo::Data::Pulse>()));

    template <PacketFieldName::RawEnum Name>
    CUDA_ENABLED static void Set(Mongo::Data::Pulse& p, Type<Name> val)
    {
        if constexpr (Name == PacketFieldName::Label)
            p.Label(static_cast<Mongo::Data::Pulse::NucleotideLabel>(val));
        else if constexpr (Name == PacketFieldName::StartFrame)
            p.Start(val);
        else if constexpr (Name == PacketFieldName::PulseWidth)
            p.Width(val);
        else if constexpr (Name == PacketFieldName::IsBase)
            p.IsReject(!val);
        else if constexpr (Name == PacketFieldName::Pkmax)
            p.MaxSignal(val);
        else if constexpr (Name == PacketFieldName::Pkmid)
            p.MidSignal(val);
        else if constexpr (Name == PacketFieldName::Pkmean)
            p.MeanSignal(val);
        else if constexpr (Name == PacketFieldName::Pkvar)
            p.SignalM2(val);
        else
            static_assert(Name == PacketFieldName::Label,
                          "PacketFieldName not supported with Mongo::Data::Pulse");
    }
};

}

#endif //PACBIO_DATA_PULSE_FIELD_ACCESSORS_H
