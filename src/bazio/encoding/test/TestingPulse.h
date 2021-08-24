// Copyright (c) 2021 Pacific Biosciences of California, Inc.
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
//  Description:
/// \file   TestingPulse.h
/// \brief  A Pulse data structure, along with the necessary pieces to plug
///         it into the baz serialization framework.  This all started as a
///         fork of Kestrel code, but the fact that it comes from Kestrel
///         is incidental.  It really serves to decouple the BazIO project
///         from the external world, allowing them to potentially diverge
///         independantly

#ifndef PACBIO_BAZIO_ENCODING_TEST_TESTING_PULSE_H
#define PACBIO_BAZIO_ENCODING_TEST_TESTING_PULSE_H

#include <cstdint>
#include <limits>

#include <bazio/encoding/BazioEnableCuda.h>
#include <bazio/encoding/FieldAccessors.h>
#include <bazio/encoding/FieldNames.h>
#include <bazio/encoding/PulseToBaz.h>

namespace PacBio::BazIO {

class Pulse
{
public:  // Types
    enum class NucleotideLabel : uint8_t {
        A = 0,
        C,
        G,
        T,
        N,  // A, C, G, or T
        NONE
    };

public:  // Property accessors
    /// The first frame of this pulse.
    BAZ_CUDA uint32_t Start() const { return start_; }

    /// The first frame after the last frame of this pulse.
    BAZ_CUDA uint32_t Stop() const { return start_ + width_; }

    /// The number of frames in this pulse.
    BAZ_CUDA uint16_t Width() const { return width_; }

    /// The mean signal level in the pulse.
    BAZ_CUDA float MeanSignal() const { return meanSignal_; }

    /// \brief The mean of the signal over the interior frames of the pulse.
    /// \details If Width() < 3, MidSignal() is NaN.
    BAZ_CUDA float MidSignal() const
    {
        if (width_ < 3)
            return std::numeric_limits<float>::quiet_NaN();
        else
            return midSignal_;
    }

    /// The max signal level in the pulse.
    BAZ_CUDA float MaxSignal() const { return maxSignal_; }

    /// The sum of squares of the signal level for interior frames of the
    /// pulse.
    BAZ_CUDA float SignalM2() const
    {
        if (width_ < 3)
            return std::numeric_limits<float>::quiet_NaN();
        else
            return signalM2_;
    }

    /// \brief The label assigned to the pulse by the pulse classifier.
    /// \details This label is guaranteed to be a DNA base; we do not lable
    ///          pulses in any other manner (no "no-call" pulses).
    BAZ_CUDA NucleotideLabel Label() const { return label_; }

    /// \brief Whether or not the pulse call was excluded as incredible.
    BAZ_CUDA bool IsReject() const { return isReject_; }

public:  // Property modifiers
    /// Sets the start frame value.
    /// \returns Reference to \code *this.
    BAZ_CUDA Pulse& Start(uint32_t value)
    {
        start_ = value;
        return *this;
    }

    /// Sets the width in frames.
    /// \returns Reference to \code *this.
    BAZ_CUDA Pulse& Width(uint16_t value)
    {
        width_ = value;
        return *this;
    }

    /// Sets the mean signal value.
    /// \returns Reference to \code *this.
    BAZ_CUDA Pulse& MeanSignal(float value)
    {
        meanSignal_ = value;
        return *this;
    }

    /// \brief Sets the mid-pulse signal value.
    /// \detail The value will be stored, but will be hidden as long as
    /// Width() < 3.
    /// \returns Reference to \code *this.
    BAZ_CUDA Pulse& MidSignal(float value)
    {
        midSignal_ = value;
        return *this;
    }

    /// Sets the max signal value.
    /// \returns Reference to \code *this.
    BAZ_CUDA Pulse& MaxSignal(float value)
    {
        maxSignal_ = value;
        return *this;
    }

    /// Sets the sum of signal values
    /// \returns Reference to \code *this.
    BAZ_CUDA Pulse& SignalM2(float value)
    {
        signalM2_ = value;
        return *this;
    }

    /// \brief Sets the label.
    /// \detail \code
    /// \returns Reference to \code *this.
    BAZ_CUDA Pulse& Label(NucleotideLabel value)
    {
        label_ = value;
        return *this;
    }

    /// \brief Sets the rejected flag.
    /// \returns Reference to \code *this.
    BAZ_CUDA Pulse& IsReject(bool value)
    {
        isReject_ = value;
        return *this;
    }

private:              // Data
    uint32_t start_;  // frames
    uint16_t width_;  // frames

    float meanSignal_;
    float midSignal_;
    float maxSignal_;
    float signalM2_;

    NucleotideLabel label_;
    bool isReject_;
};

template <>
struct FieldAccessor<Pulse, PacketFieldName>
{
    template <PacketFieldName::RawEnum Name>
    BAZ_CUDA static auto Get(const Pulse& p)
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
                          "PacketFieldName not supported with BazIO::Pulse");
        // NVCC seems to have a diagnostic bug, where it warns about no return statement
        // in a function not returning void.  This builtin helps silence that, even
        // though the constexpr statements themselves should be enough.
        __builtin_unreachable();
    }

    template <PacketFieldName::RawEnum Name>
    using Type = decltype(Get<Name>(std::declval<Pulse>()));

    template <PacketFieldName::RawEnum Name>
    BAZ_CUDA static void Set(Pulse& p, Type<Name> val)
    {
        if constexpr (Name == PacketFieldName::Label)
            p.Label(static_cast<Pulse::NucleotideLabel>(val));
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
                          "PacketFieldName not supported with BazIO::Pulse");
    }
};

using ProductionPulses = BazIO::PulseToBaz<BazIO::Field<BazIO::PacketFieldName::Label,
                                                        BazIO::StoreSigned_t<false>,
                                                        BazIO::Transform<BazIO::NoOp>,
                                                        BazIO::Serialize<BazIO::TruncateOverflow, BazIO::NumBits_t<2>>>,
                                           BazIO::Field<BazIO::PacketFieldName::PulseWidth,
                                                        BazIO::StoreSigned_t<false>,
                                                        BazIO::Transform<BazIO::NoOp>,
                                                        BazIO::Serialize<BazIO::CompactOverflow, BazIO::NumBits_t<7>>>,
                                           BazIO::Field<BazIO::PacketFieldName::StartFrame,
                                                        BazIO::StoreSigned_t<false>,
                                                        BazIO::Transform<BazIO::DeltaCompression>,
                                                        BazIO::Serialize<BazIO::CompactOverflow, BazIO::NumBits_t<7>>>>;

using InternalPulses =
    BazIO::PulseToBaz<BazIO::Field<BazIO::PacketFieldName::Label,
                                   BazIO::StoreSigned_t<false>,
                                   BazIO::Transform<BazIO::NoOp>,
                                   BazIO::Serialize<BazIO::TruncateOverflow, BazIO::NumBits_t<2>>>,
                      BazIO::Field<BazIO::PacketFieldName::PulseWidth,
                                   BazIO::StoreSigned_t<false>,
                                   BazIO::Transform<BazIO::NoOp>,
                                   BazIO::Serialize<BazIO::CompactOverflow, BazIO::NumBits_t<7>>>,
                      BazIO::Field<BazIO::PacketFieldName::StartFrame,
                                   BazIO::StoreSigned_t<false>,
                                   BazIO::Transform<BazIO::DeltaCompression>,
                                   BazIO::Serialize<BazIO::CompactOverflow, BazIO::NumBits_t<7>>>,
                      BazIO::Field<BazIO::PacketFieldName::Pkmax,
                                   BazIO::StoreSigned_t<true>,
                                   BazIO::Transform<BazIO::FixedPoint, BazIO::FixedPointScale_t<10>>,
                                   BazIO::Serialize<BazIO::SimpleOverflow, BazIO::NumBits_t<8>, BazIO::NumBytes_t<2>>>,
                      BazIO::Field<BazIO::PacketFieldName::Pkmid,
                                   BazIO::StoreSigned_t<true>,
                                   BazIO::Transform<BazIO::FixedPoint, BazIO::FixedPointScale_t<10>>,
                                   BazIO::Serialize<BazIO::SimpleOverflow, BazIO::NumBits_t<8>, BazIO::NumBytes_t<2>>>,
                      BazIO::Field<BazIO::PacketFieldName::Pkmean,
                                   BazIO::StoreSigned_t<true>,
                                   BazIO::Transform<BazIO::FixedPoint, BazIO::FixedPointScale_t<10>>,
                                   BazIO::Serialize<BazIO::SimpleOverflow, BazIO::NumBits_t<8>, BazIO::NumBytes_t<2>>>,
                      BazIO::Field<BazIO::PacketFieldName::Pkvar,
                                   BazIO::StoreSigned_t<false>,
                                   BazIO::Transform<BazIO::FixedPoint, BazIO::FixedPointScale_t<10>>,
                                   BazIO::Serialize<BazIO::SimpleOverflow, BazIO::NumBits_t<7>, BazIO::NumBytes_t<2>>>,
                      BazIO::Field<BazIO::PacketFieldName::IsBase,
                                   BazIO::StoreSigned_t<false>,
                                   BazIO::Transform<BazIO::NoOp>,
                                   BazIO::Serialize<BazIO::TruncateOverflow, BazIO::NumBits_t<1>>>>;

}  // namespace PacBio::BazIO

#endif  // PACBIO_BAZIO_ENCODING_TEST_TESTING_PULSE_H
