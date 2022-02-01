// Copyright (c) 2014-2016, Pacific Biosciences of California, Inc.
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted (subject to the limitations in the
// disclaimer below) provided that the following conditions are met:
//
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
//  * Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
//  * Neither the name of Pacific Biosciences nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE
// GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY PACIFIC
// BIOSCIENCES AND ITS CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL PACIFIC BIOSCIENCES OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
// USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
// OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGE.

// Programmer: Armin Töpfer

#pragma once

#include <array>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <vector>
#include <array>

#include <half.hpp>

#include "BlockActivityLabels.h"
#include "FloatFixedCodec.h"
#include "MetricField.h"
#include "MetricFieldMap.h"

//#define SANITY_CHECK_METRICS 1

namespace PacBio {
namespace Primary {

template <typename Flt,size_t Nc>
struct MetricBlock
{
    static constexpr size_t NCam = Nc;
    static constexpr size_t NAngles = (NCam == 2) ? 2 : 0;

    uint32_t numFrames_                 = 0;
    uint32_t numPulses_                 = 0;
    uint32_t pulseWidth_                = 0;
    uint32_t baseWidth_                 = 0;
    uint32_t numSandwiches_             = 0;
    uint32_t numHalfSandwiches_         = 0;
    uint32_t numPulseLabelStutters_     = 0;
    Flt      pulseDetectionScore_       = 0;
    Flt      traceAutoCorr_             = 0;
    int16_t  pixelChecksum_             = 0;
    uint8_t  dmeStatus_                 = 0;
    uint8_t activityLabel_ = static_cast<uint8_t>(ActivityLabeler::A0);

    // Analog-dependent metrics
    Flt      pkmidA_                    = 0;
    Flt      pkmidC_                    = 0;
    Flt      pkmidG_                    = 0;
    Flt      pkmidT_                    = 0;
    uint32_t numpkmidFramesA_           = 0;
    uint32_t numpkmidFramesC_           = 0;
    uint32_t numpkmidFramesG_           = 0;
    uint32_t numpkmidFramesT_           = 0;
    Flt      pkmaxA_                    = 0;
    Flt      pkmaxC_                    = 0;
    Flt      pkmaxG_                    = 0;
    Flt      pkmaxT_                    = 0;
    uint32_t numBasesA_                 = 0;
    uint32_t numBasesC_                 = 0;
    uint32_t numBasesG_                 = 0;
    uint32_t numBasesT_                 = 0;
    // Not for baz output, but necessary to handle the bpz/pkz vars in the
    // Merge function
    uint32_t numPkmidBasesA_                 = 0;
    uint32_t numPkmidBasesC_                 = 0;
    uint32_t numPkmidBasesG_                 = 0;
    uint32_t numPkmidBasesT_                 = 0;
    // Interpulse squared coefficient of variance (variance/pkmid^2)
    Flt      bpzvarA_                   = 0;
    Flt      bpzvarC_                   = 0;
    Flt      bpzvarG_                   = 0;
    Flt      bpzvarT_                   = 0;
    // Intrapulse variance normalized by model variance
    Flt      pkzvarA_                   = 0;
    Flt      pkzvarC_                   = 0;
    Flt      pkzvarG_                   = 0;
    Flt      pkzvarT_                   = 0;

    // Spectral channel dependent metrics
    std::array<Flt,NCam>      baselines_;
    std::array<Flt,NCam>      baselineSds_;
    std::array<uint32_t,NCam> numBaselineFrames_;
    std::array<Flt,NAngles>   angles_;

public:
    MetricBlock()
    {
        baselines_.fill(0);
        baselineSds_.fill(0);
        numBaselineFrames_.fill(0);
        angles_.fill(0);
    }

    MetricBlock(std::initializer_list<MetricBlock> parents) : MetricBlock()
    {
        std::vector<MetricBlock> v;
        std::move(parents.begin(), parents.end(), std::back_inserter(v));
        // v.insert(v.end(), parents.begin(), parents.end());
        Merge(&v[0], v.size());
    }

    MetricBlock(const std::vector<MetricBlock>& parents) : MetricBlock()
    { Merge(parents.data(), parents.size()); }

    MetricBlock(const MetricBlock* parents, size_t numParents) : MetricBlock()
    { Merge(parents, numParents); }

    // Move constructor
    MetricBlock(MetricBlock&&) = default;
    // Copy constructor
    MetricBlock(const MetricBlock&) = default;
    // Move assignment operator
    MetricBlock& operator=(MetricBlock&&) = default;
    // Copy assignment operator
    MetricBlock& operator=(const MetricBlock&) = default;
    // Destructor
    ~MetricBlock() = default;

public: // setters

    MetricBlock& NumFrames(uint16_t numFrames)
    {
        numFrames_ = static_cast<uint32_t>(numFrames);
        return *this;
    }

    MetricBlock& NumPulses(uint16_t numPulses)
    {
        numPulses_ = static_cast<uint32_t>(numPulses);
        return *this;
    }

    MetricBlock& PulseWidth(uint16_t pulseWidth)
    {
        pulseWidth_ = static_cast<uint32_t>(pulseWidth);
        return *this;
    }

    MetricBlock& BaseWidth(uint16_t baseWidth)
    {
        baseWidth_ = static_cast<uint32_t>(baseWidth);
        return *this;
    }

    MetricBlock& NumSandwiches(uint16_t numSandwiches)
    {
        numSandwiches_ = static_cast<uint32_t>(numSandwiches);
        return *this;
    }

    MetricBlock& NumHalfSandwiches(uint16_t numHalfSandwiches)
    {
        numHalfSandwiches_ = static_cast<uint32_t>(numHalfSandwiches);
        return *this;
    }

    MetricBlock& NumPulseLabelStutters(uint16_t numPulseLabelStutters)
    {
        numPulseLabelStutters_ = static_cast<uint32_t>(numPulseLabelStutters);
        return *this;
    }

    MetricBlock& PulseDetectionScore(Flt pulseDetectionScore)
    {
        pulseDetectionScore_ = pulseDetectionScore;
        return *this;
    }

    MetricBlock& TraceAutocorr(Flt traceAutoCorr)
    {
        traceAutoCorr_ = traceAutoCorr;
        return *this;
    }

    MetricBlock& PixelChecksum(int16_t pixelChecksum)
    {
        pixelChecksum_ = pixelChecksum;
        return *this;
    }

    MetricBlock& DmeStatus(uint8_t dmeStatus)
    {
        dmeStatus_ = dmeStatus;
        return *this;
    }

    MetricBlock& ActivityLabel(uint8_t activityLabel)
    {
        activityLabel_ = activityLabel;
        return *this;
    }

    MetricBlock& PkmidA(Flt pkmidA)
    {
        pkmidA_ = pkmidA;
        return *this;
    }

    MetricBlock& PkmidC(Flt pkmidC)
    {
        pkmidC_ = pkmidC;
        return *this;
    }

    MetricBlock& PkmidG(Flt pkmidG)
    {
        pkmidG_ = pkmidG;
        return *this;
    }

    MetricBlock& PkmidT(Flt pkmidT)
    {
        pkmidT_ = pkmidT;
        return *this;
    }

    MetricBlock& NumPkmidFramesA(uint16_t numpkmidFramesA)
    {
        numpkmidFramesA_ = static_cast<uint32_t>(numpkmidFramesA);
        return *this;
    }

    MetricBlock& NumPkmidFramesG(uint16_t numpkmidFramesG)
    {
        numpkmidFramesG_ = static_cast<uint32_t>(numpkmidFramesG);
        return *this;
    }

    MetricBlock& NumPkmidFramesC(uint16_t numpkmidFramesC)
    {
        numpkmidFramesC_ = static_cast<uint32_t>(numpkmidFramesC);
        return *this;
    }

    MetricBlock& NumPkmidFramesT(uint16_t numpkmidFramesT)
    {
        numpkmidFramesT_ = static_cast<uint32_t>(numpkmidFramesT);
        return *this;
    }

    MetricBlock& PkmaxA(float pkmaxA)
    {
        pkmaxA_ = pkmaxA;
        return *this;
    }

    MetricBlock& PkmaxC(float pkmaxC)
    {
        pkmaxC_ = pkmaxC;
        return *this;
    }

    MetricBlock& PkmaxG(float pkmaxG)
    {
        pkmaxG_ = pkmaxG;
        return *this;
    }

    MetricBlock& PkmaxT(float pkmaxT)
    {
        pkmaxT_ = pkmaxT;
        return *this;
    }

    MetricBlock& NumBasesA(uint16_t numBasesA)
    {
        numBasesA_ = static_cast<uint32_t>(numBasesA);
        return *this;
    }

    MetricBlock& NumBasesC(uint16_t numBasesC)
    {
        numBasesC_ = static_cast<uint32_t>(numBasesC);
        return *this;
    }

    MetricBlock& NumBasesG(uint16_t numBasesG)
    {
        numBasesG_ = static_cast<uint32_t>(numBasesG);
        return *this;
    }

    MetricBlock& NumBasesT(uint16_t numBasesT)
    {
        numBasesT_ = static_cast<uint32_t>(numBasesT);
        return *this;
    }

    MetricBlock& NumPkmidBasesA(uint16_t numPkmidBasesA)
    {
        numPkmidBasesA_ = static_cast<uint32_t>(numPkmidBasesA);
        return *this;
    }

    MetricBlock& NumPkmidBasesC(uint16_t numPkmidBasesC)
    {
        numPkmidBasesC_ = static_cast<uint32_t>(numPkmidBasesC);
        return *this;
    }

    MetricBlock& NumPkmidBasesG(uint16_t numPkmidBasesG)
    {
        numPkmidBasesG_ = static_cast<uint32_t>(numPkmidBasesG);
        return *this;
    }

    MetricBlock& NumPkmidBasesT(uint16_t numPkmidBasesT)
    {
        numPkmidBasesT_ = static_cast<uint32_t>(numPkmidBasesT);
        return *this;
    }

    MetricBlock& BpzvarA(Flt bpzvarA)
    {
        bpzvarA_ = bpzvarA;
        return *this;
    }

    MetricBlock& BpzvarC(Flt bpzvarC)
    {
        bpzvarC_ = bpzvarC;
        return *this;
    }

    MetricBlock& BpzvarG(Flt bpzvarG)
    {
        bpzvarG_ = bpzvarG;
        return *this;
    }

    MetricBlock& BpzvarT(Flt bpzvarT)
    {
        bpzvarT_ = bpzvarT;
        return *this;
    }

    MetricBlock& PkzvarA(Flt pkzvarA)
    {
        pkzvarA_ = pkzvarA;
        return *this;
    }

    MetricBlock& PkzvarC(Flt pkzvarC)
    {
        pkzvarC_ = pkzvarC;
        return *this;
    }

    MetricBlock& PkzvarG(Flt pkzvarG)
    {
        pkzvarG_ = pkzvarG;
        return *this;
    }

    MetricBlock& PkzvarT(Flt pkzvarT)
    {
        pkzvarT_ = pkzvarT;
        return *this;
    }

    MetricBlock& Baselines(const std::array<Flt,NCam>& baselines)
    {
        baselines_ = baselines;
        return *this;
    }

    MetricBlock& BaselineSds(const std::array<Flt,NCam>& baselineSds)
    {
        baselineSds_ = baselineSds;
        return *this;
    }

    MetricBlock& NumBaselineFrames(const std::array<uint32_t,NCam>& numBaselineFrames)
    {
        numBaselineFrames_ = numBaselineFrames;
        return *this;
    }

    MetricBlock& Angles(const std::array<Flt,NAngles>& angles)
    {
        angles_ = angles;
        return *this;
    }

public: // getters

    uint32_t NumFrames() const
    { return numFrames_; }

    uint32_t NumPulses() const
    { return numPulses_; }

    uint32_t PulseWidth() const
    { return pulseWidth_; }

    uint32_t BaseWidth() const
    { return baseWidth_; }

    uint32_t NumSandwiches() const
    { return numSandwiches_; }

    uint32_t NumHalfSandwiches() const
    { return numHalfSandwiches_; }

    uint32_t NumPulseLabelStutters() const
    { return numPulseLabelStutters_; }

    float PulseDetectionScore() const
    { return pulseDetectionScore_; }

    float TraceAutocorr() const
    { return traceAutoCorr_; }

    int16_t PixelChecksum() const
    { return pixelChecksum_; }

    uint8_t DmeStatus() const
    { return dmeStatus_; }

    ActivityLabeler::Activity ActivityLabel() const
    { return static_cast<ActivityLabeler::Activity>(activityLabel_); }

    Flt PkmidA() const
    { return pkmidA_; }

    Flt PkmidC() const
    { return pkmidC_; }

    Flt PkmidG() const
    { return pkmidG_; }

    Flt PkmidT() const
    { return pkmidT_; }

    uint32_t NumPkmidFramesA() const
    { return numpkmidFramesA_; }

    uint32_t NumPkmidFramesG() const
    { return numpkmidFramesG_; }

    uint32_t NumPkmidFramesC() const
    { return numpkmidFramesC_; }

    uint32_t NumPkmidFramesT() const
    { return numpkmidFramesT_; }

    Flt PkmaxA() const
    { return pkmaxA_; }

    Flt PkmaxC() const
    { return pkmaxC_; }

    Flt PkmaxG() const
    { return pkmaxG_; }

    Flt PkmaxT() const
    { return pkmaxT_; }

    uint32_t NumBasesA() const
    { return numBasesA_; }

    uint32_t NumBasesC() const
    { return numBasesC_; }

    uint32_t NumBasesG() const
    { return numBasesG_; }

    uint32_t NumBasesT() const
    { return numBasesT_; }

    uint32_t NumPkmidBasesA() const
    { return numPkmidBasesA_; }

    uint32_t NumPkmidBasesC() const
    { return numPkmidBasesC_; }

    uint32_t NumPkmidBasesG() const
    { return numPkmidBasesG_; }

    uint32_t NumPkmidBasesT() const
    { return numPkmidBasesT_; }

    Flt BpzvarA() const
    { return bpzvarA_; }

    Flt BpzvarC() const
    { return bpzvarC_; }

    Flt BpzvarG() const
    { return bpzvarG_; }

    Flt BpzvarT() const
    { return bpzvarT_; }

    Flt PkzvarA() const
    { return pkzvarA_; }

    Flt PkzvarC() const
    { return pkzvarC_; }

    Flt PkzvarG() const
    { return pkzvarG_; }

    Flt PkzvarT() const
    { return pkzvarT_; }

    const std::array<Flt,NCam> Baselines() const
    { return baselines_; }

    const std::array<Flt,NCam> BaselineSds() const
    { return baselineSds_; }

    const std::array<Flt,NAngles> Angles() const
    { return angles_; }

public:
    void AppendToBaz(const std::vector<MetricField>& metricFields,
                     std::vector<uint8_t>& buffer, size_t& c) const
    {
        // Iterate over all user-defined metric fields
        for (const auto& metricField : metricFields)
        {
            // Store correct field given name of the metric field.
            // For each case, calculate field and cast to correct type.
            switch(metricField.fieldName)
            {
                case MetricFieldName::NUM_FRAMES:
                    StoreToBaz<uint16_t>(static_cast<uint16_t>(numFrames_), buffer, c);
                    break;
                case MetricFieldName::NUM_PULSES:
                    StoreToBaz<uint16_t>(static_cast<uint16_t>(numPulses_), buffer, c);
                    break;
                case MetricFieldName::PULSE_WIDTH:
                    StoreToBaz<uint16_t>(static_cast<uint16_t>(pulseWidth_), buffer, c);
                    break;
                case MetricFieldName::BASE_WIDTH:
                    StoreToBaz<uint16_t>(static_cast<uint16_t>(baseWidth_), buffer, c);
                    break;
                case MetricFieldName::NUM_SANDWICHES:
                    StoreToBaz<uint8_t>(static_cast<uint16_t>(numSandwiches_), buffer, c);
                    break;
                case MetricFieldName::NUM_HALF_SANDWICHES:
                    StoreToBaz<uint8_t>(static_cast<uint16_t>(numHalfSandwiches_), buffer, c);
                    break;
                case MetricFieldName::NUM_PULSE_LABEL_STUTTERS:
                    StoreToBaz<uint8_t>(static_cast<uint16_t>(numPulseLabelStutters_), buffer, c);
                    break;
                case MetricFieldName::BASELINE_RED_SD:
                    StoreToBaz<int16_t>(FloatToBaz(baselineSds_[1], metricField.fieldScalingFactor), buffer, c);
                    break;
                case MetricFieldName::BASELINE_GREEN_SD:
                    StoreToBaz<int16_t>(FloatToBaz(baselineSds_[0], metricField.fieldScalingFactor), buffer, c);
                    break;
                case MetricFieldName::BASELINE_RED_MEAN:
                    StoreToBaz<int16_t>(FloatToBaz(baselines_[1], metricField.fieldScalingFactor), buffer, c);
                    break;
                case MetricFieldName::BASELINE_GREEN_MEAN:
                    StoreToBaz<int16_t>(FloatToBaz(baselines_[0], metricField.fieldScalingFactor), buffer, c);
                    break;
                case MetricFieldName::BASELINE_SD:
                    StoreToBaz<int16_t>(FloatToBaz(baselineSds_[0], metricField.fieldScalingFactor), buffer, c);
                    break;
                case MetricFieldName::BASELINE_MEAN:
                    StoreToBaz<int16_t>(FloatToBaz(baselines_[0], metricField.fieldScalingFactor), buffer, c);
                    break;
                case MetricFieldName::PKMID_A:
                    StoreToBaz<int16_t>(FloatToBaz(pkmidA_, metricField.fieldScalingFactor), buffer, c);
                    break;
                case MetricFieldName::PKMID_C:
                    StoreToBaz<int16_t>(FloatToBaz(pkmidC_, metricField.fieldScalingFactor), buffer, c);
                    break;
                case MetricFieldName::PKMID_G:
                    StoreToBaz<int16_t>(FloatToBaz(pkmidG_, metricField.fieldScalingFactor), buffer, c);
                    break;
                case MetricFieldName::PKMID_T:
                    StoreToBaz<int16_t>(FloatToBaz(pkmidT_, metricField.fieldScalingFactor), buffer, c);
                    break;
                case MetricFieldName::BPZVAR_A:
                    StoreToBaz<int16_t>(FloatToBaz(bpzvarA_, metricField.fieldScalingFactor), buffer, c);
                    break;
                case MetricFieldName::BPZVAR_C:
                    StoreToBaz<int16_t>(FloatToBaz(bpzvarC_, metricField.fieldScalingFactor), buffer, c);
                    break;
                case MetricFieldName::BPZVAR_G:
                    StoreToBaz<int16_t>(FloatToBaz(bpzvarG_, metricField.fieldScalingFactor), buffer, c);
                    break;
                case MetricFieldName::BPZVAR_T:
                    StoreToBaz<int16_t>(FloatToBaz(bpzvarT_, metricField.fieldScalingFactor), buffer, c);
                    break;
                case MetricFieldName::PKZVAR_A:
                    StoreToBaz<int16_t>(FloatToBaz(pkzvarA_, metricField.fieldScalingFactor), buffer, c);
                    break;
                case MetricFieldName::PKZVAR_C:
                    StoreToBaz<int16_t>(FloatToBaz(pkzvarC_, metricField.fieldScalingFactor), buffer, c);
                    break;
                case MetricFieldName::PKZVAR_G:
                    StoreToBaz<int16_t>(FloatToBaz(pkzvarG_, metricField.fieldScalingFactor), buffer, c);
                    break;
                case MetricFieldName::PKZVAR_T:
                    StoreToBaz<int16_t>(FloatToBaz(pkzvarT_, metricField.fieldScalingFactor), buffer, c);
                    break;
                case MetricFieldName::NUM_BASES_A:
                    StoreToBaz<uint16_t>(static_cast<uint16_t>(numBasesA_), buffer, c);
                    break;
                case MetricFieldName::NUM_BASES_C:
                    StoreToBaz<uint16_t>(static_cast<uint16_t>(numBasesC_), buffer, c);
                    break;
                case MetricFieldName::NUM_BASES_G:
                    StoreToBaz<uint16_t>(static_cast<uint16_t>(numBasesG_), buffer, c);
                    break;
                case MetricFieldName::NUM_BASES_T:
                    StoreToBaz<uint16_t>(static_cast<uint16_t>(numBasesT_), buffer, c);
                    break;
                case MetricFieldName::PKMID_FRAMES_A:
                    StoreToBaz<int16_t>(static_cast<uint16_t>(numpkmidFramesA_), buffer, c);
                    break;
                case MetricFieldName::PKMID_FRAMES_C:
                    StoreToBaz<int16_t>(static_cast<uint16_t>(numpkmidFramesC_), buffer, c);
                    break;
                case MetricFieldName::PKMID_FRAMES_G:
                    StoreToBaz<int16_t>(static_cast<uint16_t>(numpkmidFramesG_), buffer, c);
                    break;
                case MetricFieldName::PKMID_FRAMES_T:
                    StoreToBaz<int16_t>(static_cast<uint16_t>(numpkmidFramesT_), buffer, c);
                    break;
                case MetricFieldName::PIXEL_CHECKSUM:
                    StoreToBaz<int16_t>(pixelChecksum_, buffer, c);
                    break;
                case MetricFieldName::ANGLE_RED:
                    StoreToBaz<int16_t>(FloatToBaz(angles_[1], metricField.fieldScalingFactor), buffer, c);
                    break;
                case MetricFieldName::ANGLE_GREEN:
                    StoreToBaz<int16_t>(FloatToBaz(angles_[0], metricField.fieldScalingFactor), buffer, c);
                    break;
                case MetricFieldName::PKMAX_A:
                    StoreToBaz<int16_t>(FloatToBaz(pkmaxA_, metricField.fieldScalingFactor), buffer, c);
                    break;
                case MetricFieldName::PKMAX_C:
                    StoreToBaz<int16_t>(FloatToBaz(pkmaxC_, metricField.fieldScalingFactor), buffer, c);
                    break;
                case MetricFieldName::PKMAX_G:
                    StoreToBaz<int16_t>(FloatToBaz(pkmaxG_, metricField.fieldScalingFactor), buffer, c);
                    break;
                case MetricFieldName::PKMAX_T:
                    StoreToBaz<int16_t>(FloatToBaz(pkmaxT_, metricField.fieldScalingFactor), buffer, c);
                    break;
                case MetricFieldName::PULSE_DETECTION_SCORE:
                    StoreToBaz<int16_t>(FloatToBaz(pulseDetectionScore_, metricField.fieldScalingFactor), buffer, c);
                    break;
                case MetricFieldName::TRACE_AUTOCORR:
                    StoreToBaz<int16_t>(FloatToBaz(traceAutoCorr_, metricField.fieldScalingFactor), buffer, c);
                    break;
                case MetricFieldName::DME_STATUS:
                    StoreToBaz<uint8_t>(dmeStatus_, buffer, c);
                    break;
                case MetricFieldName::ACTIVITY_LABEL:
                    StoreToBaz<uint8_t>(static_cast<uint8_t>(activityLabel_), buffer, c);
                    break;
                case MetricFieldName::GAP:
                    throw std::runtime_error("GAP not allowed in metrics definition for writing");
                default:
                    throw std::runtime_error("Case not defined " + metricField.fieldName.toString());
            }
        }
    }

private:
    // Scale factor = 0 indicates use half float which is the default.
    static constexpr float defaultScalingFactor = 0;

private:
    int16_t FloatToBaz(float value, float scalingFactor=defaultScalingFactor) const
    {
        if (scalingFactor == 0)
        {
            auto v = half_float::half(value);
            int16_t ret;
            memcpy(&ret, &v, sizeof(v));
            return ret;
        }
        else
        {
            FloatFixedCodec<float,int16_t> codec(1/scalingFactor);
            return codec.Encode(value);
        }
    }

    template<typename T>
    inline void StoreToBaz(const T& tmp, std::vector<uint8_t>& buffer, size_t& c) const
    {
        memcpy(&buffer[c], &tmp, sizeof(T));
        c += sizeof(tmp);
    }

    void Merge(const MetricBlock<Flt,NCam>* parents, size_t numParents)
    {
        if (numParents == 0) return;

        std::array<std::vector<Flt>,NAngles> angles;

        std::array<Flt,4> pkMax = {{
            std::numeric_limits<Flt>::lowest(),
            std::numeric_limits<Flt>::lowest(),
            std::numeric_limits<Flt>::lowest(),
            std::numeric_limits<Flt>::lowest()
        }};

        std::array<Flt,2> frameCorr;
        frameCorr.fill(0);

        // Iterate over parents and sum
        for (size_t j = 0; j < numParents; ++j)
        {
            const auto& p = parents[j];
            numFrames_         += p.numFrames_;
            numPulses_         += p.numPulses_;

            activityLabel_    = std::max(activityLabel_, p.activityLabel_);

            numBasesA_         += p.numBasesA_;
            numBasesC_         += p.numBasesC_;
            numBasesG_         += p.numBasesG_;
            numBasesT_         += p.numBasesT_;

            numPkmidBasesA_         += p.numPkmidBasesA_;
            numPkmidBasesC_         += p.numPkmidBasesC_;
            numPkmidBasesG_         += p.numPkmidBasesG_;
            numPkmidBasesT_         += p.numPkmidBasesT_;

            numpkmidFramesA_   += p.numpkmidFramesA_;
            numpkmidFramesC_   += p.numpkmidFramesC_;
            numpkmidFramesG_   += p.numpkmidFramesG_;
            numpkmidFramesT_   += p.numpkmidFramesT_;

            for (size_t c = 0; c < NCam; c++)
            {
                baselines_[c]           += p.numBaselineFrames_[c] * p.baselines_[c];
                numBaselineFrames_[c]   += p.numBaselineFrames_[c];

                // Compute pooled variance
                size_t n = (p.numBaselineFrames_[c] > 1) ? p.numBaselineFrames_[c] - 1 : p.numBaselineFrames_[c];
                baselineSds_[c]     += n * (p.baselineSds_[c] * p.baselineSds_[c] );

                if (p.numBaselineFrames_[c] > 0) frameCorr[c]++;
            }

            // Only save angles that are not NaN.
            auto ax = angles.begin();
            for (const auto& a : p.angles_)
            {
                if (!std::isnan(a)) (*ax).push_back(a);
                ax++;
            }

            traceAutoCorr_      += p.numFrames_ * p.traceAutoCorr_;

            // PKMIDs are weighted by number of specific frames
            pkmidA_            += p.pkmidA_ * p.numpkmidFramesA_;
            pkmidC_            += p.pkmidC_ * p.numpkmidFramesC_;
            pkmidG_            += p.pkmidG_ * p.numpkmidFramesG_;
            pkmidT_            += p.pkmidT_ * p.numpkmidFramesT_;

            // The merge below of bpzvar and pkvar doesn't check
            // for NaNs. Previously, NaNs would cause the merged
            // bpzvar and pkzvar be set to NaN and using the
            // fixed-point scaling, 0 would be stored to the BAZ file.
            // With half-float support, NaNs are now properly
            // stored to the BAZ file.
            if (p.numPkmidBasesA_ > 1)
                bpzvarA_      += p.bpzvarA_ * p.pkmidA_ * p.pkmidA_ * p.numPkmidBasesA_;
            if (p.numPkmidBasesC_ > 1)
                bpzvarC_      += p.bpzvarC_ * p.pkmidC_ * p.pkmidC_ * p.numPkmidBasesC_;
            if (p.numPkmidBasesG_ > 1)
                bpzvarG_      += p.bpzvarG_ * p.pkmidG_ * p.pkmidG_ * p.numPkmidBasesG_;
            if (p.numPkmidBasesT_ > 1)
                bpzvarT_      += p.bpzvarT_ * p.pkmidT_ * p.pkmidT_ * p.numPkmidBasesT_;

            if (p.numpkmidFramesA_ > 1)
                pkzvarA_     += p.pkzvarA_ * p.numpkmidFramesA_;
            if (p.numpkmidFramesC_ > 1)
                pkzvarC_     += p.pkzvarC_ * p.numpkmidFramesC_;
            if (p.numpkmidFramesG_ > 1)
                pkzvarG_     += p.pkzvarG_ * p.numpkmidFramesG_;
            if (p.numpkmidFramesT_ > 1)
                pkzvarT_     += p.pkzvarT_ * p.numpkmidFramesT_;

            // Pulse width accumulation, avoid overflows
            pulseWidth_        += p.pulseWidth_;
            // pulseWidth_        |= -(pulseWidth_ < p.pulseWidth_);

            // Base width accumulation, avoid overflows
            baseWidth_         += p.baseWidth_;
            baseWidth_         |= -(baseWidth_ < p.baseWidth_);

            // More additive counts
            numSandwiches_         += p.numSandwiches_;
            numHalfSandwiches_     += p.numHalfSandwiches_;
            numPulseLabelStutters_ += p.numPulseLabelStutters_;

            pixelChecksum_         += p.pixelChecksum_;

            pkMax[0] = std::max(pkMax[0], p.pkmaxA_);
            pkMax[1] = std::max(pkMax[1], p.pkmaxC_);
            pkMax[2] = std::max(pkMax[2], p.pkmaxG_);
            pkMax[3] = std::max(pkMax[3], p.pkmaxT_);

            pulseDetectionScore_ += p.numFrames_ * p.pulseDetectionScore_;
        }

        for (size_t c = 0; c < NCam; c++)
        {
            // Divide by frames.
            if (numBaselineFrames_[c] > 0)
            {
                baselines_[c]   /= numBaselineFrames_[c];
                baselineSds_[c] /= (numBaselineFrames_[c] - frameCorr[c]) > 0
                                   ? (numBaselineFrames_[c] - frameCorr[c])
                                   : numBaselineFrames_[c];
            }

            // Take sqrt to get from variance to SD
            baselineSds_[c] = std::sqrt(baselineSds_[c]);
        }

        std::transform(angles.begin(), angles.end(), angles_.begin(),
        [](std::vector<Flt>& a)
        {
           // Take median of angle estimates.
           if (a.empty())
               return std::numeric_limits<Flt>::quiet_NaN();
           else
           {
               std::nth_element(a.begin(), a.begin() + a.size()/2, a.end());
               return a[a.size()/2];
           }
        });

        traceAutoCorr_ /= numFrames_;

        // Divide by weighting
        if (numpkmidFramesA_ > 0) pkmidA_ /= numpkmidFramesA_;
        if (numpkmidFramesC_ > 0) pkmidC_ /= numpkmidFramesC_;
        if (numpkmidFramesG_ > 0) pkmidG_ /= numpkmidFramesG_;
        if (numpkmidFramesT_ > 0) pkmidT_ /= numpkmidFramesT_;

        if (numpkmidFramesA_ > 0 && numPkmidBasesA_ > 1) bpzvarA_ /= numPkmidBasesA_ * pkmidA_ * pkmidA_;
        if (numpkmidFramesC_ > 0 && numPkmidBasesC_ > 1) bpzvarC_ /= numPkmidBasesC_ * pkmidC_ * pkmidC_;
        if (numpkmidFramesG_ > 0 && numPkmidBasesG_ > 1) bpzvarG_ /= numPkmidBasesG_ * pkmidG_ * pkmidG_;
        if (numpkmidFramesT_ > 0 && numPkmidBasesT_ > 1) bpzvarT_ /= numPkmidBasesT_ * pkmidT_ * pkmidT_;

        if (numpkmidFramesA_ > 1) pkzvarA_ /= numpkmidFramesA_;
        if (numpkmidFramesC_ > 1) pkzvarC_ /= numpkmidFramesC_;
        if (numpkmidFramesG_ > 1) pkzvarG_ /= numpkmidFramesG_;
        if (numpkmidFramesT_ > 1) pkzvarT_ /= numpkmidFramesT_;

        pkmaxA_ = pkMax[0];
        pkmaxC_ = pkMax[1];
        pkmaxG_ = pkMax[2];
        pkmaxT_ = pkMax[3];

        if (numFrames_ != 0)
            pulseDetectionScore_ /= numFrames_;
        else
            pulseDetectionScore_ = 0;

    }
};

using SequelMetricBlock = MetricBlock<float,2>;
using SpiderMetricBlock = MetricBlock<float,1>;    

}}
