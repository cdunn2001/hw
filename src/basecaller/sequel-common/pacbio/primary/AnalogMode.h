#ifndef Sequel_Common_AnalogMode_H_
#define Sequel_Common_AnalogMode_H_

// Copyright (c) 2015, Pacific Biosciences of California, Inc.
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
/// \file   AnalogMode.h
/// \brief  The class AnalogMode represents the relevant properties of
/// sequencing analog molecules relative to the Sequel or Spider
/// detection system.

#include <array>
#include <cassert>
#include <cmath>
#include <ostream>
#include <vector>
#include <algorithm>
#include <cstdint>

#include <pacbio/PBException.h>
#include <pacbio/process/ConfigurationBase.h>

namespace PacBio {
namespace Primary {

// internal JSON structure (deprecated)
class AnalogConfig : public PacBio::Process::ConfigurationObject
{
    CONF_OBJ_SUPPORT_COPY(AnalogConfig)
public:
    // Defaults will be handled by the ctor of AnalogSetConfig
    ADD_PARAMETER(float, red, 0.0f);
    ADD_PARAMETER(float, green, 0.0f);
    ADD_PARAMETER(float, relAmp, 0.0f);
    ADD_PARAMETER(float, excessNoise, 0.0f);
    ADD_PARAMETER(float, meanIpdSec, 0.0f);
    ADD_PARAMETER(float, meanPulseWidthSec, 0.0f);
    ADD_PARAMETER(float, pw2SlowStepRatio, 0.0f);
    ADD_PARAMETER(float, ipd2SlowStepRatio, 0.0f);
};


//external JSON structure
// This is overconstrained.  Either `spectralAngle` or `spectrumValues[]` can be specified but not both.
// The code will automatically convert one format to the other format.  For future convenience, it is suggested
// that spectralAngle be deprecated, because it only is useful in one scheme (2C2A) while spectrumValues is
// more general purpose.
class AnalogConfigEx : public PacBio::Process::ConfigurationObject
{
    ADD_PARAMETER(std::string, base, "X"); // or baseMap
    //ADD_PARAMETER(float, spectralAngle, 0.0f); - these have been removed for special processing
    //ADD_ARRAY(float, spectrumValues); - this too
    ADD_PARAMETER(float, wavelength, 0.0f);
    ADD_PARAMETER(float, relativeAmplitude, 1.0f);
    ADD_PARAMETER(float, intraPulseXsnCV, 0.0f);
    ADD_PARAMETER(float, ipdMeanSeconds, 0.0f);
    ADD_PARAMETER(float, pulseWidthMeanSeconds, 0.0f);
    ADD_PARAMETER(float, pw2SlowStepRatio, 0.0f);
    ADD_PARAMETER(float, ipd2SlowStepRatio, 0.0f);
public:
    double spectralAngle() const {  return spectralAngle_;  }
    const std::vector<double>& spectrumValues() const { return spectrumValues_; }
private:
    double spectralAngle_;
    std::vector<double> spectrumValues_;
public:
    void Normalize()
    {
        double green = 0;
        double red = 0;
        int numChannels = 0;
        if (Json().isMember("spectralAngle"))
        {
            spectralAngle_ = Json()["spectralAngle"].asDouble();
            green = cos(spectralAngle_);
            red = sin(spectralAngle_);
            numChannels = 2;
            spectrumValues_.resize(numChannels);
            spectrumValues_[0] = green;
            spectrumValues_[1] = red;
        }
        else if (Json().isMember("spectrumValues") && Json()["spectrumValues"].size() == 2)
        {
            numChannels = 2;
            spectrumValues_.resize(numChannels);
            spectrumValues_[0] = Json()["spectrumValues"][0].asDouble();
            spectrumValues_[1] = Json()["spectrumValues"][1].asDouble();
            green = spectrumValues_[0];
            red = spectrumValues_[1];
            spectralAngle_ = atan2(red, green);
        }
        else if (Json().isMember("spectrumValues") &&  Json()["spectrumValues"].size() == 1)
        {
            numChannels = 1;
            spectrumValues_.resize(numChannels);
            spectrumValues_[0] = Json()["spectrumValues"][0].asDouble();
            green = spectrumValues_[0];
            red = green;
            spectralAngle_ = 0;
        }
        else
        {
            throw PBException("spectralAngle and spectrumValues[2] not given");
        }

        if (numChannels == 2)
        {
            double sum = green + red;
            assert(sum != 0);
            green /= sum;
            red /= sum;

            spectrumValues_[0] = green;
            spectrumValues_[1] = red;
            Json()["spectrumValues"][0] = spectrumValues_[0];
            Json()["spectrumValues"][1] = spectrumValues_[1];
        }
        else if (numChannels == 1)
        {
            spectrumValues_[0] = 1.0;
            Json()["spectrumValues"][0] = spectrumValues_[0];
        }
    }
    void PostImport() override
    {
      Normalize();
    }
};

/// Input or calibration properties describing an analog detection mode.
/// Spectral data uses the conventional green --> red filter order, 
/// regardless of the chip or platform internal ordering. The spectrum
/// normalization convention here is arbitrary.  The amplitude normalization
/// convention is arbitrary but must be consistent across different analogs.
/// For the Reference Analog, the amplitude value should be set to its
/// SNR estimate on the Ideal Sequel Instrument and Chip (ISIC).
///
class AnalogMode
{
public: // Structors

    AnalogMode(uint16_t numFilters0)
        : numFilters(numFilters0)
        , baseLabel(' ')
        , relAmplitude(0.0f)
        , excessNoiseCV(0.0f)
        , interPulseDistance(0.0f)
        , pulseWidth(0.0f)
        , pw2SlowStepRatio(0.0f)
        , ipd2SlowStepRatio(0.0f)
    {
        assert (0 < numFilters0);
        assert (numFilters0 < 3);
        dyeSpectrum.resize(numFilters, 0.0f);
    }

    AnalogMode(const AnalogMode& a) = default;

    AnalogMode& operator=(const AnalogMode& a) = default;

    AnalogMode(const AnalogConfigEx& config)
            :
            numFilters(config.spectrumValues().size())
            , baseLabel(config.base()[0])
            , relAmplitude(config.relativeAmplitude())
            , excessNoiseCV(config.intraPulseXsnCV())
            , interPulseDistance(config.ipdMeanSeconds())
            , pulseWidth(config.pulseWidthMeanSeconds())
            , pw2SlowStepRatio(config.pw2SlowStepRatio())
            , ipd2SlowStepRatio(config.ipd2SlowStepRatio())
    {
        dyeSpectrum.clear();
        for(auto x : config.spectrumValues())
        {
            dyeSpectrum.push_back(x);
        }
    }

    /// support label,[spectrum],relAmplitude constructor, xsNoiseCV=0, ipd=0 and pw=0
    /// In general, I am not a fan of these long anonymous argument lists for constructors because
    /// they are fragile if you accidentally slip one argument. While it is verbose, I recommend constructing
    /// the object with some default values, then overwriting each member by name.
    template<size_t NCam>
    AnalogMode(char label,
               const std::array<float, NCam>& spectrum,
               float amplitude, float xsNoiseCV = 0.0f,
               float ipdSeconds = 0.0f, float pwSeconds = 0.0f)
            : numFilters(NCam)
            , baseLabel(label)
            , dyeSpectrum(spectrum.begin(), spectrum.end())
            , relAmplitude(amplitude)
            , excessNoiseCV(xsNoiseCV)
            , interPulseDistance(ipdSeconds)
            , pulseWidth(pwSeconds)
            , pw2SlowStepRatio(0.0)
            , ipd2SlowStepRatio(0.0)
    { }

    /// Constructor without spectrum argument assumes single camera
    /// filter (NCam = 1).
    AnalogMode(char label, float amplitude, float xsNoiseCV = 0.0f)
            : AnalogMode(label, std::array<float, 1>{{1.0f}}, amplitude,
                         xsNoiseCV)
    { }

#if 0
    /// Constructor without spectrum argument assumes single camera
    /// filter (NCam = 1).
    AnalogMode(char label, float amplitude, float xsNoiseCV = 0.0f,
               float ipdSeconds = 0.0f, float pwSeconds = 0.0f)
        : AnalogMode(label, std::array<float, 1>{{1.0f}}, amplitude,
                     xsNoiseCV, ipdSeconds, pwSeconds)
    { }
#endif

public:
    unsigned short numFilters;
    char baseLabel;
    std::vector<float> dyeSpectrum;    // Always green --> red ordering here.
    float relAmplitude;                     // Relative amplitude
    float excessNoiseCV;
    float interPulseDistance;               // seconds
    float pulseWidth;                       // seconds
    float pw2SlowStepRatio;
    float ipd2SlowStepRatio;

public:
    float RelativeAmplitude() const
    { return relAmplitude; }

    double SpectralAngle() const
    {
        if (numFilters == 2)
        {
            return atan2(dyeSpectrum[1 /*red*/],dyeSpectrum[0 /*green*/]);
        }
        else if (numFilters == 1) return 0.0;
        else throw PBException("Not supported");
    }

    /// Glue code for legacy calls
    std::array<float, 2> DyeSpectrumAsArray() const
    {
        std::array<float,2> x;
        x[0] = dyeSpectrum[0];
        x[1] = dyeSpectrum[1];
        return x;
    }
};



inline std::ostream& operator<<(std::ostream& os, const AnalogMode& am)
{
    static const std::array<char,4> channelLabel = {'G','R','3','4'};

    os << "[" << am.baseLabel
            << ", RA " << am.relAmplitude;
    for(size_t i=0;i<am.numFilters;i++)
    {
        os << ", "<< channelLabel[i] << " " << am.dyeSpectrum[i];
    }

    os      << ", exnCV " << am.excessNoiseCV
            << ", ipd " << am.interPulseDistance
            << ", pw " << am.pulseWidth
            << ", ipd2ssr " << am.ipd2SlowStepRatio
            << ", pw2ssr " << am.pw2SlowStepRatio
            << "]";
    return os;
}


using AnalogSet = std::vector<AnalogMode>;

/// \param json - JSON array of AnalogConfigEx objects, in T, G, C, A order
inline AnalogSet ParseAnalogSet(const Json::Value& json)
{
    AnalogSet analogs;
    for (auto j : json)
    {
        AnalogConfigEx conf;
        conf.Load(j);
        analogs.push_back(AnalogMode(conf));
    }
    return analogs;
}


 inline void Sort(AnalogSet& analogs)
 {
     std::sort(analogs.begin(), analogs.end(), [](const AnalogMode&a , const AnalogMode& b)
     {
         if (a.SpectralAngle() < b.SpectralAngle()) return true;
         if (a.SpectralAngle() == b.SpectralAngle())
         {
             return a.RelativeAmplitude() > b.RelativeAmplitude();
         }
         else
         {
             return false;
         }
     });
 }
}} // ::PacBio::Primary

#endif // Sequel_Common_AnalogMode_H_
