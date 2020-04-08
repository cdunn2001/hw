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
#include <numeric>
#include <ostream>
#include <vector>
#include <algorithm>
#include <cstdint>

#include <pacbio/configuration/PBConfig.h>
#include <pacbio/configuration/Validation.h>
#include <pacbio/PBException.h>

namespace PacBio {
namespace Primary {

//external JSON structure
// This is overconstrained.  Either `spectralAngle` or `spectrumValues[]` can be specified but not both.
// In either case the raw input will be inaccessible, and one should access the public normalizedSpectrumValues
// that defaults to the normalized version of either input.
//
// For future convenience, it is suggested
// that spectralAngle be deprecated, because it only is useful in one scheme (2C2A) while spectrumValues is
// more general purpose.
class AnalogConfigEx : public Configuration::PBConfig<AnalogConfigEx>
{
    PB_CONFIG(AnalogConfigEx);

    PB_CONFIG_PARAM(std::string, base, "X"); // or baseMap
private:
    PB_CONFIG_PARAM(float, spectralAngle, -1.0f);  // Negative values reserved to mean "unset"
    PB_CONFIG_PARAM(std::vector<float>, spectrumValues, std::vector<float>{});
public:
    PB_CONFIG_PARAM(float, wavelength, 0.0f);
    PB_CONFIG_PARAM(float, relativeAmplitude, 1.0f);
    PB_CONFIG_PARAM(float, intraPulseXsnCV, 0.0f);
    PB_CONFIG_PARAM(float, ipdMeanSeconds, 0.0f);
    PB_CONFIG_PARAM(float, pulseWidthMeanSeconds, 0.0f);
    PB_CONFIG_PARAM(float, pw2SlowStepRatio, 0.0f);
    PB_CONFIG_PARAM(float, ipd2SlowStepRatio, 0.0f);

    PB_CONFIG_PARAM(std::vector<float>, normalizedSpectrumValues,
                    Configuration::DefaultFunc(Normalize, {"spectralAngle","spectrumValues"}));

private:
    static std::vector<float> Normalize(float specAngle, const std::vector<float>& specValues)
    {
        static constexpr size_t green = 0;
        static constexpr size_t red = 1;
        std::vector<float> normalSpectrum;
        const size_t sum = std::accumulate(specValues.begin(), specValues.end(), 0.0f);
        if (specAngle >= 0)
        {
            normalSpectrum.resize(2);
            normalSpectrum[green] = cos(specAngle);
            normalSpectrum[red] = sin(specAngle);
        }
        else if (specValues.size() == 2)
        {
            normalSpectrum = specValues;
            assert(sum != 0);
            normalSpectrum[0] /= sum;
            normalSpectrum[2] /= sum;
        }
        else if (specValues.size() == 1)
        {
            normalSpectrum.resize(1);
            normalSpectrum[0] = 1.0f;
        }
        return normalSpectrum;
    }
};

}}

namespace PacBio {
namespace Configuration {

template <typename T>
void ValidateConfig(const Primary::AnalogConfigEx& conf, ValidationResults* results)
{
    if(conf.spectralAngle < 0 && conf.spectrumValues.size() == 0)
        results->AddError("Neither of spectralAngle nor spectumValues are specified");
    if(conf.normalizedSpectrumValues.size() == 0 || conf.normalizedSpectrumValues.size() > 2)
        results->AddError("Incorrect number of spectrum values");
    const auto sum = std::accumulate(conf.normalizedSpectrumValues.begin(),
                                     conf.normalizedSpectrumValues.end(),
                                     0.0f);
    if (std::abs(1.0f - sum) > 1e-2f)
        results->AddError("normalizedSpectrum Values is not normalized");
}

}}

namespace PacBio {
namespace Primary {

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
            numFilters(config.normalizedSpectrumValues.size())
            , baseLabel(config.base[0])
            , relAmplitude(config.relativeAmplitude)
            , excessNoiseCV(config.intraPulseXsnCV)
            , interPulseDistance(config.ipdMeanSeconds)
            , pulseWidth(config.pulseWidthMeanSeconds)
            , pw2SlowStepRatio(config.pw2SlowStepRatio)
            , ipd2SlowStepRatio(config.ipd2SlowStepRatio)
    {
        dyeSpectrum.clear();
        for(auto x : config.normalizedSpectrumValues)
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
        AnalogConfigEx conf(j);
        auto validation = conf.Validate();
        if (validation.ErrorCount() > 0)
        {
            validation.PrintErrors();
            throw PBException("Error validating analog json");
        }
        analogs.push_back(conf);
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
