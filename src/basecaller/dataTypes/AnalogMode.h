#ifndef mongo_dataTypes_AnalogMode_H_
#define mongo_dataTypes_AnalogMode_H_

// Copyright (c) 2015-2019, Pacific Biosciences of California, Inc.
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
/// sequencing analog molecules relative to the sensor.

#include <array>
#include <cassert>
#include <cmath>
#include <ostream>
#include <algorithm>
#include <cstdint>

#include <pacbio/PBException.h>
#include <pacbio/process/ConfigurationBase.h>

#include <common/MongoConstants.h>

namespace PacBio {
namespace Mongo {
namespace Data {

class AnalogConfigEx : public PacBio::Process::ConfigurationObject
{
    ADD_PARAMETER(std::string, base, "X");
    ADD_PARAMETER(float, wavelength, 0.0f);
    ADD_PARAMETER(float, relativeAmplitude, 1.0f);
    ADD_PARAMETER(float, intraPulseXsnCV, 0.0f);
    ADD_PARAMETER(float, ipdMeanSeconds, 0.0f);
    ADD_PARAMETER(float, pulseWidthMeanSeconds, 0.0f);
    ADD_PARAMETER(float, pw2SlowStepRatio, 0.0f);
    ADD_PARAMETER(float, ipd2SlowStepRatio, 0.0f);
};


/// Input or calibration properties describing an analog detection mode.
class AnalogMode
{
public: // Structors

    AnalogMode()
        : baseLabel(' ')
        , relAmplitude(0.0f)
        , excessNoiseCV(0.0f)
        , interPulseDistance(0.0f)
        , pulseWidth(0.0f)
        , pw2SlowStepRatio(0.0f)
        , ipd2SlowStepRatio(0.0f)
    { }

    AnalogMode(const AnalogMode& a) = default;

    AnalogMode& operator=(const AnalogMode& a) = default;

    AnalogMode(const AnalogConfigEx& config)
            : baseLabel(config.base()[0])
            , relAmplitude(config.relativeAmplitude())
            , excessNoiseCV(config.intraPulseXsnCV())
            , interPulseDistance(config.ipdMeanSeconds())
            , pulseWidth(config.pulseWidthMeanSeconds())
            , pw2SlowStepRatio(config.pw2SlowStepRatio())
            , ipd2SlowStepRatio(config.ipd2SlowStepRatio())
    { }

    /// support label,[spectrum],relAmplitude constructor, xsNoiseCV=0, ipd=0 and pw=0
    /// In general, I am not a fan of these long anonymous argument lists for constructors because
    /// they are fragile if you accidentally slip one argument. While it is verbose, I recommend constructing
    /// the object with some default values, then overwriting each member by name.
    AnalogMode(char label,
               float amplitude, float xsNoiseCV = 0.0f,
               float ipdSeconds = 0.0f, float pwSeconds = 0.0f)
            : baseLabel(label)
            , relAmplitude(amplitude)
            , excessNoiseCV(xsNoiseCV)
            , interPulseDistance(ipdSeconds)
            , pulseWidth(pwSeconds)
            , pw2SlowStepRatio(0.0)
            , ipd2SlowStepRatio(0.0)
    { }

public:
    char baseLabel;
    float relAmplitude;                     // Relative amplitude
    float excessNoiseCV;
    float interPulseDistance;               // seconds
    float pulseWidth;                       // seconds
    float pw2SlowStepRatio;
    float ipd2SlowStepRatio;
};


inline std::ostream& operator<<(std::ostream& os, const AnalogMode& am)
{
    os << "[" << am.baseLabel
       << ", RA " << am.relAmplitude
       << ", exnCV " << am.excessNoiseCV
       << ", ipd " << am.interPulseDistance
       << ", pw " << am.pulseWidth
       << ", ipd2ssr " << am.ipd2SlowStepRatio
       << ", pw2ssr " << am.pw2SlowStepRatio
       << "]";
    return os;
}

}}}     // namespace PacBio::Mongo::Data

#endif // mongo_dataTypes_AnalogMode_H_
