// Copyright (c) 2019-2020, Pacific Biosciences of California, Inc.
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
//  Defines members of class MovieConfig.

#include "MovieConfig.h"

#include <cmath>

namespace PacBio {
namespace Mongo {
namespace Data {

MovieConfig MockMovieConfig()
{
    constexpr float raMin = 0.25f;

    MovieConfig movConfig;
    movConfig.frameRate = 100.0f;
    movConfig.photoelectronSensitivity = 2.0f;
    movConfig.refSnr = 20.0f;
    {
        auto& mca = movConfig.analogs[0];
        mca.baseLabel = 'C';
        mca.relAmplitude = 1.0f;
        mca.excessNoiseCV = 0.1f;
        mca.interPulseDistance = 0.10f;
        mca.pulseWidth = 0.10f;
        mca.pw2SlowStepRatio = 0.5;
        mca.ipd2SlowStepRatio = 0.0;
    }
    {
        auto& mca = movConfig.analogs[1];
        mca.baseLabel = 'A';
        mca.relAmplitude = std::pow(raMin, 1.0f/3);
        mca.excessNoiseCV = 0.1f;
        mca.interPulseDistance = 0.10f;
        mca.pulseWidth = 0.10f;
        mca.pw2SlowStepRatio = 0.5;
        mca.ipd2SlowStepRatio = 0.0;
    }
    {
        auto& mca = movConfig.analogs[2];
        mca.baseLabel = 'T';
        mca.relAmplitude = std::pow(raMin, 2.0f/3);
        mca.excessNoiseCV = 0.1f;
        mca.interPulseDistance = 0.10f;
        mca.pulseWidth = 0.10f;
        mca.pw2SlowStepRatio = 0.5;
        mca.ipd2SlowStepRatio = 0.0;
    }
    {
        auto& mca = movConfig.analogs[3];
        mca.baseLabel = 'G';
        mca.relAmplitude = raMin;
        mca.excessNoiseCV = 0.1f;
        mca.interPulseDistance = 0.10f;
        mca.pulseWidth = 0.10f;
        mca.pw2SlowStepRatio = 0.5;
        mca.ipd2SlowStepRatio = 0.0;
    }

    return movConfig;
}

}}}     // namespace PacBio::Mongo::Data
