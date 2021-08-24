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

#ifndef mongo_dataTypes_configs_PrelimHQConfig_H_
#define mongo_dataTypes_configs_PrelimHQConfig_H_

#include <pacbio/configuration/PBConfig.h>

namespace PacBio::Mongo::Data {

class PrelimHQConfig : public Configuration::PBConfig<PrelimHQConfig>
{
    PB_CONFIG(PrelimHQConfig);

    // TODO this should maybe be in units of frames?  Something to
    // think about while the PrelimHQ stage gets implemented for real
    PB_CONFIG_PARAM(size_t, bazBufferChunks, 8);
    PB_CONFIG_PARAM(size_t, zmwOutputStride, 1);

    PB_CONFIG_PARAM(bool, enableLookback, false);
    PB_CONFIG_PARAM(uint32_t, lookbackSize, 10);
    // *If* enableLookback is turned on, this fraction controls the
    // percentage of remaining ZMW allowed to become HQ at once.
    PB_CONFIG_PARAM(float, hqThrottleFraction, .10f);

    // This is a trash config, but right now nothing in the config
    // tree knows what the frame rate is yet...
    PB_CONFIG_PARAM(float, expectedFrameRate, 100);

    // The maximum supported pulse rate.  Note, this is specifically
    // for the PrelimHQ stage, other computations over different
    // timescales may have a different max pulse rate.
    PB_CONFIG_PARAM(float, maxSlicePulseRate, 4);

    // The number of pulses expected per ZMW in a metric block interval.
    // Note: One would normally prefer to handle such computation in
    //       the C++ implementation directly rather than as a dependent
    //       default.  In this case the computation itself needs to be
    //       overrideable since it's not clear what the best thing to do
    //       is.  This calculation will reserve enough space for all ZMW
    //       to go full-tilt at the max rate.  If we reserved a smaller
    //       amount of space we might have a lowered memory requirement,
    //       but we also might have a performance impact as the data for
    //       a single zmw gets fragmented and scattered amongst more smaller
    //       allocations.
    PB_CONFIG_PARAM(size_t,
                    expectedPulsesPerZmw,
                    Configuration::DefaultFunc(
                        [](size_t numFrames, float frameRate, float pulseRate) -> size_t {
                            return static_cast<size_t>(std::ceil(numFrames / frameRate * pulseRate));
                        },
                        {"Metrics.framesPerHFMetricBlock", "expectedFrameRate", "maxSlicePulseRate"}));
};

}  // namespace PacBio::Mongo::Data


#endif  // mongo_dataTypes_configs_PrelimHQConfig_H_
