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
//  Description:
//  Defines members of class MultiScaleBaseliner.

#include "MultiScaleBaseliner.h"

#include <cmath>
#include <sstream>

#include <tbb/task_arena.h>
#include <tbb/parallel_for.h>

#include <dataTypes/BasicTypes.h>
#include <dataTypes/BaselinerStatAccumulator.h>
#include <dataTypes/configs/AnalysisConfig.h>


namespace PacBio {
namespace Mongo {
namespace Basecaller {

// static
float MultiScaleBaseliner::cMeanBiasAdj_ = 0.0f;
float MultiScaleBaseliner::cSigmaBiasAdj_ = 0.0f;
float MultiScaleBaseliner::meanEmaAlpha_ = 0.0f;
float MultiScaleBaseliner::sigmaEmaAlpha_ = 0.0f;
float MultiScaleBaseliner::jumpTolCoeff_ = std::numeric_limits<float>::infinity();

void MultiScaleBaseliner::Configure(const Data::BasecallerBaselinerConfig& bbc,
                                        const Data::AnalysisConfig& analysisConfig)
{

    // static things need to be done in static places - implement this in your
    // device-specific method, then call this general method
    //hostExecution_ = true; // or false
    //InitFactory(hostExecution_, analysisConfig);

    {
        // Validation has already been handled in the configuration framework.
        // This just asserts that the configuration is indeed valid.
        assert(bbc.Validate());

        cMeanBiasAdj_ = bbc.MeanBiasAdjust;

        cSigmaBiasAdj_ = bbc.SigmaBiasAdjust;

        const float meanEmaScale = bbc.MeanEmaScaleStrides;
        meanEmaAlpha_ = std::pow(0.5f, 1.0f / meanEmaScale);
        assert(0.0f <= meanEmaAlpha_ && meanEmaAlpha_ < 1.0f);

        const float sigmaEmaScale = bbc.SigmaEmaScaleStrides;
        std::ostringstream msg;
        msg << "SigmaEmaScaleStrides = " << sigmaEmaScale << '.';
        // TODO: Use a scoped logger.
        PBLOG_INFO << msg.str();
        sigmaEmaAlpha_ = std::exp2(-1.0f / sigmaEmaScale);
        assert(0.0f <= sigmaEmaAlpha_ && sigmaEmaAlpha_ <= 1.0f);

        // TODO: Enable jumpTolCoeff_
        // const float js = bbc.JumpSuppression;
        // jumpTolCoeff_ = (js > 0.0f ? 1.0f / js : std::numeric_limits<float>::infinity());
    }
}

void MultiScaleBaseliner::Finalize() {}

}}}     // namespace PacBio::Mongo::Basecaller
