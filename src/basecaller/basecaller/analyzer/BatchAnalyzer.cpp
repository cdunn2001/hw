
// Copyright (c) 2019, Pacific Biosciences of California, Inc.
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
//  Defines members of class BatchAnalyzer.

#include "BatchAnalyzer.h"

#include <dataTypes/BasecallBatch.h>
#include <dataTypes/TraceBatch.h>
#include <dataTypes/BasecallerConfig.h>

using namespace PacBio::Mongo::Data;

namespace PacBio {
namespace Mongo {
namespace Basecaller {

BatchAnalyzer::BatchAnalyzer(uint32_t batchId,
                             const BasecallerAlgorithmConfig& bcConfig,
                             const MovieConfig& movConfig)
    : batchId_ (batchId)
{ }

BasecallBatch BatchAnalyzer::operator()(TraceBatch<int16_t> tbatch)
{
    if (tbatch.Metadata().PoolId() != batchId_)
    {
        // TODO: Log error. Throw exception.
    }

    if (tbatch.Metadata().FirstFrame() != nextFrameId_)
    {
        // TODO: Log error. Throw exception.
    }

    // TODO: Define this so that it scales properly with chunk size, frame rate,
    // and max polymerization rate.
    const uint16_t maxCallsPerZmwChunk = 96;

    // TODO: Implement the analysis logic!

    nextFrameId_ = tbatch.Metadata().LastFrame();

    auto basecalls = BasecallBatch(maxCallsPerZmwChunk, tbatch.Dimensions(), tbatch.Metadata());

    using NucleotideLabel = PacBio::SmrtData::NucleotideLabel;

    // Repeating sequence of ACGT.
    const NucleotideLabel labels[] = { NucleotideLabel::A, NucleotideLabel::C,
                                       NucleotideLabel::G, NucleotideLabel::T };

    // Associated values
    const std::array<float, 4> meanSignals { { 20.0f, 10.0f, 16.0f, 8.0f } };
    const std::array<float, 4> midSignals { { 21.0f, 11.0f, 17.0f, 9.0f } };
    const std::array<float, 4> maxSignals { { 21.0f, 11.0f, 17.0f, 9.0f } };

    static constexpr int8_t qvDefault_ = 0;

    for (uint32_t z = 0; z < basecalls.Dims().zmwsPerBatch(); z++)
    {
        for (uint16_t b = 0; b < maxCallsPerZmwChunk; b++)
        {
            BasecallBatch::Basecall bc;
            auto& pulse = bc.GetPulse();

            size_t iL = b % 4;
            size_t iA = (b + 1) % 4;

            auto label = labels[iL];
            auto altLabel = labels[iA];

            // Populate pulse data
            pulse.Start(1).Width(3);
            pulse.MeanSignal(meanSignals[iL]).MidSignal(midSignals[iL]).MaxSignal(maxSignals[iL]);
            pulse.Label(label).LabelQV(qvDefault_);
            pulse.AltLabel(altLabel).AltLabelQV(qvDefault_);
            pulse.MergeQV(qvDefault_);

            // Populate base data.
            bc.Base(label).InsertionQV(qvDefault_);
            bc.DeletionTag(NucleotideLabel::N).DeletionQV(qvDefault_);
            bc.SubstitutionTag(NucleotideLabel::N).SubstitutionQV(qvDefault_);

            basecalls.PushBack(z, bc);
        }
    }

    return basecalls;
}

}}}     // namespace PacBio::Mongo::Basecaller
