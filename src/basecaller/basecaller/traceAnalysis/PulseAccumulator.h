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

#ifndef PACBIO_MONGO_BASECALLER_PULSE_ACCUMULATOR_H_
#define PACBIO_MONGO_BASECALLER_PULSE_ACCUMULATOR_H_

#include <stdint.h>

#include <dataTypes/BatchMetrics.h>
#include <dataTypes/ConfigForward.h>
#include <dataTypes/LabelsBatch.h>
#include <dataTypes/PulseBatch.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

class PulseAccumulator
{
public:     // Static functions

    static void Configure(size_t maxCallsPerZmw);
    static void Finalize();

    static void InitFactory(bool hostExecution, size_t maxCallsPerZmw);

protected: // static members
    static std::unique_ptr<Data::PulseBatchFactory> batchFactory_;


public:
    PulseAccumulator(uint32_t poolId);
    virtual ~PulseAccumulator() = default;

public:
    /// \returns LabelsBatch with estimated labels for each frame, along with the associated
    ///          baseline subtracted trace
    std::pair<Data::PulseBatch, Data::PulseDetectorMetrics>
    operator()(Data::LabelsBatch labels)
    {
        // TODO
        assert(labels.GetMeta().PoolId() == poolId_);
        return Process(std::move(labels));
    }

    auto EmptyPulseBatch(const Data::BatchMetadata& metadata, const Data::BatchDimensions& dims)
    {
        auto ret = batchFactory_->NewBatch(metadata, dims);

        for (size_t laneIdx = 0; laneIdx < ret.first.Dims().lanesPerBatch; ++laneIdx)
        {
            auto lanePulses = ret.first.Pulses().LaneView(laneIdx);
            lanePulses.Reset();
        }

        return ret;
    }

private:    // Data
    uint32_t poolId_;

private:    // Customizable implementation
    virtual std::pair<Data::PulseBatch, Data::PulseDetectorMetrics>
    Process(Data::LabelsBatch trace); // = 0;
};

}}}     // namespace PacBio::Mongo::Basecaller

#endif //PACBIO_MONGO_BASECALLER_PULSE_ACCUMULATOR_H_
