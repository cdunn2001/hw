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

#ifndef PACBIO_MONGO_BASECALLER_FRAME_LABELER_H_
#define PACBIO_MONGO_BASECALLER_FRAME_LABELER_H_

#include <stdint.h>

#include <dataTypes/BasicTypes.h>
#include <dataTypes/BatchMetrics.h>
#include <dataTypes/LabelsBatch.h>
#include <dataTypes/LaneDetectionModel.h>
#include <dataTypes/configs/BasecallerFrameLabelerConfig.h>
#include <dataTypes/configs/ConfigForward.h>


namespace PacBio {
namespace Mongo {
namespace Basecaller {

class FrameLabeler
{
public:  // types
    using ElementType = Data::BaselinedTraceElement;
    using LaneModelParameters = Data::LaneModelParameters<Cuda::PBHalf, laneSize>;
    using PoolModelParameters = Cuda::Memory::UnifiedCudaArray<LaneModelParameters>;

public:     // Static functions
    static void Configure();
    static void Finalize();

    static void InitFactory(bool hostExecution, size_t latentFrames);

protected: // static members
    static std::unique_ptr<Data::LabelsBatchFactory> batchFactory_;
    static Data::BasecallerRoiConfig::RoiFilterType roiType;

public:
    FrameLabeler(uint32_t poolId);
    virtual ~FrameLabeler() = default;

public:
    /// \returns LabelsBatch with estimated labels for each frame, along with the associated
    ///          baseline subtracted trace
    std::pair<Data::LabelsBatch, Data::FrameLabelerMetrics>
    operator()(Data::TraceBatch<ElementType> trace,
               const PoolModelParameters& models)
    {
        // TODO
        assert(trace.GetMeta().PoolId() == poolId_);
        return Process(std::move(trace), models);
    }

    auto EmptyMetrics(const Data::BatchDimensions& dims)
    {
        auto ret = batchFactory_->NewMetrics(dims);

        for (size_t laneIdx = 0; laneIdx < dims.lanesPerBatch; ++laneIdx)
        {
            auto laneMetrics = ret.viterbiScore.GetHostView()[laneIdx];
            for (size_t zi = 0; zi < laneSize; ++zi)
            {
                laneMetrics[zi] = 0.0f;
            }
        }

        return ret;
    }

    auto EmptyLabelsBatch(Data::TraceBatch<ElementType> trace)
    {
        auto ret = batchFactory_->NewLabels(std::move(trace));
        for (size_t laneIdx = 0; laneIdx < ret.LanesPerBatch(); laneIdx++)
        {
            std::memset(ret.GetBlockView(laneIdx).Data(), 0,
                        ret.GetBlockView(laneIdx).Size() * sizeof(Data::LabelsBatch::ElementType));
        }
        return ret;
    }

private:    // Data
    uint32_t poolId_;

private:    // Customizable implementation
    virtual std::pair<Data::LabelsBatch, Data::FrameLabelerMetrics>
    Process(Data::TraceBatch<ElementType> trace,
            const PoolModelParameters& models); //= 0;
};

}}}     // namespace PacBio::Mongo::Basecaller

#endif //PACBIO_MONGO_BASECALLER_FRAME_LABELER_H_
