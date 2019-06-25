#ifndef mongo_basecaller_traceAnalysis_DetectionModelEstimator_H_
#define mongo_basecaller_traceAnalysis_DetectionModelEstimator_H_

#include <stdint.h>

#include <common/cuda/memory/UnifiedCudaArray.h>
#include <common/cuda/PBCudaSimd.h>

#include <dataTypes/AnalogMode.h>
#include <dataTypes/BaselineStats.h>
#include <dataTypes/ConfigForward.h>
#include <dataTypes/PoolDetectionModel.h>
#include <dataTypes/PoolHistogram.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

class DetectionModelEstimator
{
public:     // Types
    using DetModelElementType = Cuda::PBHalf;
    using PoolDetModel = Data::PoolDetectionModel<DetModelElementType>;
    using LaneDetModel = Data::LaneDetectionModel<DetModelElementType>;

public:     // Static functions
    static void Configure(const Data::BasecallerDmeConfig& dmeConfig,
                          const Data::MovieConfig& movConfig);

public:     // Structors and assignment
    DetectionModelEstimator(uint32_t poolId, unsigned int poolSize);

    PoolDetModel operator()(const Data::PoolHistogram<float, unsigned short>& hist,
                            const Cuda::Memory::UnifiedCudaArray<Data::BaselineStats<laneSize>>& blStats)
    {
        assert (hist.poolId == poolId_);

        PoolDetModel pdm (poolId_, poolSize_, Cuda::Memory::SyncDirection::Symmetric);

        auto pdmHost = pdm.laneModels.GetHostView();
        const auto& blStatsHost = blStats.GetHostView();
        for (unsigned int lane = 0; lane < poolSize_; ++lane)
        {
            InitDetModel(blStatsHost[lane], pdmHost[lane]);
        }

        // TODO

        return pdm;
    }

private:    // Static data
    static Cuda::Utility::CudaArray<Data::AnalogMode, numAnalogs> analogs_;
    static float refSnr_;   // Expected SNR for analog with relative amplitude of 1.

private:
    uint32_t poolId_;
    unsigned int poolSize_;

private:    // Functions
    void InitDetModel(const Data::BaselineStats<laneSize>& blStats, LaneDetModel& ldm);
};

}}}     // namespace PacBio::Mongo::Basecaller

#endif // mongo_basecaller_traceAnalysis_DetectionModelEstimator_H_
