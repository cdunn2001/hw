#ifndef mongo_basecaller_traceAnalysis_DetectionModelEstimation_H_
#define mongo_basecaller_traceAnalysis_DetectionModelEstimation_H_

#include <stdint.h>

#include <dataTypes/DetectionModel.h>
#include <dataTypes/PoolHistogram.h>
#include <dataTypes/ConfigForward.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

class DetectionModelEstimator
{
public:     // Static functions
    static void Configure(const Data::BasecallerDmeConfig& dmeConfig,
                          const Data::MovieConfig& movConfig);

public:     // Structors and assignment
    DetectionModelEstimator(uint32_t poolId);

    void operator()(const Data::PoolHistogram<float, unsigned short>& hist,
                    Data::DetectionModel& model)
    {
        // TODO
    }

private:
    uint32_t poolId_;
};

}}}     // namespace PacBio::Mongo::Basecaller

#endif // mongo_basecaller_traceAnalysis_DetectionModelEstimation_H_
