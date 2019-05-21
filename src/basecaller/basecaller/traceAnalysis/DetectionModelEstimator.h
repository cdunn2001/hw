#ifndef mongo_basecaller_traceAnalysis_DetectionModelEstimation_H_
#define mongo_basecaller_traceAnalysis_DetectionModelEstimation_H_

#include <dataTypes/DetectionModel.h>
#include <dataTypes/PoolHistogram.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

class DetectionModelEstimator
{
public:
    DetectionModelEstimator();

    Data::DetectionModel operator()(const Data::PoolHistogram& hist)
    {
        // TODO
        return Data::DetectionModel();
    }
};

}}}     // namespace PacBio::Mongo::Basecaller

#endif // mongo_basecaller_traceAnalysis_DetectionModelEstimation_H_
