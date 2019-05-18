#ifndef mongo_basecaller_traceAnalysis_Baseliner_H_
#define mongo_basecaller_traceAnalysis_Baseliner_H_

#include <stdint.h>

#include <dataTypes/BasicTypes.h>
#include <dataTypes/ConfigForward.h>
#include <dataTypes/CameraTraceBatch.h>
#include <dataTypes/TraceBatch.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

class Baseliner
{
public:     // Types
    using ElementTypeIn = Data::RawTraceElement;

public:
    Baseliner(uint32_t poolId,
              const Data::BasecallerBaselinerConfig& baselinerConfig,
              const Data::MovieConfig& movConfig);

public:
    Data::CameraTraceBatch operator()(Data::TraceBatch<ElementTypeIn> rawTrace)
    {
        // TODO
        return Data::CameraTraceBatch(rawTrace.GetMeta(),
                                      rawTrace.Dimensions(),
                                      Cuda::Memory::SyncDirection::Symmetric);
    }

private:
    uint32_t poolId_;
};

}}}     // namespace PacBio::Mongo::Basecaller

#endif // mongo_basecaller_traceAnalysis_Baseliner_H_
