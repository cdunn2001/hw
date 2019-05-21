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

public:     // Static functions
    /// Sets algorithm configuration and system calibration properties.
    /// Static because the config types keep a JSON representation and
    /// deserialize on each reference, but the values will be the same for
    /// each Baseliner instance for a given movie.
    /// \note Not thread safe. Do not call this function while threads are
    /// running analysis.
    static void Configure(const Data::BasecallerBaselinerConfig& baselinerConfig,
                          const Data::MovieConfig& movConfig);

public:
    Baseliner(uint32_t poolId);

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
