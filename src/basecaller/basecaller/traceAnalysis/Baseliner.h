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
    using ElementTypeOut = Data::BaselinedTraceElement;

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
    Baseliner(uint32_t poolId, float scaler = 1.0f)
        : poolId_(poolId)
        , scaler_(scaler)
    { }

public:
    /// Estimate and subtract baseline from rawTrace.
    /// \returns Baseline-subtracted traces with certain trace statistics.
    Data::CameraTraceBatch operator()(Data::TraceBatch<ElementTypeIn> rawTrace)
    {
        assert(rawTrace.GetMeta().PoolId() == poolId_);
        return Process(std::move(rawTrace));
    }

private:    // Customizable implementation
    virtual Data::CameraTraceBatch Process(Data::TraceBatch<ElementTypeIn> rawTrace) = 0;

private:    // Data
    uint32_t poolId_;
    float scaler_;
};



}}}     // namespace PacBio::Mongo::Basecaller

#endif // mongo_basecaller_traceAnalysis_Baseliner_H_
