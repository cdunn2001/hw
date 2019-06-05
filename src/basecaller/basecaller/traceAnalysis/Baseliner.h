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

    static void InitAllocationPools(bool hostExecution);
    static void DestroyAllocationPools();

    static void Finalize();

private: // static members
    static std::unique_ptr<Data::CameraBatchFactory> batchFactory_;

public:
    Baseliner(uint32_t poolId);

public:
    /// Estimate and subtract baseline from rawTrace.
    /// \returns Baseline-subtracted traces with certain trace statistics.
    Data::CameraTraceBatch operator()(Data::TraceBatch<ElementTypeIn> rawTrace)
    {
        // TODO
        assert(rawTrace.GetMeta().PoolId() == poolId_);
        return process(std::move(rawTrace));
    }

private:    // Data
    uint32_t poolId_;

private:    // Customizable implementation
    virtual Data::CameraTraceBatch process(Data::TraceBatch<ElementTypeIn> rawTrace);
};

}}}     // namespace PacBio::Mongo::Basecaller

#endif // mongo_basecaller_traceAnalysis_Baseliner_H_
