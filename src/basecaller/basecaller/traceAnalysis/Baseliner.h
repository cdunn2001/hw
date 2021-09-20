#ifndef mongo_basecaller_traceAnalysis_Baseliner_H_
#define mongo_basecaller_traceAnalysis_Baseliner_H_

#include <stdint.h>

#include <dataTypes/BasicTypes.h>
#include <dataTypes/BatchMetrics.h>
#include <dataTypes/configs/ConfigForward.h>
#include <dataTypes/CameraTraceBatch.h>
#include <dataTypes/TraceBatch.h>
#include <dataTypes/BasicTypes.h>

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

    static void InitFactory(bool hostExecution);

    static void Finalize();

protected: // static members
    static std::unique_ptr<Data::CameraBatchFactory> batchFactory_;

public:
    Baseliner(uint32_t poolId, float scaler)
        : poolId_(poolId)
        , scaler_(scaler)
    { }
    virtual ~Baseliner() = default;

public:
    /// Estimate and subtract baseline from rawTrace.
    /// \returns Baseline-subtracted traces with certain trace statistics.
    std::pair<Data::TraceBatch<ElementTypeOut>,
                      Data::BaselinerMetrics>
    FilterBaseline(const Data::TraceBatch<ElementTypeIn>& rawTrace)
    {
        assert(rawTrace.GetMeta().PoolId() == poolId_);
        return FilterBaseline_(rawTrace);
    }

    float Scale() const { return scaler_; }

    // Will indicate the number of frames that are input
    // before all startup transients are flushed.  Data
    // before this point isn't necessarily reliable.
    virtual size_t StartupLatency() const = 0;

private:    // Customizable implementation
    virtual std::pair<Data::TraceBatch<ElementTypeOut>,
                      Data::BaselinerMetrics>
    FilterBaseline_(const Data::TraceBatch<ElementTypeIn>& rawTrace) = 0;

private:    // Data
    uint32_t poolId_;
    float scaler_;
};

}}}     // namespace PacBio::Mongo::Basecaller

#endif // mongo_basecaller_traceAnalysis_Baseliner_H_
