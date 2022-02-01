#ifndef mongo_basecaller_traceAnalysis_Baseliner_H_
#define mongo_basecaller_traceAnalysis_Baseliner_H_

#include <stdint.h>

#include <common/IntInterval.h>
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
    using ElementTypeOut = Data::BaselinedTraceElement;
    using FrameIntervalType = IntInterval<Data::FrameIndexType>;

public:     // Static functions
    /// Sets algorithm configuration and system calibration properties.
    /// Static because the config types keep a JSON representation and
    /// deserialize on each reference, but the values will be the same for
    /// each Baseliner instance for a given movie.
    /// \note Not thread safe. Do not call this function while threads are
    /// running analysis.
    static void Configure(const Data::BasecallerBaselinerConfig& baselinerConfig,
                          const Data::AnalysisConfig& analysisConfig);

    static void InitFactory(bool hostExecution, const Data::AnalysisConfig& analysisConfig);

    static void Finalize();

protected: // static members
    static std::unique_ptr<Data::CameraBatchFactory> batchFactory_;
    static float movieScaler_;
    static int16_t pedestal_;
    static DataSource::PacketLayout::EncodingFormat expectedEncoding_;

public:
    Baseliner(uint32_t poolId)
        : poolId_(poolId)
    { }
    virtual ~Baseliner() = default;

public:
    /// Estimate and subtract baseline from rawTrace.
    /// \returns Baseline-subtracted traces with certain trace statistics.
    std::pair<Data::TraceBatch<ElementTypeOut>,
              Data::BaselinerMetrics>
    operator()(const Data::TraceBatchVariant& rawTrace)
    {
        assert(rawTrace.Metadata().PoolId() == poolId_);
        auto result = FilterBaseline(rawTrace);

        // Transcribe the frame interval to the metrics object.
        const auto& tracemd = result.first.Metadata();
        const FrameIntervalType fi {tracemd.FirstFrame(), tracemd.LastFrame()};
        result.second.frameInterval = fi;

        return result;
    }

    float Scale() const { return movieScaler_; }

    // Will indicate the number of frames that are input
    // before all startup transients are flushed.  Data
    // before this point isn't necessarily reliable.
    virtual size_t StartupLatency() const = 0;

private:    // Customizable implementation
    virtual std::pair<Data::TraceBatch<ElementTypeOut>,
                      Data::BaselinerMetrics>
    FilterBaseline(const Data::TraceBatchVariant& rawTrace) = 0;

private:    // Data
    uint32_t poolId_;
};

}}}     // namespace PacBio::Mongo::Basecaller

#endif // mongo_basecaller_traceAnalysis_Baseliner_H_
