#ifndef mongo_basecaller_traceAnalysis_TraceHistogramAccumulator_H_
#define mongo_basecaller_traceAnalysis_TraceHistogramAccumulator_H_

#include <vector>
#include <dataTypes/CameraTraceBatch.h>
#include <dataTypes/PoolHistogram.h>
#include <dataTypes/ConfigForward.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

class TraceHistogramAccumulator
{
public:     // Types
    using DataType = Data::CameraTraceBatch::ElementType;
    using CountType = unsigned int;

public:     // Static functions
    static void Configure(const Data::BasecallerTraceHistogramConfig& histConfig,
                          const Data::MovieConfig& movConfig);

public:     // Structors and assignment
    TraceHistogramAccumulator(uint32_t poolId);

public:     // Const functions
    /// Total number of frames accumulated for each ZMW of the pool.
    /// Note that these counts are not necessarily uniform due to filtering of
    /// edge frames.
    std::vector<CountType> FrameCount() const
    {
        // TODO
        return std::vector<CountType>();
    }

    Data::PoolHistogram Histogram() const
    {
        // TODO
        return Data::PoolHistogram();
    }

public:     // Non-const functions
    /// Adds data to histograms for a pool.
    /// May include filtering of edge frames.
    void AddBatch(const Data::CameraTraceBatch& ctb)
    {
        // TODO
    }

private:    // Data
    uint32_t poolId_;
};

}}}     // namespace PacBio::Mongo::Basecaller

#endif // mongo_basecaller_traceAnalysis_TraceHistogramAccumulator_H_
