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
    using HistDataType = float;
    using HistCountType = unsigned short;
    using PoolHistType = Data::PoolHistogram<HistDataType, HistCountType>;

public:     // Static functions
    static void Configure(const Data::BasecallerTraceHistogramConfig& histConfig,
                          const Data::MovieConfig& movConfig);

public:     // Structors and assignment
    TraceHistogramAccumulator(uint32_t poolId, unsigned int poolSize);

public:     // Const functions
    /// Total pool frames added.
    /// Total of counts in each zmw histogram will typically be somewhat
    /// smaller due to filtering of edge frames.
    size_t FramesAdded() const
    { return frameCount_; }

    const PoolHistType& Histogram() const
    { return poolHist_; }

public:     // Non-const functions
    /// Adds data to histograms for a pool.
    /// May include filtering of edge frames.
    void AddBatch(const Data::CameraTraceBatch& ctb)
    {
        assert (ctb.GetMeta().PoolId() == poolId_);
        AddBatchImpl(ctb);
        frameCount_ += ctb.numFrames();
    }

protected:  // Data

private:    // Data
    uint32_t poolId_;
    size_t frameCount_ = 0;
    PoolHistType poolHist_;

private:
    // Bins frames in ctb and updates poolHist_.
    virtual void AddBatchImpl(const Data::CameraTraceBatch& ctb) = 0;
};

}}}     // namespace PacBio::Mongo::Basecaller

#endif // mongo_basecaller_traceAnalysis_TraceHistogramAccumulator_H_
