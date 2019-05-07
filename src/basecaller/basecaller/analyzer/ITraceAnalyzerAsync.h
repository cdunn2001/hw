#ifndef Mongo_Basecaller_Analyzer_ITraceAnalyzerAsync_H_
#define Mongo_Basecaller_Analyzer_ITraceAnalyzerAsync_H_

#include <vector>
#include <tbb/concurrent_bounded_queue>
#include <pacbio/common/Acquisition_Setup.h>

namespace PacBio {
namespace Primary {

// A movable but non-copyable container of trace data.
class TraceBatch;

// A movable but non-copyable container of basecall results.
class ReadBatch;    // TODO: Need a better term than "read".

namespace Acquisition {
    class Setup;
}

namespace Basecaller {

/// The interface for the streaming or "real-time" analyzer that calls bases.
/// Client should create a new analyzer for each sequencing acquisition.
/// Data are processed in _chip chunks_, which is one _chunk_ of frames
/// (i.e., sensor readouts) for the entire sensor.
/// Each chip chunk is spatially segmented into a number of _batches_, which
/// the analyzer processes concurrently.
/// For a description of the memory layout of trace data, see
/// https://confluence.pacificbiosciences.com/display/PA/Memory+Layout+of+Trace+Data+in+Mongo+Basecaller
class ITraceAnalyzerAsync
{
public:     // Static functions
    // TODO: Do we really need this static function?
    /// Prepares any static objects used by all instances.
    /// \returns \c true on success.
    static bool Initialize(const PacBio::Primary::Basecaller::BasecallerInitConfig& startupConfig);

    /// Creates a new analyzer.
    /// The implementation is specified the config.
    /// See each implementation for details on implementation specific
    /// configuration parameters.
    /// \param config
    /// Configuration object. See http://smrtanalysis-docs/primary/top/doc/PacBioPrimaryConfiguration.html.
    /// \param numLaneBatches
    /// The total number of lane batches for which the constructed analyzer will
    /// receive data through \a inputQueue.
    /// \param inputQueue
    /// The queue from which the analyzer will retrieve input data.
    /// \param outputQueue
    /// The queue to which the analyzer will put analysis results.
    static std::unique_ptr<ITraceAnalyzerAsync> Create(
            const PacBio::Primary::Basecaller::BasecallerAlgorithmConfig& config,
            const PacBio::Primary::Acquisition::Setup& setup,
            unsigned int numLaneBatches,
            tbb::concurrent_bounded_queue<std::vector<TraceBatch>>& inputQueue,
            tbb::concurrent_bounded_queue<std::vector<ReadBatch>& outputQueue
    );

public:     // Polymorphic functions
    virtual ~ITraceAnalyzerAsync() noexcept = default;

    /// Returns the number of worker threads.
    unsigned int NumWorkerThreads() const;

    /// Activate the analyzer. Returns when the acquisition is aborted or
    /// stopped, which occurs after the analyzer retrieves a "last chunk"
    /// tranche from the inputQueue for each lane.
    /// Idempotent.
    virtual void Activate() = 0;

    /// Abort analysis. In other words, deactivate. Idempotent.
    virtual void Abort() = 0;

    /// Return activated state.
    virtual bool IsActivated() = 0;

    /// Return terminated state.
    virtual bool IsTerminated() = 0;

private:    // Functions
    /// Set the number of worker threads to be requested.
    /// To choose the default value for the platform, specify 0.
    void NumWorkerThreads(unsigned int);

private:    // Data
    unsigned int numWorkerThreads_;
};


}}} // PacBio::Primary::Basecaller

#endif  // Mongo_Basecaller_Analyzer_ITraceAnalyzerAsync_H_
