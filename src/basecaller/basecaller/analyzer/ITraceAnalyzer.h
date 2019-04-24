#ifndef Mongo_Basecaller_Analyzer_ITraceAnalyzer_H_
#define Mongo_Basecaller_Analyzer_ITraceAnalyzer_H_

#include <vector>
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

/// The interface for a chip-block trace analyzer that calls bases.
/// Client should create a new analyzer for each sequencing acquisition.
/// Data are processed in _chip blocks_, which is one _block_ of frames
/// (i.e., sensor readouts) for the entire sensor.
/// Each chip block is spatially segmented into a number of _batches_, which
/// the analyzer processes concurrently.
/// For a description of the memory layout of trace data, see
/// https://confluence.pacificbiosciences.com/display/PA/Memory+Layout+of+Trace+Data+in+Mongo+Basecaller
class ITraceAnalyzer
{
public:     // Static functions
    // TODO: Do we really need this static function?
    /// Prepares any static objects used by all instances.
    /// \returns \c true on success.
    static bool Initialize(const PacBio::Primary::Basecaller::BasecallerInitConfig& startupConfig);

    /// Creates a new analyzer.
    /// The implementation is specified by the config.
    /// See each implementation for details on implementation specific
    /// configuration parameters.
    /// \param config
    /// Configuration object.
    /// See http://smrtanalysis-docs/primary/top/doc/PacBioPrimaryConfiguration.html.
    /// \param numLaneBatches
    /// The total number of lane batches for which the constructed analyzer
    /// will asked to process.
    static std::unique_ptr<ITraceAnalyzer> Create(
            const PacBio::Primary::Basecaller::BasecallerAlgorithmConfig& config,
            const PacBio::Primary::Acquisition::Setup& setup,
            unsigned int numLaneBatches
    );

public:
    virtual ~ITraceAnalyzer() noexcept = default;

public:
    /// Returns the number of worker threads.
    unsigned int NumWorkerThreads() const;

    /// The workhorse function. Not const because ZMW-specific state is updated.
    std::vector<ReadBatch> operator()(std::vector<TraceBatch> input)
    {
        // TODO: Might want to add some framework logic.
        return analyze(input);
    }

private:    // Functions
    /// Sets the number of worker threads requested.
    /// To choose the default value for the platform, specify 0.
    void NumWorkerThreads(unsigned int);

    /// The polymorphic implementation point.
    virtual std::vector<ReadBatch> analyze(std::vector<TraceBatch> input);

private:    // Data members
    unsigned int numWorkerThreads_;
};


}}} // PacBio::Primary::Basecaller

#endif  // Mongo_Basecaller_Analyzer_ITraceAnalyzer_H_
