#ifndef Mongo_Basecaller_Analyzer_ITraceAnalyzer_H_
#define Mongo_Basecaller_Analyzer_ITraceAnalyzer_H_

#include <vector>

#include <BasecallBatch.h>
#include <TraceBatch.h>
#include <ConfigForward.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

/// The interface for a chip-block trace analyzer that calls bases.
/// Client should create a new analyzer for each sequencing acquisition.
/// Data are processed in _chip blocks_, which is one _block_ of frames
/// (i.e., sensor readouts) for the entire sensor.
/// Each chip block is spatially segmented into a number of _batches_, which
/// the analyzer processes concurrently.
/// For a description of the memory layout of trace data, see
/// https://confluence.pacificbiosciences.com/display/PA/Memory+Layout+of+Trace+Data+in+Mongo+Basecaller.
class ITraceAnalyzer
{
public:     // Static functions
    // TODO: Do we really need this static function?
    /// Prepares any static objects used by all instances.
    /// \returns \c true on success.
    static bool Initialize(const Data::BasecallerInitConfig& startupConfig);

    // TODO: Relocate this functionality to a "factory" class.
    /// Creates a new analyzer.
    /// The implementation is specified by the config.
    /// See each implementation for details on implementation specific
    /// configuration parameters.
    /// \param numPools
    /// The total number of pools of ZMW lanes that the constructed analyzer
    /// will be asked to process.
    /// \param bcConfig
    /// Configuration object.
    /// See http://smrtanalysis-docs/primary/top/doc/PacBioPrimaryConfiguration.html.
    /// \param movConfig
    /// Describes the configuration of the instrument and chemistry for the
    /// movie to be analyzed.
    static std::unique_ptr<ITraceAnalyzer>
    Create(unsigned int numPools,
           const Data::BasecallerAlgorithmConfig& bcConfig,
           const Data::MovieConfig& movConfig);

public:
    virtual ~ITraceAnalyzer() noexcept = default;

public:
    /// Returns the number of worker threads.
    unsigned int NumWorkerThreads() const;

    /// The workhorse function. Not const because ZMW-specific state is updated.
    std::vector<Mongo::Data::BasecallBatch>
    operator()(std::vector<Mongo::Data::TraceBatch<int16_t>> input)
    {
        // TODO: Might want to add some framework logic.
        return analyze(std::move(input));
    }

private:    // Functions
    /// Sets the number of worker threads requested.
    /// To choose the default value for the platform, specify 0.
    void NumWorkerThreads(unsigned int);

    /// The polymorphic implementation point.
    virtual std::vector<Mongo::Data::BasecallBatch>
    analyze(std::vector<Mongo::Data::TraceBatch<int16_t>> input) = 0;

private:    // Data members
    unsigned int numWorkerThreads_;
};


}}} // PacBio::Mongo::Basecaller

#endif  // Mongo_Basecaller_Analyzer_ITraceAnalyzer_H_
