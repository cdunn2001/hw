#ifndef Mongo_Basecaller_Analyzer_ITraceAnalyzer_H_
#define Mongo_Basecaller_Analyzer_ITraceAnalyzer_H_

#include <vector>

#include <dataTypes/BasecallBatch.h>
#include <dataTypes/TraceBatch.h>
#include <dataTypes/ConfigForward.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

/// The interface for a chip-chunk trace analyzer that calls bases.
/// Client should create a new analyzer for each sequencing acquisition.
/// Data are processed in _chip chunks_, which is one _chunk_ of frames
/// (i.e., sensor readouts) for the entire sensor.
/// Each chip chunk is spatially segmented into a number of _batches_, which
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

    // TODO: Maybe relocate this functionality to a "factory" class?
    /// Creates a new analyzer.
    /// The implementation is specified by the config.
    /// See each implementation for details on implementation specific
    /// configuration parameters.
    /// \param numPools
    /// The total number of pools of ZMW lanes that the constructed analyzer
    /// will be asked to process.
    /// \param bcConfig
    /// Configuration object.
    /// \param movConfig
    /// Describes the configuration of the instrument and chemistry for the
    /// movie to be analyzed.
    static std::unique_ptr<ITraceAnalyzer>
    Create(unsigned int numPools,
           const Data::BasecallerAlgorithmConfig& bcConfig,
           const Data::MovieConfig& movConfig,
           bool simulateBasecalls=false);

public:
    virtual ~ITraceAnalyzer() noexcept = default;

public:
    /// The number of worker threads used by this analyzer.
    virtual unsigned int NumWorkerThreads() const = 0;

    /// The number of ZMW pools supported by this analyzer.
    virtual unsigned int NumZmwPools() const = 0;

    /// The workhorse function. Not const because ZMW-specific state is updated.
    /// GetMeta().PoolId() must be in [0, NumZmwPools) and unique for all
    /// elements of input.
    /// GetMeta().FirstFrame() must be the same for all elements of input.
    /// GetMeta().LastFrame() must be the same for all elements of input.
    std::vector<Data::BasecallBatch>
    operator()(std::vector<Data::TraceBatch<int16_t>> input)
    {
        if (input.empty()) return std::vector<Data::BasecallBatch>();
        assert(IsValid(input));
        return Analyze(std::move(input));
    }

private:    // Functions
    /// Sets the number of worker threads requested.
    /// To choose the default value for the platform, specify 0.
    virtual void NumWorkerThreads(unsigned int) = 0;

    /// The polymorphic implementation point.
    virtual std::vector<Data::BasecallBatch>
    Analyze(std::vector<Data::TraceBatch<int16_t>> input) = 0;

    // Returns true if the input meets basic contracts.
    bool IsValid(const std::vector<Data::TraceBatch<int16_t>>& input);
};


}}} // PacBio::Mongo::Basecaller

#endif  // Mongo_Basecaller_Analyzer_ITraceAnalyzer_H_
