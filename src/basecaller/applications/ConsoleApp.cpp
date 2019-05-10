
#include <basecaller/analyzer/ITraceAnalyzer.h>
#include <dataTypes/BasecallerConfig.h>
#include <dataTypes/BatchMetadata.h>
#include <dataTypes/MovieConfig.h>

#include <pacbio/PBException.h>
#include <pacbio/logging/Logger.h>
#include <pacbio/process/OptionParser.h>
#include <pacbio/process/ProcessBase.h>

using namespace PacBio::Mongo::Data;
using namespace PacBio::Mongo::Basecaller;

class MongoBasecallerConsole : public PacBio::Process::ThreadedProcessBase
{
public:
    MongoBasecallerConsole()
    {}

    ~MongoBasecallerConsole()
    {}

    void HandleProcessArguments(const std::vector<std::string>& args)
    {
        if (args.size() == 1)
        {
            inputTargetFile_ = args[0];
            PBLOG_INFO << "Input Target: " << inputTargetFile_;
        }
    }

    void HandleProcessOptions(const PacBio::Process::Values& options)
    {
        // Determine data layout.

        // Determine data selection.

        // Determine if tiling is needed.
    }

    void Run()
    {
        Setup();
        RunAnalyzer();
        Join();
        Teardown();
    }

    void RunAnalyzer()
    {

    }

    void Teardown()
    {

    }

    BasecallerAlgorithmConfig& BasecallerConfig()
    { return basecallerConfig_; };

private:
    void Setup()
    {
        MovieConfig movConfig;
        analyzer_ = ITraceAnalyzer::Create(numPools_, basecallerConfig_, movConfig);
    }

    void RunWriter()
    {

    }


private:
    BasecallerAlgorithmConfig basecallerConfig_;

    std::unique_ptr<ITraceAnalyzer> analyzer_;
    std::string inputTargetFile_;
    unsigned int numPools_;
};

int main(int argc, char* argv[])
{
    try
    {
        auto parser = PacBio::Process::OptionParser().description("Prototype to demonstrate mongo basecaller");
        parser.version("0.1");

        parser.epilog("");

        parser.add_option("--config").action_append().help("Loads JSON configuration file, JSON string or Boost ptree value");
        parser.add_option("--strict").action_store_true().help("Strictly check all configuration options. Do not allow unrecognized configuration options");
        parser.add_option("--showconfig").action_store_true().help("Shows the entire configuration namespace and exits");

        parser.add_option("--inputfile").set_default("").help("input file (must be *.trc.h5)");
        parser.add_option("--outputbazfile").set_default("").help("BAZ output file");

        auto group1 = PacBio::Process::OptionGroup(parser, "Data Layout Options",
                                                   "Controls the layout of the data to the basecaller");
        group1.add_option("--framesPerBlock").type_int().set_default(64).help(
                "Number of frames per block (Default: %default)");
        group1.add_option("--blocksPerBatch").type_int().set_default(10).help(
                "Number of blocks per a batch (Default: %default)");
        group1.add_option("--zmwsPerLane").type_int().set_default(64).help(
                "Number of ZMWs per a lane (Default: %default)");
        group1.add_option("--lanesPerBatch").type_int().set_default(5000).help(
                "Number of lanes per a batch (Default: %default");
        parser.add_option_group(group1);

        auto group2 = PacBio::Process::OptionGroup(parser, "Data Selection Options",
                                                   "Controls data selection from input data");
        group2.add_option("--zmwHoleNumbers").type_string().help("");
        group2.add_option("--startFrame").type_int().help("Starting frame to begin analysis");
        group2.add_option("--numFrames").type_int().help(
                "Number of frames, defaults to number of frames in input file");
        parser.add_option_group(group2);

        auto group3 = PacBio::Process::OptionGroup(parser, "Data Tiling Options",
                                                   "Controls data tiling options for simulation and testing");
        group3.add_option("--tileBatches").type_int().help("Specifies number of batches to tile out");
        group3.add_option("--tileBlocks").type_int().help("Specifies in terms of blocks to tile out");
        parser.add_option_group(group3);

        auto options = parser.parse_args(argc, (const char* const*) argv);
        PacBio::Process::ThreadedProcessBase::HandleGlobalOptions(options);

        PacBio::Process::ConfigMux mux;
        BasecallerAlgorithmConfig basecallerConfig;
        mux.Add("basecaller", basecallerConfig);
        mux.Add("common", primaryConfig);
        mux.ProcessCommandLine(options.all("config"));

        if (options.get("showconfig"))
        {
            std::cout << mux.ToJson() << std::endl;
            return 0;
        }

        auto bc = std::unique_ptr<MongoBasecallerConsole>(new MongoBasecallerConsole());

        bc->HandleProcessArguments(parser.args());

        bc->BasecallerConfig().SetStrict(options.get("strict"));
        bc->BasecallerConfig().Load(basecallerConfig);

        bc->HandleProcessOptions(options);

        bc->Run();

    } catch (std::exception &ex) {
        PBLOG_ERROR << "Exception caught: " << ex.what();
        return 1;
    }

    return 0;
}

