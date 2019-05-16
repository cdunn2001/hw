#include <pacbio/process/OptionParser.h>
#include <common/DataGenerators/PicketFenceGenerator.h>
#include <common/DataGenerators/SignalGenerator.h>
#include <common/ZmwDataManager.h>

#include "BaselineRunner.h"

using namespace PacBio::Cuda::Memory;
using namespace PacBio::Cuda::Data;
using namespace PacBio::Cuda;

int main(int argc, char* argv[])
{
    DataManagerParams defaultFillerParams;
    PicketFenceParams defaultDataParams;

    auto parser = PacBio::Process::OptionParser().description(
        "Prototype to demonstrate streaming data transfer to gpu");
    parser.version("0.1");

    auto groupMand = PacBio::Process::OptionGroup(parser, "Mandatory parameters");
    groupMand.add_option("--frameRate").type_float().set_default(defaultFillerParams.frameRate);
    groupMand.add_option("--numZmw").type_int().set_default(8388608);
    groupMand.add_option("--zmwLaneWidth").type_int().set_default(64);
    groupMand.add_option("--numBlocks").type_int().set_default(10);
    groupMand.add_option("--blockLength").type_int().set_default(64);
    groupMand.add_option("--kernelLanes").type_int().set_default(2048*2);
    groupMand.add_option("--simulKernels").type_int().set_default(2);
    groupMand.add_option("--immediateCopy").type_bool().action_store_true().set_default(false);
    groupMand.add_option("--filterMode").type_string().set_default(BaselineFilterMode::toString(BaselineFilterMode::MultipleFull));
    parser.add_option_group(groupMand);

    auto pfOpts = PacBio::Process::OptionGroup(parser, "Picket fence optional parameters");
    pfOpts.add_option("--numSignals").type_int().set_default(defaultDataParams.numSignals);
    pfOpts.add_option("--pulseWidth").type_int().set_default(defaultDataParams.pulseWidth);
    pfOpts.add_option("--pulseIpd").type_int().set_default(defaultDataParams.pulseIpd);
    pfOpts.add_option("--pwRate").type_float().set_default(defaultDataParams.pulseWidthRate);
    pfOpts.add_option("--ipdRate").type_float().set_default(defaultDataParams.pulseIpdRate);
    pfOpts.add_option("--generatePoisson").type_bool().action_store_true().set_default(defaultDataParams.generatePoisson);
    pfOpts.add_option("--baselineLevel").type_int().set_default(defaultDataParams.baselineSignalLevel);
    pfOpts.add_option("--baselineStd").type_int().set_default(defaultDataParams.baselineSigma);
    pfOpts.add_option("--pulseLevels").action_append();
    pfOpts.add_option("--validate").type_bool().action_store_true().set_default(defaultDataParams.validate);
    parser.add_option_group(pfOpts);

    auto trOpts = PacBio::Process::OptionGroup(parser, "Trace file optional parameters");
    trOpts.add_option("--inputTraceFile").set_default("");
    parser.add_option_group(trOpts);

    PacBio::Process::Values options = parser.parse_args(argc, argv);

    const float frameRate = options.get("frameRate");
    const size_t numZmw = options.get("numZmw");
    const size_t zmwLaneWidth = options.get("zmwLaneWidth");
    const size_t numBlocks = options.get("numBlocks");
    const size_t blockLength = options.get("blockLength");
    const size_t kernelLanes = options.get("kernelLanes");
    const size_t simulKernels = options.get("simulKernels");
    const bool immediateCopy = options.get("immediateCopy");
    const BaselineFilterMode filterMode(options["filterMode"]);

    if (numZmw % zmwLaneWidth != 0) throw PBException("numZmw must be evenly divisible by zmwLaneWidth");
    const size_t numZmwLanes = numZmw / zmwLaneWidth;

    auto params = DataManagerParams()
            .ZmwLaneWidth(zmwLaneWidth)
            .ImmediateCopy(immediateCopy)
            .FrameRate(frameRate)
            .NumZmwLanes(numZmwLanes)
            .KernelLanes(kernelLanes)
            .NumBlocks(numBlocks)
            .BlockLength(blockLength);

    const size_t numSignals = options.get("numSignals");
    const uint16_t pulseWidth = options.get("pulseWidth");
    const uint16_t pulseIpd = options.get("pulseIpd");
    const float pulseWidthRate = options.get("pwRate");
    const float pulseIpdRate = options.get("ipdRate");
    const bool generatePoisson = options.get("generatePoisson");
    const short baselineLevel = options.get("baselineLevel");
    const short baselineStd = options.get("baselineStd");
    const bool validate = options.get("validate");

    if (numSignals > PicketFenceParams::MaxSignals)
        throw PBException("numSignals > " + std::to_string(PicketFenceParams::MaxSignals) + " specified");

    auto pfParams = PicketFenceParams()
            .NumSignals(numSignals)
            .PulseWidth(pulseWidth)
            .PulseIpd(pulseIpd)
            .PulseWidthRate(pulseWidthRate)
            .PulseIpdRate(pulseIpdRate)
            .GeneratePoisson(generatePoisson)
            .BaselineSignalLevel(baselineLevel)
            .BaselineSigma(baselineStd)
            .Validate(validate);

    if (options.is_set_by_user("pulseLevels"))
        pfParams.PulseSignalLevels(options.all("pulseLevels"));

    const std::string traceFileName = options["inputTraceFile"];

    auto trParams = TraceFileParams()
            .TraceFileName(traceFileName);

    run(params, pfParams, trParams, simulKernels, filterMode);
}
