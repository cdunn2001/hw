#include <pacbio/process/OptionParser.h>
#include <common/DataGenerators/PicketFenceGenerator.h>
#include <common/ZmwDataManager.h>

#include "FrameLabeler.h"

using namespace PacBio::Cuda::Memory;
using namespace PacBio::Cuda::Data;
using namespace PacBio::Cuda;
using namespace PacBio::Mongo;

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
    groupMand.add_option("--laneWidth").type_int().set_default(64);
    groupMand.add_option("--numBlocks").type_int().set_default(10);
    groupMand.add_option("--blockLength").type_int().set_default(64);
    groupMand.add_option("--kernelLanes").type_int().set_default(2048*2);
    groupMand.add_option("--simulKernels").type_int().set_default(3);
    groupMand.add_option("--immediateCopy").type_bool().action_store_true().set_default(false);
    //groupMand.add_option("--filterMode").type_string().set_default(BaselineFilterMode::toString(BaselineFilterMode::MultipleFull));
    parser.add_option_group(groupMand);

    auto groupData = PacBio::Process::OptionGroup(parser, "Picket fence optional parameters");
    groupData.add_option("--numSignals").type_int().set_default(defaultDataParams.numSignals);
    groupData.add_option("--pulseWidth").type_int().set_default(defaultDataParams.pulseWidth);
    groupData.add_option("--pulseIpd").type_int().set_default(defaultDataParams.pulseIpd);
    groupData.add_option("--pwRate").type_float().set_default(defaultDataParams.pulseWidthRate);
    groupData.add_option("--ipdRate").type_float().set_default(defaultDataParams.pulseIpdRate);
    groupData.add_option("--generatePoisson").type_bool().action_store_true().set_default(defaultDataParams.generatePoisson);
    groupData.add_option("--baselineLevel").type_int().set_default(0);
    groupData.add_option("--baselineStd").type_int().set_default(defaultDataParams.baselineSigma);
    groupData.add_option("--pulseLevels").action_append();
    groupData.add_option("--validate").type_bool().action_store_true().set_default(defaultDataParams.validate);
    parser.add_option_group(groupData);

    auto trOpts = PacBio::Process::OptionGroup(parser, "Trace file optional parameters");
    trOpts.add_option("--inputTraceFile").set_default("");
    parser.add_option_group(trOpts);

    auto groupMeta = PacBio::Process::OptionGroup(parser, "Analog metadata parameters");
    groupMeta.add_option("--analogMeans").type_float().action_append();
    groupMeta.add_option("--analogVars").type_float().action_append();
    groupMeta.add_option("--analogIpds").type_float().action_append();
    groupMeta.add_option("--analogPws").type_float().action_append();
    groupMeta.add_option("--analogSSPws").type_float().action_append();
    groupMeta.add_option("--analogSSIpds").type_float().action_append();
    groupMeta.add_option("--BaselineMean").type_float().set_default(0);
    groupMeta.add_option("--BaselineVar").type_float().set_default(33);
    parser.add_option_group(groupMeta);

    PacBio::Process::Values options = parser.parse_args(argc, argv);

    const float frameRate = options.get("frameRate");
    const size_t numZmw = options.get("numZmw");
    const size_t laneWidth = options.get("laneWidth");
    const size_t numBlocks = options.get("numBlocks");
    const size_t blockLength = options.get("blockLength");
    const size_t kernelLanes = options.get("kernelLanes");
    const size_t simulKernels = options.get("simulKernels");
    const bool immediateCopy = options.get("immediateCopy");
    //const BaselineFilterMode filterMode(options["filterMode"]);

    if (numZmw % laneWidth != 0) throw PBException("numZmw must be evenly divisible by laneWidth");
    const size_t numZmwLanes = numZmw / laneWidth;

    auto params = DataManagerParams()
            .LaneWidth(laneWidth)
            .ImmediateCopy(immediateCopy)
            .FrameRate(frameRate)
            .NumZmwLanes(numZmwLanes)
            .KernelLanes(kernelLanes)
            .NumBlocks(numBlocks)
            .BlockLength(blockLength);
    (void)params;

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

    auto dataParams = PicketFenceParams()
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
        dataParams.PulseSignalLevels(options.all("pulseLevels"));

    const std::string traceFileName = options["inputTraceFile"];

    auto trParams = TraceFileParams()
            .TraceFileName(traceFileName);

    // Defaults that match current mongo simulated trace file
    std::array<PacBio::DataSource::AnalogMode, 4> analogs;
    LaneModelParameters<PBHalf, laneSize> refModel;

    analogs[0].ipd2SlowStepRatio = 0;
    analogs[1].ipd2SlowStepRatio = 0;
    analogs[2].ipd2SlowStepRatio = 0;
    analogs[3].ipd2SlowStepRatio = 0;

    analogs[0].interPulseDistance = .308f;
    analogs[1].interPulseDistance = .234f;
    analogs[2].interPulseDistance = .234f;
    analogs[3].interPulseDistance = .188f;

    analogs[0].pulseWidth = .232f;
    analogs[1].pulseWidth = .185f;
    analogs[2].pulseWidth = .181f;
    analogs[3].pulseWidth = .214f;

    analogs[0].pw2SlowStepRatio = 3.2f;
    analogs[1].pw2SlowStepRatio = 3.2f;
    analogs[2].pw2SlowStepRatio = 3.2f;
    analogs[3].pw2SlowStepRatio = 3.2f;

    refModel.AnalogMode(0).SetAllMeans(227.13f);
    refModel.AnalogMode(1).SetAllMeans(154.45f);
    refModel.AnalogMode(2).SetAllMeans(97.67f);
    refModel.AnalogMode(3).SetAllMeans(61.32f);

    refModel.AnalogMode(0).SetAllVars(776);
    refModel.AnalogMode(1).SetAllVars(426);
    refModel.AnalogMode(2).SetAllVars(226);
    refModel.AnalogMode(3).SetAllVars(132);

    auto MetaSetter = [&](std::string option, auto&& setFunc) {
        const int numAnalogs = 4;
        if (options.is_set_by_user(option))
        {
            auto opts = options.all(option);
            if (opts.size() != numAnalogs) throw PBException("--" + option + "required 0 or 4 arguments");
            int idx = 0;
            for (auto val : opts)
            {
                setFunc(idx, PacBio::Process::Value(val));
                idx++;
            }
        }
    };

    MetaSetter("analogMeans", [&](int idx, float val) {refModel.AnalogMode(idx).SetAllMeans(val);});
    MetaSetter("analogVars", [&](int idx, float val) {refModel.AnalogMode(idx).SetAllVars(val);});
    MetaSetter("analogIpds", [&](int idx, float val) {analogs[idx].interPulseDistance = val; });
    MetaSetter("analogPws", [&](int idx, float val) {analogs[idx].pulseWidth = val; });
    MetaSetter("analogSSPws", [&](int idx, float val) {analogs[idx].pw2SlowStepRatio = val; });
    MetaSetter("analogSSIpds", [&](int idx, float val) {analogs[idx].ipd2SlowStepRatio = val; });

    refModel.BaselineMode().SetAllMeans(options.get("BaselineMean"));
    refModel.BaselineMode().SetAllVars(options.get("BaselineVar"));

    run(params, dataParams, trParams, analogs, refModel, simulKernels);
}
