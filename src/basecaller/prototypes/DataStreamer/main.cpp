#include <pacbio/process/OptionParser.h>
#include <common/ZmwDataManager.h>

#include <StreamingTestRunner.h>


using namespace PacBio::Cuda::Memory;
using namespace PacBio::Cuda::Data;
using namespace PacBio::Cuda;

int main(int argc, char* argv[])
{
    DataManagerParams defaultFillerParams;

    auto parser = PacBio::Process::OptionParser().description(
        "Prototype to demonstrate streaming data transfer to gpu");
    parser.version("0.1");

    auto groupMand = PacBio::Process::OptionGroup(parser, "Mandatory parameters");
    groupMand.add_option("--frameRate").type_float().set_default(defaultFillerParams.frameRate);
    groupMand.add_option("--numZmwLanes").type_int().set_default(defaultFillerParams.numZmwLanes);
    groupMand.add_option("--laneWidth").type_int().set_default(defaultFillerParams.laneWidth);
    groupMand.add_option("--numBlocks").type_int().set_default(defaultFillerParams.numBlocks);
    groupMand.add_option("--blockLength").type_int().set_default(defaultFillerParams.blockLength);
    groupMand.add_option("--kernelLanes").type_int().set_default(defaultFillerParams.kernelLanes);
    groupMand.add_option("--simulKernels").type_int().set_default(2);
    groupMand.add_option("--immediateCopy").type_bool().action_store_true().set_default(defaultFillerParams.immediateCopy);
    parser.add_option_group(groupMand);

    PacBio::Process::Values options = parser.parse_args(argc, argv);

    const float frameRate = options.get("frameRate");
    const size_t numZmwLanes = options.get("numZmwLanes");
    const size_t laneWidth = options.get("laneWidth");
    const size_t numBlocks = options.get("numBlocks");
    const size_t blockLength = options.get("blockLength");
    const size_t kernelLanes = options.get("kernelLanes");
    const size_t simulKernels = options.get("simulKernels");
    const bool immediateCopy = options.get("immediateCopy");

    auto params = DataManagerParams()
            .ImmediateCopy(immediateCopy)
            .FrameRate(frameRate)
            .NumZmwLanes(numZmwLanes)
            .LaneWidth(laneWidth)
            .KernelLanes(kernelLanes)
            .NumBlocks(numBlocks)
            .BlockLength(blockLength);

    RunTest(params, simulKernels);
}
