// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted (subject to the limitations in the
// disclaimer below) provided that the following conditions are met:
//
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
//  * Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
//  * Neither the name of Pacific Biosciences nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE
// GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY PACIFIC
// BIOSCIENCES AND ITS CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL PACIFIC BIOSCIENCES OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
// USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
// OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGE.

// Programmer: Armin Töpfer

#include <iostream>
#include <string>

#include <pacbio/process/OptionParser.h>

#include <bazio/BazCore.h>
#include <bazio/file/FileHeaderBuilder.h>

#include <BazVersion.h>

#include "Simulation.h"
#include "SimulateConfigs.h"
#include "SimulationFromFasta.h"

using namespace PacBio::Primary;
using namespace PacBio::Primary::Postprimary;

// Entry point
int main(int argc, char* argv[])
{
    auto parser = PacBio::Process::OptionParser().description(
        "simbazwriter simulates base calls and writes a BAZ intermediate "
        "base file. Bases can come from a ground truth fasta file or be randomly generated.");

    auto groupMand = PacBio::Process::OptionGroup(parser, "Mandatory parameters");
    groupMand.add_option("-o")
        .dest("output")
        .help("BAZ output filename")
        .metavar("STRING");
    parser.add_option_group(groupMand);

    auto groupOpt = PacBio::Process::OptionGroup(parser, "Optional parameters");
    groupOpt.add_option("-v", "--version")
        .dest("version")
        .action_store_true()
        .help("Print the tool version and exit");
    groupOpt.add_option("-f")
        .dest("fasta")
        .help("Fasta input filename")
        .metavar("STRING");
    //groupOpt.add_option("-F")
    //    .dest("future")
    //    .action_store_true()
    //    .help("Adds an unsupported packet field (Forward compatibility test)");
    groupOpt.add_option("-z")
        .action("store")
        .type_int()
        .set_default(1000000)
        .dest("zmws")
        .help("Number of ZMWs to simulate. Default: %default");
    groupOpt.add_option("-b","--bps")
        .type_double()
        .set_default(5.0)
        .dest("bps")
        .help("Polymerase base pair rate per second. Default: %default");
    groupOpt.add_option("-s")
        .action("store")
        .type_int()
        .set_default(163)
        .dest("seconds")
        .help("Chunk length in seconds. Default: %default");
    groupOpt.add_option("-c")
        .action("store")
        .type_int()
        .set_default(66)
        .dest("chunks")
        .help("Chunks per movie. Default: %default");
    groupOpt.add_option("-r").action_store_true().dest("rtal").help("RTAL BAZ layout.");
    groupOpt.add_option("-p").action_store_true().dest("internal").help("Internal pulse mode.");
    groupOpt.add_option("-n").action_store_true().dest("noMetrics").help("No metrics mode.");
    groupOpt.add_option("-t").action_store_true().dest("trivial").help("Const data.");
    groupOpt.add_option("--silent").action_store_true().dest("silent").help("No progress output.");
    groupOpt.add_option("--sizeBases").type_int().dest("sizeBases").help("Output expected BAZ size for movie length in frames.");
    groupOpt.add_option("--sizePulses").type_int().dest("sizePulses").help("Output expected BAZ size for movie length in frames.");
    groupOpt.add_option("--summarize").action_store_true().help("display estimates and actual file sizes");
    parser.add_option_group(groupOpt);

    PacBio::Process::Values options = parser.parse_args(argc, argv);

    // Print version
    if (options.get("version"))
    {
        std::cerr << "simbazwriter version: " << BAZIO_VERSION << std::endl;
        return 0;
    }

    bool internal = static_cast<bool>(options.get("internal"));
    std::string basecallerConfig = generateBasecallerConfig(internal);
    std::string experimentMetadata = generateExperimentMetadata();

    // Set #zmw
    size_t zmws = (int)options.get("zmws");

    std::vector<uint32_t> zmwNumbers;
    {
        for (size_t z = 0; z < zmws; z++) zmwNumbers.push_back(z);
    }

    using FileHeaderBuilder = PacBio::BazIO::FileHeaderBuilder;

    constexpr uint32_t framesPerMetricBlock = 1024;
    if ((int)options.get("sizeBases") > 0)
    {
        uint32_t frames = (int)options.get("sizeBases");

        FileHeaderBuilder fhb("m00001_052415_013000", 100.0, frames,
                              PacBio::BazIO::ProductionPulses::Params(),
                              experimentMetadata,
                              basecallerConfig,
                              Simulation::SimulateZmwInfo(zmwNumbers),
                              framesPerMetricBlock,
                              FileHeaderBuilder::Flags().RealTimeActivityLabels(true));

        PBLOG_ERROR << "Expected file size based on file header not yet implemented!";

//        std::cerr << "BAZ size: "
//            << fhb.ExpectedFileByteSize(2.5)
//            << std::endl;
        return 0;
    }
    if ((int)options.get("sizePulses") > 0)
    {
        std::vector<uint32_t> dummyNumbers(zmws);
        std::vector<uint32_t> emptyList;
        uint32_t frames = (int)options.get("sizePulses");

        FileHeaderBuilder fhb("m00001_052415_013000", 100.0f, frames,
                              PacBio::BazIO::InternalPulses::Params() ,
                              experimentMetadata,
                              basecallerConfig,
                              Simulation::SimulateZmwInfo(zmwNumbers),
                              framesPerMetricBlock,
                              FileHeaderBuilder::Flags().RealTimeActivityLabels(true));

        PBLOG_ERROR << "Expected file size based on file header not yet implemented!";

//        std::cerr << "BAZ size: "
//                << fhb.ExpectedFileByteSize(2.5)
//                << std::endl;
        return 0;
    }


    // Set bp/s
    double bps = options.get("bps");
    if (bps == 0) bps = 5;

    // Set seconds per chunk
    int seconds = (int)options.get("seconds");
    if (seconds == 0) seconds = 163;

    // Set chunks per movie
    int chunks = (int)options.get("chunks");
    if (chunks == 0) chunks = 66;

    // set output file name
    std::string filename = options["output"];

    // set input file name
    std::string fasta = options["fasta"];

    bool silent = options.get("silent");

    if (filename.empty())
    {
        std::cerr << "Please see --help" << std::endl;
        return 1;
    }

    //if (options.get("future"))
    //{
    //    SimulationFuture(filename, chipLayoutName, zmwNumbers, zmws, bps, seconds, chunks, silent);
    //    return 0;
    //}

    Simulation sim(filename, zmwNumbers, zmws, bps, seconds, chunks, silent);
    sim.Summarize((bool)options.get("summarize"));

    if (fasta.empty())
    {
        if (options.get("trivial"))
        {
            sim.SimulateTrivial();
        }
        else if (options.get("noMetrics"))
        {
            if (options.get("internal"))
                sim.SimulatePulsesNoMetrics();
            else
                sim.SimulateBasesNoMetrics();
        }
        else
        {
            if (options.get("internal"))
                sim.SimulatePulses();
            else
                sim.SimulateBases();
        }
    }
    else
    {
        using PacBio::SmrtData::Readout;
        Readout readout = internal ? Readout::PULSES : Readout::BASES_WITHOUT_QVS;

        if (options.is_set_by_user("zmws"))
            SimulationFromFasta(filename, fasta, readout, silent, zmwNumbers, options.get("rtal"), (int)options.get("zmws"));
        else
            SimulationFromFasta(filename, fasta, readout, silent, zmwNumbers, options.get("rtal"));
    }
    return 0;
}
