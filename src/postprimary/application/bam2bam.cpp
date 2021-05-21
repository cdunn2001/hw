// Copyright (c) 2014-2016, Pacific Biosciences of California, Inc.
//
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

#include <algorithm>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <pbbam/BamFile.h>

#include <pacbio/process/OptionParser.h>

#include "ConvertBam2Bam.h"
#include "UserParameters.h"

#include <git-rev.h>

const std::string versionString = cmakeGitBranch() + "_" + cmakeGitHash();

using namespace PacBio::Primary::Postprimary;

int32_t ConvertString2Int(const std::string& str)
{
    std::stringstream ss(str);
    int32_t x;
    if (! (ss >> x))
    {
        std::cerr << "Error converting " << str << " to integer" << std::endl;
        abort();
    }
    return x;
}

std::vector<std::string> SplitStringToArray(const std::string& str, char splitter)
{
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string temp;
    while (std::getline(ss, temp, splitter)) // split into new "lines" based on character
    {
        tokens.push_back(temp);
    }
    return tokens;
}


std::vector<int32_t> ParseData(const std::string& data)
{
    std::vector<std::string> tokens = SplitStringToArray(data, ',');

    std::vector<int32_t> result;
    for (std::vector<std::string>::const_iterator it = tokens.begin(), end_it = tokens.end(); it != end_it; ++it)
    {
        const std::string& token = *it;
        std::vector<std::string> range = SplitStringToArray(token, '-');
        if (range.size() == 1)
        {
            result.push_back(ConvertString2Int(range[0]));
        }
        else if (range.size() == 2)
        {
            int32_t start = ConvertString2Int(range[0]);
            int32_t stop = ConvertString2Int(range[1]);
            for (int32_t i = start; i <= stop; i++)
            {
                result.push_back(i);
            }
        }
        else
        {
            std::cerr << "Error parsing token " << token << std::endl;
            abort();
        }
    }

    return result;
}

// Entry point
int main(int argc, char* argv[])
{
    try
    {
        auto parser = PacBio::Process::OptionParser().description(
            "bam2bam operates on BAM files in one convention (subreads+scraps "
            "or hqregions+scraps), allows reprocessing (for "
            "example, with a different set of adapter sequences than was "
            "originally used) and then outputs the resulting BAM files in "
            "the desired convention (subreads/hqregions/zmws plus scraps).\n\n"
            "\"Scraps\" BAM files are always required to reconstitute the ZMW reads internally. "
            "Conversely, \"scraps\" BAM files will be output.\n\n"
            "ZMW reads are not allowed as input, due to the missing HQ-region annotations!\n\n"
            "Input read convention is determined from the READTYPE "
            "annotation in the @RG::DS tags of the input BAM files."
            "A subreadset *must* be used as input instead of the individual BAM files.")
        .usage("-s subreadset.xml -o outputPrefix [options]")
        .epilog("Example: bam2bam -s subreadset.xml -o out --adapters adapters.fasta\n");

        parser.version(versionString);

        auto groupMand = PacBio::Process::OptionGroup(parser, "Mandatory parameters");
        groupMand.add_option("-o").dest("output").metavar("STRING").help("Prefix of output filenames");
        groupMand.add_option("-s", "--subreadset").dest("subreadset").metavar("STRING").help("Input subreadset.xml");
        parser.add_option_group(groupMand);

        auto groupOpt = PacBio::Process::OptionGroup(parser, "Optional parameters");
        groupOpt.add_option("-j", "--nProcs").type_int().set_default(1).dest("threads").help(
            "Number of threads for parallel ZMW processing");
        groupOpt.add_option("-b --bamThreads").type_int().set_default(4).dest("bamthreads").help(
                "Number of threads for parallel BAM compression, can only be set when not generating pbindex inline with --inlinePbi");
        groupOpt.add_option("--inlinePbi").action_store_true().help(
                "Generate pbindex inline with BAM writing");
        groupOpt.add_option("--silent").action_store_true().help("No progress output.");
        parser.add_option_group(groupOpt);

        auto groupConv =  PacBio::Process::OptionGroup(parser, "BAM conventions");
        groupConv.add_option("--zmw", "--polymerase").action_store_true().dest("zmw").help(
            "Create ZMW reads.");
        groupConv.add_option("--hqregion").action_store_true().dest("hqonly").help(
            "Output *.hqregions.bam and *.scraps.bam.");
        parser.add_option_group(groupConv);

        auto groupAdapter =
            PacBio::Process::OptionGroup(parser, "Adapter finding parameters");
        groupAdapter.add_option("--disableAdapterFinding").action_store_true();
        groupAdapter.add_option("--adapters").metavar("adapterSequences.fasta");
        groupAdapter.add_option("--globalAlnFlanking").action_store_true();
        groupAdapter.add_option("--flankLength").type_int().set_default(-1);
        groupAdapter.add_option("--minSoftAccuracy").type_float().set_default(-1.0);
        groupAdapter.add_option("--minHardAccuracy").type_float().set_default(-1.0);
        groupAdapter.add_option("--minFlankingScore").type_float().set_default(-1.0);
        groupAdapter.add_option("--disableSensitiveCorrection").action_store_true();
        groupAdapter.add_option("--disableAdapterCorrection").action_store_true();
        groupAdapter.add_option("--lookForLoop").action_store_true().help(PacBio::Process::SUPPRESS_HELP);
        std::vector<std::string> boolChoices{"True", "False"};
        groupAdapter.add_option("--enableBarcodedAdapters").choices(boolChoices.begin(), boolChoices.end())
                    .set_default("True").help(
                "True: look for full and stemless (loop-only) adapters. "
                "False: look for full adapter sequence only. "
                "Default True if not also specifying --adapters");
        groupAdapter.add_option("--stemLength").type_int().set_default(8);
#ifdef DIAGNOSTICS
        groupAdapter.add_option("--adpqc").action_store_true();
#endif
        parser.add_option_group(groupAdapter);

        auto groupControlSequences =
            PacBio::Process::OptionGroup(parser, "Parameters needed for control sequence filtering --\n"
                                          "  specifying this flag enables control sequence filtering");
        groupControlSequences.add_option("--disableControlFiltering").action_store_true();
        groupControlSequences.add_option("--controls").dest("control").metavar("controlSequences.fasta");
        groupControlSequences.add_option("--controlAdapters").dest("controlAdapters").metavar("controlAdapterSequences.fasta");
        groupControlSequences.add_option("--noSplitControlWorkflow").action_store_true().help("Use old control workflow that uses full reads");
        groupControlSequences.add_option("--splitControlLength").type_int().set_default(10000).help("split read length");
        groupControlSequences.add_option("--splitControlRefLength").type_int().set_default(15000).help("split control reference length");
        groupControlSequences.add_option("--unlabelControls").action_store_true().help(
            "Revert previous control filtering");
#ifdef DIAGNOSTICS
        groupControlSequences.add_option("--ctrlqc").action_store_true();
#endif
        parser.add_option_group(groupControlSequences);

        auto groupF =  PacBio::Process::OptionGroup(parser, "Additional output read types");
        groupF.add_option("--fasta").action_store_true().help("Output fasta.gz");
        groupF.add_option("--fastq").action_store_true().help("Output fastq.gz");
        groupF.add_option("--noBam").action_store_true().help("Do NOT produce BAM outputs.");
        groupF.add_option("--noScraps").action_store_true().help("Do NOT store scrap bam file");
        groupF.add_option("--noPbi").action_store_true().help("Do NOT generate pbindex files");
        parser.add_option_group(groupF);

        auto groupTuning =  PacBio::Process::OptionGroup(parser, "Fine tuning");
        groupTuning.add_option("--minAdapterScore").
            metavar("int").help("Minimal score for an adapter").set_default(20);
            groupTuning.add_option("--minZmwLength").type_int().set_default(1).help(
                "Minimal ZMW read length. Default: %default");
        groupTuning.add_option("--minSubLength").type_int().set_default(100).dest("minSubreadLength").help(
            "Minimal subread length. Default: %default");
        groupTuning.add_option("--fullHQ").action_store_true().help("Disable HQRF; entire ZMW read will be deemed 'HQ'. Disables --minSnr filtering.");
        groupTuning.add_option("--minSnr").type_float().help(
            "Minimal SNR across channels. Default is platform specific");
        parser.add_option_group(groupTuning);

        auto groupWhite =  PacBio::Process::OptionGroup(parser, "White list");
        groupWhite.add_option("--whitelistZmwNum").dest("number").metavar("RANGES").type_string().help("Only process given ZMW NUMBERs");
        parser.add_option_group(groupWhite);

        PacBio::Process::Values options = parser.parse_args(argc, argv);
        std::vector<std::string> args = parser.args();

        bool problem = false;

        // Store parameters
        auto user = std::make_shared<UserParameters>(argc,argv);

        if (args.size() != 0)
        {
            std::cerr << "ERROR: Must specify input using options!" << std::endl;
            problem = true;
        }

        user->subreadsetFilePath = options["subreadset"];
        user->outputPrefix = options["output"];

        if (user->outputPrefix.empty())
        {
            std::cerr << "ERROR: No output provided -o" << std::endl;
            problem = true;
        }

        if (user->subreadsetFilePath.empty())
        {
            std::cerr << "ERROR: No subreadset provided -s, --subreadset" << std::endl;
            problem = true;
        }

        user->minSubreadLength = (int)options.get("minSubreadLength");

        user->threads = (int)options.get("threads");

        if (options.is_set_by_user("bamthreads"))
        {
            if (!options.is_set_by_user("inlinePbi"))
                user->bamthreads = (int) options.get("bamthreads");
            else
            {
                std::cerr << "Cannot specify -b --bamThreads if specifying --inlinePbi" << std::endl;
                return 1;
            }
        }

        user->minPolymerasereadLength = (int)options.get("minZmwLength");

        user->adaptersFilePath = options["adapters"];
        user->disableAdapterFinding = options.get("disableAdapterFinding");
        if (user->disableAdapterFinding)
        {
            std::cout << "Disabling adapter finding" << std::endl;
            user->adaptersFilePath = "";
        }
        if (options.is_set_by_user("adapters") && !options.is_set_by_user("enableBarcodedAdapters"))
        {
            std::cerr << "ERROR: Adapters specified but not what to do with them. "
                      << "Please provide \"--enableBarcodedAdapters=[True|False]\""
                      << std::endl;
            return 1;
        }

        user->saveControls = !options.get("unlabelControls");
        user->controlFilePath = options["control"];
        user->controlAdaptersFilePath = options["controlAdapters"];
        if (options.is_set_by_user("disableControlFiltering") || !user->saveControls)
        {
            user->disableControlFiltering = true;
            std::cout << "Disabling control filtering" << std::endl;
            user->controlFilePath = "";
            user->controlAdaptersFilePath = "";
        }
        user->useSplitControlWorkflow = !options.get("noSplitControlWorkflow");
        user->splitReadLength = (int)options.get("splitControlLength");
        user->splitReferenceLength = (int)options.get("splitControlRefLength");
#ifdef DIAGNOSTICS
        user->emitControlMetrics = options.get("ctrlqc");
#endif
        if (!user->useSplitControlWorkflow &&
            (options.is_set_by_user("splitControlLength") ||
             options.is_set_by_user("splitControlRefLength")))
        {
            std::cout << "--noSplitControlWorkflow specified but split control workflow parameters were set" << std::endl;
        }

        // Must specify both control and control adapters.
        bool onlyControlFile = (!user->controlFilePath.empty() && user->controlAdaptersFilePath.empty());
        bool onlyControlAdaptersFile = (user->controlFilePath.empty() && !user->controlAdaptersFilePath.empty());
        if (onlyControlFile || onlyControlAdaptersFile)
        {
            std::cerr << "ERROR: Must specify both control and control adapter sequences for control filtering" << std::endl;
            problem = true;
        }

        user->polymeraseread = options.get("zmw");
        user->hqonly = options.get("hqonly");
        user->nobam = options.get("noBam");
        user->saveScraps = !options.get("noScraps");
        user->savePbi = !options.get("noPbi");

        user->flankLength = options.get("flankLength");
        user->localAlnFlanking = !options.get("globalAlnFlanking");
        user->minSoftAccuracy = options.get("minSoftAccuracy");
        user->minHardAccuracy = options.get("minHardAccuracy");
        user->minFlankingScore = options.get("minFlankingScore");
        user->sensitiveCorrector = !options.get("disableSensitiveCorrection");
        user->correctAdapters = !options.get("disableAdapterCorrection");
        user->lookForLoop = options.get("lookForLoop");
        user->trimToLoop = std::string(options.get("enableBarcodedAdapters")) == "True" ? true : false;
        if (user->lookForLoop || user->trimToLoop)
        {
            user->stemLength = options.get("stemLength");
        }
#ifdef DIAGNOSTICS
        user->emitAdapterMetrics = options.get("adpqc");
#endif
        user->hqrf = !options.get("fullHQ");

        if (user->hqrf)
        {
            if (options.is_set_by_user("minSnr"))
                user->minSnr = (float) options.get("minSnr");
            else
                user->minSnr = std::numeric_limits<float>::quiet_NaN();
        }
        else
            user->minSnr = 0;


        user->outputFasta = options.get("fasta");
        user->outputFastq = options.get("fastq");

        user->silent = options.get("silent");

        if (user->hqonly && user->polymeraseread)
        {
            problem = true;
            std::cerr << "Options --hqonly and --zmw are mutually exclusive." << std::endl;
        }

        if (!options["number"].empty())
            user->whiteListHoleNumbers = ParseData(options["number"]);

        if (problem)
        {
            std::cerr << "Please see --help" << std::endl;
            return 1;
        }

        user->runReport = true;

        // Start with command line parameters
        ConvertBam2Bam bam2bam(user);
        return 0;

    }
    catch(const std::exception& ex)
    {
        // PBLOG_FATAL << "Exception caught: " << ex.what();
        std::cerr << "Exception caught: " << ex.what() << std::endl;
        return 1;
    }
}
