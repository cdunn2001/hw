// Copyright (c) 2018, Pacific Biosciences of California, Inc.
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

static const std::string versionString = cmakeGitBranch() + "_" + cmakeGitHash();

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
            "recalladapters operates on BAM files in one convention (subreads+scraps "
            "or hqregions+scraps), allows reprocessing adapter calls "
            "then outputs the resulting BAM files as "
            "subreads plus scraps.\n\n"
            "\"Scraps\" BAM files are always required to reconstitute the ZMW reads internally. "
            "Conversely, \"scraps\" BAM files will be output.\n\n"
            "ZMW reads are not allowed as input, due to the missing HQ-region annotations!\n\n"
            "Input read convention is determined from the READTYPE "
            "annotation in the @RG::DS tags of the input BAM files."
            "A subreadset *must* be used as input instead of the individual BAM files.")
        .usage("-s subreadset.xml -o outputPrefix [options]")
        .epilog("Example: recalladapters -s in.subreadset.xml -o out --adapters adapters.fasta\n");

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

        auto groupAdapter =
            PacBio::Process::OptionGroup(parser, "Adapter finding parameters");
        groupAdapter.add_option("--disableAdapterFinding").action_store_true();
        groupAdapter.add_option("--adapters").metavar("adapterSequences.fasta");
        groupAdapter.add_option("--globalAlnFlanking").action_store_true();
        groupAdapter.add_option("--flankLength").type_int().set_default(-1);
        groupAdapter.add_option("--minSoftAccuracy").type_float().set_default(-1.0);
        groupAdapter.add_option("--minHardAccuracy").type_float().set_default(-1.0);
        groupAdapter.add_option("--minFlankingScore").type_float().set_default(-1.0);
        groupAdapter.add_option("--disableAdapterCorrection").action_store_true();
        groupAdapter.add_option("--lookForLoop").action_store_true().help(PacBio::Process::SUPPRESS_HELP);
        std::vector<std::string> boolChoices{"True", "False"};
        groupAdapter.add_option("--enableBarcodedAdapters").choices(boolChoices.begin(), boolChoices.end())
                    .set_default("True").help(
                "True: look for full and stemless (loop-only) adapters. "
                "False: look for full adapter sequence only. "
                "Default True if using default adapters");
        groupAdapter.add_option("--stemLength").type_int().set_default(8);
#ifdef DIAGNOSTICS
        groupAdapter.add_option("--adpqc").action_store_true();
#endif
        parser.add_option_group(groupAdapter);

        auto groupTuning =  PacBio::Process::OptionGroup(parser, "Fine tuning");
        groupTuning.add_option("--minAdapterScore").
            metavar("int").help("Minimal score for an adapter").set_default(20);
        groupTuning.add_option("--minSubLength").type_int().set_default(100).dest("minSubreadLength").help(
            "Minimal subread length. Default: %default");
        groupTuning.add_option("--minSnr").type_float().set_default(3.75).help(
            "Minimal SNR across channels. Default: %default");
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

        user->minPolymerasereadLength = 1;
        user->adaptersFilePath = options["adapters"];
        user->disableAdapterFinding = options.get("disableAdapterFinding");
        if (user->disableAdapterFinding)
        {
            std::cout << "Disabling adapter finding, setting empty adapter file path";
            user->adaptersFilePath = "";
        }
        if (options.is_set_by_user("adapters") && !options.is_set_by_user("enableBarcodedAdapters"))
        {
            std::cerr << "ERROR: Adapters specified but not what to do with them. "
                      << "Please provide \"--enableBarcodedAdapters=[True|False]\""
                      << std::endl;
            return 1;
        }
        user->saveControls = false;

        user->polymeraseread = false;
        user->hqonly = false;
        user->nobam = false;
        user->saveScraps = true;
        user->savePbi = true;

        user->flankLength = options.get("flankLength");
        user->localAlnFlanking = !options.get("globalAlnFlanking");
        user->minSoftAccuracy = options.get("minSoftAccuracy");
        user->minHardAccuracy = options.get("minHardAccuracy");
        user->minFlankingScore = options.get("minFlankingScore");
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
        user->hqrf = true;
        if (options.is_set_by_user("minSnr"))
            user->minSnr = (float) options.get("minSnr");
        else
            user->minSnr = std::numeric_limits<float>::quiet_NaN();

        user->outputFasta = false;
        user->outputFastq = false;

        user->silent = options.get("silent");

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
