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
#include <limits>

#include <pacbio/process/OptionParser.h>

#include <pacbio/primary/HQRFMethod.h>

#include "ConvertBaz2Bam.h"
#include "UserParameters.h"

#include <git-rev.h>

static const std::string versionString = cmakeGitBranch() + "_" + cmakeGitHash();

using namespace PacBio::Primary::Postprimary;

static uint32_t ConvertString2Int(const std::string& str)
{
    std::stringstream ss(str);
    uint32_t x;
    if (! (ss >> x))
    {
        std::cerr << "Error converting " << str << " to integer" << std::endl;
        abort();
    }
    return x;
}

static std::vector<std::string> SplitStringToArray(const std::string& str, char splitter)
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

static std::vector<uint32_t> ParseData(const std::string& data)
{
    std::vector<std::string> tokens = SplitStringToArray(data, ',');

    std::vector<uint32_t> result;
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
            uint32_t start = ConvertString2Int(range[0]);
            uint32_t stop = ConvertString2Int(range[1]);
            for (uint32_t i = start; i <= stop; i++)
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

static std::vector<std::string> ReadFileList(const std::string& fileListPath)
{
    std::vector<std::string> inputFilePaths;
    std::ifstream in(fileListPath);
    std::string filePath;
    while (std::getline(in, filePath))
        inputFilePaths.push_back(filePath);
    return inputFilePaths;
}

// Entry point
int main(int argc, char* argv[])
{
    try
    {
        auto parser = PacBio::Process::OptionParser().description(
            "baz2bam converts the intermediate BAZ file format to BAM, FASTA, AND FASTQ."
            "\n\n"
            + cmakeGitHash() + " "
            + cmakeGitCommitDate());
        parser.usage("\n"
                     "-o outputPrefix -m rmd.xml [options] input.baz"
                     "\n"
                     "-o outputPrefix -s subreadset.xml [options] input.baz");

        parser.version(versionString);

        auto groupMand = PacBio::Process::OptionGroup(parser, "Mandatory parameters");
        groupMand.add_option("-o").dest("output").metavar("STRING").help("Prefix of output filenames");
        groupMand.add_option("-m", "--metadata").dest("rmd").metavar("STRING").help("Runtime meta data filename");
        groupMand.add_option("-s", "--subreadset").dest("subreadset").metavar("STRING").help("Input subreadset.xml");
        parser.add_option_group(groupMand);

        auto groupOpt = PacBio::Process::OptionGroup(parser, "Optional parameters");
        groupOpt.add_option("--logoutput").dest("logoutput").help("Log output filename. Default <output_prefix>.log");
        std::vector<std::string> loglevels = { "TRACE", "DEBUG", "INFO", "NOTICE", "WARN", "ERROR", "CRITICAL", "FATAL" };
        groupOpt.add_option("--logfilter").dest("logfilter").type_choice().choices(loglevels).set_default("INFO").help("Log level to filter on. Default INFO");
        groupOpt.add_option("--filelist").dest("filelist").help("Text file containing full paths to BAZ files, one per line");
        groupOpt.add_option("--uuid").help("If this is specified, it must"
                                            " compare to the UUID of the subreadset within the metadata XML. pa-ws "
                                            " will set this option for sanity checking.  baz2bam will return "
                                            " with an error if there is a mismatch.");
        groupOpt.add_option("--statusfd").type_int().set_default(-1).help("Write status messages to this file description. Default -1 (null)");
        groupOpt.add_option("-j --nProcs").type_int().set_default(1).dest("threads").help(
            "Number of threads for parallel ZMW processing");
        groupOpt.add_option("-b --bamThreads").type_int().set_default(4).dest("bamthreads").help(
            "Number of threads for parallel BAM compression. Note due to technical reasons, out-of-line pbi indexing "
            "will use 2x these threads (one set for each bam file), and inline pbi indexing will use 4x (inline indexing "
            "requires a separate and simultaneous decompressing read)");
        groupOpt.add_option("--inlinePbi").action_store_true().help(
            "Generate pbindex inline with BAM writing");
        groupOpt.add_option("--silent").action_store_true().help("No progress output.");
        groupOpt.add_option("-Q").dest("fakeHQ").metavar("STRING").help("Fake HQ filename [simulation file only]");
        groupOpt.add_option("--streamedBam").action_store_true().help("Stream uncompressed BAM data to stdout. This option will force "
            "--noScraps, --noPbi, and --silent. Output prefix (-o) will still be used for naming all other output files.");
        parser.add_option_group(groupOpt);

        auto groupHQ = PacBio::Process::OptionGroup(
            parser, "HQRegion only mode --\n"
                    "  specifying this flag disables ZMW output and activates hqregion output.");
        groupHQ.add_option("--hqregion").action_store_true().dest("hqonly").help(
            "Output *.hqregions.bam and *.lqregions.bam.");
        parser.add_option_group(groupHQ);

        auto groupAdapter = PacBio::Process::OptionGroup(
            parser, "Adapter finding parameters --\n"
                    "  specifying this flag enables adapter finding and subread output");
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

        auto groupInsert = PacBio::Process::OptionGroup(
            parser, "Insertion classifier parameter --\n"
                    "  specifying this flag enables insertion filtration where some bases will be removed from reads");
        groupInsert.add_option("--noClassifyInsertions").action_store_true().help(
            "Disable insertion classifier");
        groupInsert.add_option("--minInsertionLength").type_int().set_default(50).help(
            "Minimum insertion length to be considered for removal. Default: %default"
        );
        parser.add_option_group(groupInsert);

        auto groupControlSequences =
            PacBio::Process::OptionGroup(parser, "Control sequence filtering parameter --\n"
                                          "  specifying this flag enables control sequence filtering, must specify control and control adapter sequences");
        groupControlSequences.add_option("--disableControlFiltering").action_store_true();
        groupControlSequences.add_option("--controls").dest("control").metavar("controlSequences.fasta");
        groupControlSequences.add_option("--controlAdapters").dest("controlAdapters").metavar("controlAdapterSequences.fasta");
        groupControlSequences.add_option("--noSplitControlWorkflow").action_store_true().help("Use old control workflow that uses full reads");
        groupControlSequences.add_option("--splitControlLength").type_int().set_default(10000).help("split read length");
        groupControlSequences.add_option("--splitControlRefLength").type_int().set_default(15000).help("split control reference length");
#ifdef DIAGNOSTICS
        groupControlSequences.add_option("--ctrlqc").action_store_true();
#endif
        parser.add_option_group(groupControlSequences);

        auto groupF = PacBio::Process::OptionGroup(parser, "Additional output read types");
        groupF.add_option("--fasta").action_store_true().help("Output fasta.gz");
        groupF.add_option("--fastq").action_store_true().help("Output fastq.gz");
        groupF.add_option("--noBam").action_store_true().help("Do NOT store bam files");
        groupF.add_option("--noScraps").action_store_true().help("Do NOT store scrap bam file");
        groupF.add_option("--noPbi").action_store_true().help("Do NOT generate pbindex files");
        parser.add_option_group(groupF);

        auto groupTuning = PacBio::Process::OptionGroup(parser, "Fine tuning");
        groupTuning.add_option("--minAdapterScore").
            metavar("INT").help("Minimal score for an adapter. Default: %default").set_default(20);
        groupTuning.add_option("--minZmwLength").type_int().set_default(1).help(
            "Minimal ZMW read length. Default: %default");
        groupTuning.add_option("--minSubLength").type_int().set_default(100).dest("minSubreadLength").help(
            "Minimal subread length. Default: %default");
        groupTuning.add_option("--noStats").action_store_true().help(
                "Do NOT compute stats. Also disables sts.h5 file generation");
        groupTuning.add_option("--noStsH5").action_store_true().dest("noStsH5").help(
                "Do not output per zmw stats to sts.h5 file");
        groupTuning.add_option("--diagStsH5").action_store_true().dest("diagStsH5").help(
                "Output additional diagnostic datasets to the sts.h5");
        groupTuning.add_option("--fullHQ").action_store_true().help(
                "Disable HQRF; entire ZMW read will be deemed 'HQ'. Disables --minSnr filtering.");
        //groupTuning.add_option("--polymerase").action_store_true().dest("zmw").help(
        //        "Create ZMW reads.");

        groupTuning.add_option("--minSnr").type_float().help(
            "Minimal SNR across channels. Default: Sequel = " + std::to_string(UserParameters::minSnrSequel) +
            ", Spider = " + std::to_string(UserParameters::minSnrSpider));

        std::vector<std::string> hqrfChoices;
        std::string hqrfChoiceString;
        for (const auto& s : PacBio::Primary::HqrfPublicMethod::allValuesAsStrings())
        {
            if (hqrfChoiceString.size())
                hqrfChoiceString += ", ";
            hqrfChoiceString += s;
            hqrfChoices.push_back(s);
        }
        groupTuning.add_option("--hqrfConfig").choices(hqrfChoices).help(
                "Choose an HQRF configuration from [" + hqrfChoiceString + "]");
        groupTuning.add_option("--ignoreBazActivityLabels").action_store_true().help(
                "Ignore HQRF Activity Labels found in the baz file");
        groupTuning.add_option("--minEmptyTime").type_int().set_default(60).help(
            "Minimum quiet start time (minutes) for a hole to be considered "
            "empty. Set to 0 to use the length of the movie. Default: %default");
        groupTuning.add_option("--emptyOutlierTime").type_int().set_default(3).help(
            "The total time (minutes) over which the empty classifier will forgive "
            "or ignore excessive pulse activity in its calculation. Times are rounded "
            "to block boundaries, where blocks have a duration of 4096 frames, or "
            "approximately 51.2 sec at 80 fps acquisition. "
            "Set to 0 to use the full time. Default: %default");
        parser.add_option_group(groupTuning);

        auto groupWhite = PacBio::Process::OptionGroup(parser, "White list");
        groupWhite.add_option("--whitelistZmwNum").dest("number").metavar("RANGES").type_string().help(
                "Only process given ZMW NUMBERs. Use dashes to express ranges, and commas for more than one range.");
        groupWhite.add_option("--whitelistZmwId").dest("id").metavar("RANGES").type_string().help(
                "Only process given ZMW IDs. Use dashes to express ranges, and commas for more than one range.");
        groupWhite.add_option("--maxNumZmwsToProcess").type_int().set_default(std::numeric_limits<int32_t>::max()).help(
                "Only process this many ZMWs. An alternative to using the white list.");

        parser.add_option_group(groupWhite);

        auto groupIO = PacBio::Process::OptionGroup(parser, "IO Tuning Parameters");
        groupIO.add_option("--zmwBatchMB").type_int().set_default(1000).help("Controls"
                " how many MB of zmw data to load at once.  A larger number will give you much better"
                " disk IO, but requires more RAM and may do a lot of excess work if a sparse"
                " whitelist is specified");
        groupIO.add_option("--zmwHeaderBatchMB").type_int().set_default(30).help("Controls"
                " how many MB of header data to read at once.  It is independent of --zmwBatchMB"
                " but again a larger number gives faster reads but requires more RAM");
        groupIO.add_option("--maxInputQueueMB").type_int().set_default(10000).help("Maximum size in MB"
                " the input queue can grow to before additional reads are blocked");
        groupIO.add_option("--maxOutputQueueMB").type_int().set_default(30000).help("Maximum size in MB"
                " the output queue can grow to before additional reads are blocked");
        groupIO.add_option("--defaultZmwBatchSize").type_int().set_default(100).help(
               "Default number of zmw each compute thread will tackle at once.");
        parser.add_option_group(groupIO);

        PacBio::Process::Values options = parser.parse_args(argc, argv);
        std::vector<std::string> args = parser.args();

        bool problem = false;

        // Store parameters
        auto user = std::make_shared<UserParameters>(argc,argv);

        if (args.size() >= 1)
        {
            user->inputFilePaths.insert(std::end(user->inputFilePaths), args.begin(), args.end());
        }
        else
        {
            user->fileListPath = options["filelist"];
            if ("" == user->fileListPath)
            {
                std::cerr << "ERROR: INPUT EMPTY." << std::endl;
                problem = true;
            }
            user->inputFilePaths = ReadFileList(user->fileListPath);
            if (user->inputFilePaths.empty())
            {
                std::cerr << "ERROR: INPUT FILE LIST EMPTY." << std::endl;
                problem = true;
            }
        }

        // Mandatory parameters
        user->outputPrefix = options["output"];
        if (user->outputPrefix.empty())
        {
            std::cerr << "ERROR: OUTPUT EMPTY." << std::endl;
            problem = true;
        }

        user->runtimeMetaDataFilePath = options["rmd"];
        if (user->runtimeMetaDataFilePath.empty())
        {
            user->subreadsetFilePath = options["subreadset"];
            if (user->subreadsetFilePath.empty())
            {
                std::cerr << "WARNING: META DATA AND SUBREADSET EMPTY." << std::endl;
                problem = true;
            }
        }

        // Optional parameters
        if (options.is_set_by_user("logoutput"))
        {
            if (options["logoutput"] != "")
            {
                user->logFileName = options["logoutput"];
            }
            else
            {
                std::cerr << "ERROR: Empty log file name specified." << std::endl;
                problem = true;
            }
        }
        else
        {
            user->logFileName = user->outputPrefix + ".baz2bam.log";
        }
        user->logFilterLevel = options["logfilter"];

        user->uuid = options["uuid"];

        // Whitelist processing
        if (!options["number"].empty() && !options["id"].empty())
        {
            std::cerr << "Options --whitelistZmwNum and --whitelistZmwId are mutually exclusive" << std::endl;
            return 1;
        }

        user->statusFileDescriptor = options.get("statusfd");
        user->noStats = options.get("noStats");
        user->noStatsH5 = options.get("noStsH5");
        user->diagStatsH5 = options.get("diagStsH5");

        if (user->noStats)
        {
            user->noStatsH5 = true;
            user->diagStatsH5 = false;
        }

        user->nobam = options.get("noBam");
        user->saveScraps = !options.get("noScraps");
        user->savePbi = !options.get("noPbi");
        user->inlinePbi = options.get("inlinePbi");
        user->fakeHQ = options["fakeHQ"];

        user->threads = (int)options.get("threads");
        user->bamthreads = (int) options.get("bamthreads");

        user->minSubreadLength = (int)options.get("minSubreadLength");

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

        user->controlFilePath = options["control"];
        user->controlAdaptersFilePath = options["controlAdapters"];
        if (options.is_set_by_user("disableControlFiltering"))
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

        user->saveControls = !user->controlFilePath.empty();

        user->hqonly = options.get("hqonly");
        if (user->hqonly)
            user->polymeraseread = false;
        else
        {
            if (user->disableAdapterFinding)
                user->polymeraseread = true;
        }

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

        user->outputFasta = options.get("fasta");
        user->outputFastq = options.get("fastq");

        user->silent = options.get("silent");

        user->noClassifyInsertions = options.get("noClassifyInsertions");
        user->minInsertionLength = (int)options.get("minInsertionLength");

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

        user->ignoreBazAL = options.get("ignoreBazActivityLabels");
        if (options.is_set_by_user("hqrfConfig"))
        {
            user->hqrfMethod = options["hqrfConfig"];
        }

        if (!options["number"].empty())
            user->whiteListZmwNumbers = ParseData(options["number"]);
        else if (!options["id"].empty())
            user->whiteListZmwIds = ParseData(options["id"]);

        user->maxNumZmwsToProcess = options.get("maxNumZmwsToProcess");

        user->minEmptyTime = options.get("minEmptyTime");

        user->emptyOutlierTime = options.get("emptyOutlierTime");

        user->zmwBatchMB = options.get("zmwBatchMB");
        user->zmwHeaderBatchMB = options.get("zmwHeaderBatchMB");
        user->maxInputQueueMB = options.get("maxInputQueueMB");
        user->maxOutputQueueMB = options.get("maxOutputQueueMB");
        user->defaultZmwBatchSize = options.get("defaultZmwBatchSize");

        const bool streamedBam = options.get("streamedBam");
        if (streamedBam)
        {
            user->inlinePbi = false;
            user->streamBam = true;
            user->savePbi = false;
            user->saveScraps = false;
            user->silent = true;
        }

        if (problem)
        {
            std::cerr << "Please see --help" << std::endl;
            return 1;
        }

        ConvertBaz2Bam baz2bam(user);
        return baz2bam.Run();

    }
    catch(const std::exception& ex)
    {
        std::cerr << "Exception caught: " << ex.what() << std::endl;
        PBLOG_FATAL << "Exception caught: " << ex.what();
        return 1;
    }
    catch(const H5::Exception& ex)
    {
        std::cerr << "H5::Exception caught: " << ex.getCDetailMsg() << std::endl;
        PBLOG_FATAL << "H5::Exception caught: " << ex.getCDetailMsg();
        return 2;
    }
    catch(...)
    {
        std::cerr << "Unknown exception caught at main level" << std::endl;
        PBLOG_FATAL << "Unknown exception caught at main level";
        return 3;
    }
}
