// Copyright (c) 2020, Pacific Biosciences of California, Inc.
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

#include <cmath>
#include <iostream>
#include <memory>
#include <utility>

#include <pacbio/process/OptionParser.h>
#include <postprimary/adapterfinder/AdapterLabeler.h>

#include <bazio/FastaUtilities.h>
#include <bazio/FastaEntry.h>

using namespace PacBio::Primary;

std::string AdapterSequenceToName(const std::string& adp)
{
    if   (adp == "ATCTCTCTCAACAACAACAACGGAGGAGGAGGAAAAGAGAGAGAT") {
        return "TC6";
    } if (adp == "ATCTCTCTCAATTTTTTTTTTTTTTTTTTTTTTTAAGAGAGAGAT") {
        return "POLYA";
    } if (adp == "ATCTCTCTCAACAACAACAGGCGAAGAGGAAGGAAAGAGAGAGAT") {
        return "SCR";
    }
    return "OTHER";
}

void ChunkPolyReads(const std::vector<FastaEntry>& polyReads,
                   const std::string& outprefix,
                   size_t chunkSize,
                   size_t overlapSize)
{
    std::vector<FastaEntry> chunks;
    for (const auto& read : polyReads)
    {
        auto baseid = read.id;
        for (size_t i = 0; i < read.sequence.size(); i += chunkSize)
        {
            size_t realstart = (i >= overlapSize) ? i - overlapSize : i;
            size_t realend = std::min(read.sequence.size(), i + chunkSize);
            std::ostringstream chunkname;
            chunkname << baseid << "/" << realstart << "-" << realend;
            chunks.emplace_back(chunkname.str(), read.sequence.substr(realstart, realend - realstart));
        }
    }
    std::ostringstream protofname;
    protofname << outprefix << "." << chunkSize << "bp_" << overlapSize << "ovlp_bp.fasta";
    FastaUtilities::WriteSingleFastaFile(chunks, protofname.str());
}

int main(int argc, char** argv)
{
    try
    {
        auto parser = PacBio::Process::OptionParser().description(
            "Experiment with PPA's adapter calling API without calling bam2bam")
        .usage("-a adapters.fasta -p polymerase_reads.fasta")
        .epilog("Example: bam2bam -p polymerase_reads.fasta -a adapters.fasta\n");

        auto groupMand = PacBio::Process::OptionGroup(parser, "Mandatory parameters");
        groupMand.add_option("-p", "--polymerase-reads").dest("polyReadFile").metavar("STRING").help(
            "Input polymerase read fasta file");
        groupMand.add_option("-a", "--adapters").dest("adapterFile").metavar("STRING").help(
            "Input adapters fasta file");
        parser.add_option_group(groupMand);

        auto groupOpt = PacBio::Process::OptionGroup(parser, "Optional parameters");
        groupOpt.add_option("--summarize").action_store_true().help(
            "Product summary instead of per-adapter results");
        groupOpt.add_option("--genchunks").action_store_true().help(
            "Generate chunked polymerase reads instead of running adapterfinding");
        groupOpt.add_option("--chunk-size").dest("chunkSize").type_int().set_default(100).help(
            "Chunk size for generating polymerase read chunks (not including overlap)");
        groupOpt.add_option("--overlap-size").dest("overlapSize").type_int().set_default(100).help(
            "Overlap size for generating polymerase read chunks");
        groupOpt.add_option("-o", "--outprefix").dest("outputPrefix").metavar("STRING").help(
            "Output prefix");
        parser.add_option_group(groupOpt);

        PacBio::Process::Values options = parser.parse_args(argc, argv);

        bool genchunks = options.get("genchunks");
        bool summarize = options.get("summarize");
        std::string adapterFile = options["adapterFile"];
        std::string polyReadFile = options["polyReadFile"];
        if (options.is_set_by_user("genchunks") && options.is_set_by_user("summarize"))
        {
            std::cerr << "Cannot summarize adapter finding when generating "
                      << "chunked polymerase reads" << std::endl;
            return 1;
        }
        // Read sequences from the adapter file into memory and create an
        // AdapterLabeler object to search for them
        auto adapterList = std::make_shared<std::vector<FastaEntry>>(
                FastaUtilities::ParseSingleFastaFile(adapterFile));
        auto readList = std::make_shared<std::vector<FastaEntry>>(
                FastaUtilities::ParseSingleFastaFile(polyReadFile));

        if (genchunks)
        {
            if (!options.is_set_by_user("outputPrefix"))
            {
                std::cerr << "Must specify output prefix to chunk polymerase reads " << std::endl;
                return 1;
            }
            std::cout << "Chunking polymerase reads to " << options["chunkSize"]
                      << " bases + " << options["overlapSize"]
                      << " overlap bases" << std::endl;
            ChunkPolyReads(*readList, options["outputPrefix"],
                           std::stoi(options["chunkSize"]), std::stoi(options["overlapSize"]));
            return 0;
        }

        Postprimary::AdapterLabeler adpLabeler(adapterList, Postprimary::Platform::SEQUELII);

        // Convert those adapter sequences to their names if possible for clarity
        std::vector<std::string> adpNames;
        for (const auto& adp : *adapterList)
            adpNames.push_back( AdapterSequenceToName(adp.sequence) );

        // Print the header
        if (!summarize)
        {
            std::cout << "Zmw,AdpStart,AdpEnd,AdpAccuracy,AdpScore,AdpFlankingScore,AdpIndex,AdpPassRSII,AdpPassSequel,ObservedSeq,ExpectedSeq" << std::endl;
        }

        uint32_t totalFound = 0;
        RegionLabel lastAdapter;
        for (const auto& read : *readList)
        {
            RegionLabel hqRegion(
                0, read.sequence.size(), 100, RegionLabelType::HQREGION);

            const auto& adapters = adpLabeler.Label(read.sequence, hqRegion, false);

            if (summarize)
            {
                totalFound += adapters.size();
            }
            else
            {
                for (const auto& label : adapters)
                {
                    // Which platform-specific filtering criteria, if any, do we pass?
                    const bool hardPassRSII  = label.accuracy >= 0.66 ? true : false;
                    const bool softPassRSII  = label.accuracy >= 0.60 && label.flankingScore >= 10 ? true : false;
                    const std::string adpPassRSII = hardPassRSII || softPassRSII ? "T" : "F";
                    const bool hardPassSeq   = label.accuracy >= 0.61 ? true : false;
                    const bool softPassSeq   = label.accuracy >= 0.55 && label.flankingScore >= 15 ? true : false;
                    const std::string adpPassSeq  = hardPassSeq  || softPassSeq  ? "T" : "F";

                    // Extracting the observed vs. expected sequence
                    const std::string adpSeq = read.sequence.substr(label.begin, label.end - label.begin);
                    const std::string expSeq = adpNames[label.sequenceId];

                    std::cout << read.id << ","
                              << label.begin << ","
                              << label.end << ","
                              << label.accuracy << ","
                              << label.score << ","
                              << label.flankingScore << ","
                              << label.sequenceId << ","
                              << adpPassRSII << ","
                              << adpPassSeq << ","
                              << adpSeq << ","
                              << expSeq
                              << std::endl;
                }
            }
        }
        if (summarize)
        {
            std::cout << totalFound << " adapters found" << std::endl;
        }
        return 0;
    }
    catch(const std::exception& ex)
    {
        std::cerr << "Exception caught: " << ex.what() << std::endl;
        return 1;
    }
}
