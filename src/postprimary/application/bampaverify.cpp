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

// Programmer: John Nguyen, Armin Töpfer

#include <fstream>
#include <memory>

#include <pacbio/process/OptionParser.h>

#include <pbbam/BamRecord.h>
#include <pbbam/PbiFilterQuery.h>

#include <postprimary/test/SentinelVerification.h>

#include <git-rev.h>

static const std::string versionString = cmakeGitBranch() + "_" + cmakeGitHash();

using namespace PacBio::BAM;
using namespace PacBio::Primary::Postprimary;

// Entry point
int main(int argc, char* argv[])
{
    auto parser = PacBio::Process::OptionParser().description(
        "bampaverify verifies the BAM file using the PA ZMW hole number "
        "encoding scheme on full polymerase ZMW sequences")
    .usage("[options] movieName.zmw.bam")
    .epilog("Example: bampaverify movieName.zmw.bam");

    parser.version(versionString);
    parser.add_option("--outputFile").set_default("").dest("outputFile").help("output file");

    auto options = parser.parse_args(argc, argv);
    std::vector<std::string> args = parser.args();

    std::ofstream fileOut;
    if (options["outputFile"] != "")
    {
        fileOut.open(options["outputFile"]);
        fileOut << "holeNumber,expSeq,err_ind,errSeq" << std::endl;
    }

    if (args.size() == 0)
    {
        std::cerr << "ERROR: INPUT EMPTY." << std::endl;
        return 0;
    }

    // for each input file
    for (const auto& input : args)
    {
        SentinelVerification sv;

        // setup query (respecting dataset filter, if present)
        const PbiFilter filter = PbiFilter::FromDataSet(input);
        const PbiFilterQuery bamQuery(filter, input);

        // Iterate over all records 
        for (const auto record : bamQuery)
            sv.CheckZMW(record.Sequence(), record.HoleNumber());

        std::cout << input << std::endl;
        std::cout << "Unmatched ZMWs = " << sv.NumSentinelZMWs() << std::endl;
        std::cout << "Matched ZMWs = " << sv.NumNormalZMWs() << std::endl;

        if (options["outputFile"] != "")
        {
            for (const auto &i : sv.UnmatchedZMWsInfo())
                fileOut << i << std::endl;
        }
    }

    if (fileOut.is_open())
        fileOut.close();

    return 0;
}
