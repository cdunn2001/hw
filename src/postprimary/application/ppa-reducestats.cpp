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

// Programmer: Mark Lakata
//
// A program that reads in the *.sts.h5 and writes a *.rsts.h5

#include <iostream>
#include <string>

#include <pacbio/process/ProcessBase.h>
#include <pacbio/process/ConfigurationBase.h>
#include <pacbio/logging/Logger.h>

#include <pacbio/primary/ZmwReducedStatsFile.h>
#include <pacbio/primary/ZmwStatsFile.h>

#include <git-rev.h>

static const std::string versionString = cmakeGitBranch() + "_" + cmakeGitHash();

using namespace PacBio::Process;
using namespace PacBio::Primary;

// Entry point
int main(int argc, char* argv[])
{
    try
    {

        OptionParser parser = ProcessBase::OptionParserFactory();

        parser.description("ppa-reducestats reduces the sts.h5 file to a rsts.h5 (reduced statistics)");
        parser.usage("--input yours.sts.h5 --output mine.rsts.h5 --config file.json");

        parser.version(versionString);

        auto groupMand = PacBio::Process::OptionGroup(parser, "Mandatory parameters");
        groupMand.add_option("--input").help("Pathname of input *.sts.h5 file");
        groupMand.add_option("--output").help("Pathname of output *.rsts.h5 file");
        groupMand.add_option("--images").action_store_true().help("Generate pgm images for debugging");
        parser.add_option_group(groupMand);

        auto group1 = PacBio::Process::OptionGroup(parser, "Configuration parameters");
        group1.add_option("--config").action_append().help("Loads JSON configuration. Can be file name or inline JSON object, e.g. \"{ ... }\"");
        group1.add_option("--strict").action_store_true().help("Strictly check all configuration options. Do not allow unrecognized configuration options");
        group1.add_option("--showconfig").action_store_true().help("Shows the entire configuration namespace with default values and exits");
        parser.add_option_group(group1);

        PacBio::Process::Values options = parser.parse_args(argc, argv);
        ProcessBase::HandleGlobalOptions(options);

        PacBio::Process::ConfigMux mux;
        mux.Ignore("acquisition");
        mux.Ignore("basecaller");
        mux.Ignore("basewriter");
        mux.Ignore("ppa");
        mux.ProcessCommandLine(options.all("config"));

        // chipClass is set by now for reduced stats config
        ReducedStatsConfig reducedStatsConfig{};
        mux.Add("ppa-reducestats", reducedStatsConfig);

        // reprocess config parameters
        mux.ProcessCommandLine(options.all("config"));

        if (options.get("showconfig"))
        {
            std::cout << mux.ToJson();
            return 0;
        }

        // write
        {

            PBLOG_INFO << "configuration:" << reducedStatsConfig.Json();

            ZmwStatsFile input(options["input"]);
            ZmwReducedStatsFile output(options["output"], reducedStatsConfig);
            input.CopyScanData(output.ScanData());

            if (options.get("images"))
            {
                output.SetImageOption(true);
            }
            output.Reduce(input, reducedStatsConfig);
        }

    }
    catch(const std::exception& ex)
    {
        PBLOG_FATAL << ex.what();
    }
    catch(...)
    {
        PBLOG_FATAL << "Uncaught non standard exception";
    }
}
