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

// Programmer: Mark Lakata

#include <string>
#include <pacbio/primary/ZmwStatsFile.h>
#include <pacbio/text/String.h>


/// simple program used to convert sts.h5 to CSV for viewing
///
/// usage:
///  stsTool filename.sts.h5
///     Will dump out all nontemporal statistics to a CSV formatted stdout
///
///  stsTool filename.sts.h5 zmwIndex
///     Will dump out all temporal statistics for a particular ZMW index to a CSV formatted stdout.
///
///  Any field that is multidimensional will be flattened and delimited with /

int main(int argc, char* argv[])
{
    try
    {
        if (argc < 2) throw PBException("need filename");
        std::string filename = argv[1];

        PacBio::Primary::ZmwStatsSetQuiet();
        PacBio::Primary::ZmwStatsFile file(filename);

        if (argc == 3)
        {
            uint32_t zmwIndex = std::stoi(argv[2]);

            auto headers = file.TemporalColumnHeaders();
            std::cout << "t," << PacBio::Text::String::Join(headers.begin(), headers.end(), ',') << std::endl;
            for (uint32_t t = 0; t < file.nMF(); t++)
            {
                auto values = file.TemporalColumnValues(zmwIndex, t);
                std::cout << t << "," << PacBio::Text::String::Join(values.begin(), values.end(), ',') << std::endl;
            }
        }
        else if (argc == 2)
        {

            auto headers = file.ColumnHeaders();
            std::cout << PacBio::Text::String::Join(headers.begin(), headers.end(), ',') << std::endl;
            for (uint32_t i = 0; i < file.nH(); i++)
            {
                auto values = file.ColumnValues(i);
                std::cout << PacBio::Text::String::Join(values.begin(), values.end(), ',') << std::endl;
            }
        }
        return 0;
    }
    catch(const std::exception& ex)
    {
        std::cerr << "EXCEPTION:" << ex.what() << std::endl;
    }
    return 1;
}
