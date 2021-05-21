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

#pragma once

#include <atomic>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include <pacbio/logging/Logger.h>


#include <pbbam/BamRecord.h>
#include <pbbam/BamHeader.h>
#include <pbbam/IndexedBamWriter.h>
#include <pbbam/BamWriter.h>
#include <pbbam/PbiFile.h>

#include "Validation.h"

namespace PacBio {
namespace Primary {
namespace Postprimary {

using namespace PacBio::BAM;

struct BamProducer
{
private:
    std::unique_ptr<BamWriter>          bamStream_;
    std::unique_ptr<IndexedBamWriter>   ibamStream_;
    bool                                savePbi_;
    std::string                         bamPath_; ///< will be "-" when writing to stdout, otherwise disk path
    std::string                         pbiPath_; ///< will be "" when writing bam to stdout
    int                                 pbiThreads_;
    BamHeader                           bamHeader_;
public:
    BamProducer(const std::string& outputPrefix,
                const std::string convention,
                const BamHeader& header,
                const bool savePbiFile,
                const bool inlinePbiFile,
                const int bamThreads,
                const int threads,
                const bool streamBam)
    {
        savePbi_ = savePbiFile;
        if (streamBam)
        {
            bamPath_ = "-";
            pbiPath_ = "";
        }
        else
        {
            bamPath_ = outputPrefix + "." + convention + ".bam";
            pbiPath_ = bamPath_ + ".pbi";
        }
        pbiThreads_ = bamThreads + threads;
        bamHeader_ = header;

        PBLOG_INFO << "bam file: " << bamPath_;
        PBLOG_INFO << "bamThreads = " << bamThreads;
        try
        {
            if (streamBam) {
                PBLOG_INFO << "Writing streaming BAM file to stdout";
                bamStream_.reset(new BamWriter(
                    bamPath_, header,
                    BamWriter::CompressionLevel::NoCompression, bamThreads,
                    BamWriter::BinCalculation_OFF));
            }
            else if (inlinePbiFile) {
                PBLOG_INFO << "Writing inline pbi file";
                IndexedBamWriterConfig config;
                config.outputFilename = bamPath_;
                config.header = header;
                config.bamCompressionLevel = BamWriter::DefaultCompression;
                config.pbiCompressionLevel = PbiBuilder::DefaultCompression;
                config.numBamThreads = bamThreads;
                config.numPbiThreads = bamThreads;
                config.numGziThreads = bamThreads;
                config.tempFileBufferSize = 0x8000000;

                ibamStream_.reset(new IndexedBamWriter(config));
            }
            else {
                PBLOG_INFO << "Writing local BAM file";
                bamStream_.reset(new BamWriter(
                        bamPath_, header,
                        BamWriter::CompressionLevel::CompressionLevel_1, bamThreads,
                        BamWriter::BinCalculation_OFF));
            }
        }
        catch(std::exception& e)
        {
            PBLOG_ERROR << "[ResultWriter] " << e.what() << " while opening " << bamPath_ ;
            throw;
        }
    }

    inline void Write(const BamRecord& r)
    {
        r.header_ = bamHeader_;
        if (ibamStream_)
            ibamStream_->Write(r);
        else if (bamStream_)
            bamStream_->Write(r);
    }

    bool Close(const std::atomic_bool& abortNow)
    {
        // BAM output
        if (ibamStream_)
        {
            ibamStream_.reset();
        }
        else
        {
            if (bamStream_)
            {
                bamStream_.reset();
            }
            if (!abortNow && bamPath_ != "-")
            {
                try
                {
                    BamFile validationBam(bamPath_);
                    if (savePbi_)
                    {
                        PBLOG_INFO << "Creating pbi file " << pbiPath_;
                        PbiFile::CreateFrom(validationBam,
                                            PbiBuilder::CompressionLevel::CompressionLevel_1,
                                            pbiThreads_);
                    }
                }
                catch (const std::runtime_error& e)
                {
                    PBLOG_ERROR << e.what();
                    return false;
                }
            }
        }

        return true;
    }
    const std::string BamPath() const { return bamPath_; }
    const std::string BamPbiPath() const { return pbiPath_; }
};

}}} // ::PacBio::Primary::Postprimary
