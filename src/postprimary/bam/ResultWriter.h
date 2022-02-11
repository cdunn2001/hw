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

#include <thread>
#include <string>
#include <vector>
#include <memory>

#include <zlib.h>

#include <pbbam/ProgramInfo.h>
#include <pbbam/DataSet.h>
#include <pbbam/RunMetadata.h>

#include <bazio/file/FileHeader.h>

#include <postprimary/application/PpaAlgoConfig.h>
#include <postprimary/application/UserParameters.h>
#include <postprimary/stats/ChipStats.h>

#include "Validation.h"
#include "ComplexResult.h"
#include "ThreadSafeMap.h"
#include "BamCommentSideband.h"

namespace PacBio {
namespace Primary {
namespace Postprimary {

struct RuntimeMetaData;
class BamProducer;

static const std::string chipStatsMetaType        = "PacBio.SubreadFile.ChipStatsFile";
static const std::string adapterFileMetaType      = "PacBio.SubreadFile.AdapterFastaFile";
static const std::string controlFileMetaType      = "PacBio.SubreadFile.ControlFastaFile";

// Different output BAM configurations for the polymerase, subread, and 
// hqregion mode
struct BamConfig
{
    std::string primaryMetaType;
    std::string secondaryMetaType;
    std::string primaryRgType;
    std::string primaryBamInfix;
};

static BamConfig polyConfig   {"PacBio.SubreadFile.ZmwBamFile",
                               "PacBio.SubreadFile.ZmwScrapsBamFile",
                               "ZMW",
                               "zmws"};
static BamConfig hqConfig     {"PacBio.SubreadFile.HqRegionBamFile",
                               "PacBio.SubreadFile.HqScrapsBamFile",
                               "HQREGION",
                               "hqregions"};
static BamConfig subreadConfig{"PacBio.SubreadFile.SubreadBamFile",
                               "PacBio.SubreadFile.ScrapsBamFile",
                               "SUBREAD",
                               "subreads"};

/// \brief ResultWriter is a wrapper for BAM::BamWriter.
///
/// \details Allows queueing of BamRecords, wrapped in ResultPackets.
///          Controls a thread, dedicated to writing BamRecords to disk
///          as subreads.bam, scraps.bam, fasta.gz, and fastq.gz.
class ResultWriter
{
public:
    using FileHeader = BazIO::FileHeader;
public: // structors
    /// Opens stream to new output BAM files. These may include
    ///   subreads.bam, scraps.bam, fasta.gz, fastq.gz, consensusreads.bam
    /// \param user - needs to set outputPrefix, nobam, outputFasta, outputFastq, runReport, savePbi, saveScraps and a few others
    /// \param cmd - read-only information from the Colllection Metadata
    /// \param rmd - read-only information from the Run Metadata (aka run design)
    /// \param ppaAlgoConfig - read-only information about the PPA algorithms
    /// \param fileHeader - read-only header from the BAZ file
    /// \param apps - a collection of the apps that will be written to the BAM header.
    /// \param computerStats - if true, additional statistics are calculated
    /// \param maxNumZmws - the maximum number of ZMWs that will be processed (either limited by BAZ file or whitelist)
    ResultWriter(const UserParameters* user,
                 const PacBio::BAM::CollectionMetadata* cmd,
                 std::shared_ptr<RuntimeMetaData>& rmd,
                 const PpaAlgoConfig* ppaAlgoConfig,
                 const std::string& movieName,
                 float movieTimeInHrs,
                 const std::string& bazVersion,
                 const std::string& bazWriterVersion,
                 const std::string& basecallerVersion,
                 double frameRateHz,
                 bool internal,
                 const std::vector<PacBio::BAM::ProgramInfo>& apps,
                 bool computeStats,
                 uint32_t maxNumZmws);

    // Default constructor
    ResultWriter() = delete;

    // Move constructor
    ResultWriter(ResultWriter&&) = delete;

    // Copy constructor
    ResultWriter(const ResultWriter&) = delete;

    // Move assignment operator
    ResultWriter& operator=(ResultWriter&&) = delete;

    // Copy assignment operator
    ResultWriter& operator=(const ResultWriter&) = delete;


    /// Closes output streams.
    ~ResultWriter();

public: // modifying methods
    void SetCollectionMetadataDataSet(const PacBio::BAM::DataSet* ds)
    { ds_ = ds; }

public: // non-modifying methods
    Validation CloseAndValidate();

    /// Saves vector of ResultPackets in a local buffer.
    inline void AddResultsToBuffer(uint32_t batchId,
                                   std::vector<ComplexResult>&& results)
    {
        if (abortNow_) return;

        // The check on the batchId is necessary since results are pulled out
        // of the map in order to avoid deadlocking if the WriteToDiskLoop() is
        // waiting on this batchId.

        if (batchId != batchIdCounter_ && ResultMapFull())
            PBLOG_INFO << "[AddResultsToBuffer] Output buffer full, waiting output size = " << resultMapSizeBytes_;

        while (batchId != batchIdCounter_ && ResultMapFull())
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

        size_t resultsSizeBytes = std::accumulate(results.cbegin(), results.cend(), size_t(0),
                                                  [](size_t v, const ComplexResult& cr)
                                                  { return v + cr.EstimatedBytesUsed(); });
        AddBatchBytes(batchId, resultsSizeBytes);

        resultMap_.Set(batchId, std::move(results));
    }

    inline size_t WriteBufferSize() { return resultMap_.Size(); }

    inline void Abort() { abortNow_ = true; }

private: //static data

private: // data
    const std::string newline_ = "\n";
    const std::string plusline = "\n+\n";
    const std::string preFasta_ = ">";
    const std::string preFastq_ = "@";
    const std::string bamVersion_ = "1.5";
    const std::string pacbioVersion_ = "3.0.1";
    const std::string sortOrder_ = "unknown";

    PacBio::BAM::DataSet dataset_;

    std::string fastqFileName_;
    std::string fastaFileName_;

    std::string fastqFileTmpName_;
    std::string fastaFileTmpName_;

    std::atomic_bool abortNow_;

    const UserParameters* user_;
    const PacBio::BAM::CollectionMetadata* cmd_;
    std::shared_ptr<RuntimeMetaData> rmd_;
    const PpaAlgoConfig* ppaAlgoConfig_;
    const std::string movieName_;
    const float movieTimeInHrs_;
    const std::string bazVersion_;
    const std::string bazWriterVersion_;
    const std::string basecallerVersion_;
    const double frameRateHz_;
    const bool internal_;

    const PacBio::BAM::DataSet* ds_;

    bool computeStats_;

    gzFile fastaStream_;
    gzFile fastqStream_;

    PacBio::BAM::ReadGroupInfo primaryRG_;
    PacBio::BAM::ReadGroupInfo scrapsRG_;

    std::atomic_bool finishSaving_;
    std::atomic_bool writeThreadContinue_;

    std::thread cleanThread_;
    std::thread writeThread_;

    std::unique_ptr<BamProducer> bamPrimaryStream_;
    std::unique_ptr<BamProducer> bamScrapsStream_;

    PacBio::BAM::Tag scrapsAdapter_, scrapsBarcode_, scrapsLQRegion_, scrapsFiltered_;
    PacBio::BAM::Tag scrapsZmwNormal_, scrapsZmwSentinel_, scrapsZmwControl_, scrapsZmwMalformed_;

    ThreadSafeMap<uint32_t, std::vector<ComplexResult>> resultMap_;
    std::map<uint32_t,size_t> resultsToSizeBytes_;
    size_t resultMapSizeBytes_ = 0;
    std::mutex resultSizeMutex_;

    uint32_t batchIdCounter_ = 0;
    uint32_t numRecords_ = 0;
    uint64_t totalLength_ = 0;
    uint32_t numHasStem_ = 0;
    uint32_t numHasAdapters_ = 0;
    // The number of ZMWs with stems/adapters and greater than or equal to three
    // adapters
    uint32_t numHasStemGEQ3_ = 0;
    uint32_t numHasAdaptersGEQ3_ = 0;

    Validation validationStatus_ = Validation::OPEN;

    ChipStats chipStats_;

    PacBio::BAM::DataSet subreadSet_;

    /// full pathname to the .sts.xml file. If no file is written, this should be empty.
    std::string stsXmlFilename_;
    /// full pathname to the .sts.h5 file. If no file is written, this should be empty.
    std::string stsH5Filename_;

    uint32_t maxNumZmws_ = 0;
private:
    inline bool ResultMapFull()
    {
        std::lock_guard<std::mutex> lk(resultSizeMutex_);
        return resultMapSizeBytes_ > (user_->maxOutputQueueMB << 20);
    }

    inline void AddBatchBytes(uint32_t batchId, size_t resultsSizeBytes)
    {
        std::lock_guard<std::mutex> lk(resultSizeMutex_);
        resultsToSizeBytes_[batchId] = resultsSizeBytes;
        resultMapSizeBytes_ += resultsToSizeBytes_[batchId];
    }

    inline void RemoveBatchBytes(uint32_t batchId)
    {
        std::lock_guard<std::mutex> lk(resultSizeMutex_);
        resultMapSizeBytes_ -= resultsToSizeBytes_[batchId];
        resultsToSizeBytes_.erase(batchId);
    }

    /// Writes content of buffer to disk.
    void WriteToDiskLoop();

    void WriteFasta(const ResultPacket& result, gzFile* outputStream);

    void WriteFastq(const ResultPacket& result, gzFile* outputStream);

    void WriteRecord(const ResultPacket& result,
                            std::unique_ptr<BamProducer>& bamStream);

    inline void WriteSHPRecord(ResultPacket* result)
    {
        result->bamRecord.Impl().AddTag("RG", primaryRG_.Id());
        WriteRecord(*result, bamPrimaryStream_);
    }

    inline void WriteScrapRecord(const PacBio::BAM::Tag& scrapRegionType,
                                 ResultPacket* result)
    {
        if (user_->saveScraps)
        {
            result->bamRecord.Impl().AddTag("RG", scrapsRG_.Id());
            result->bamRecord.Impl().AddTag("sc", scrapRegionType);
            AddZmwType(result);
            WriteRecord(*result, bamScrapsStream_);
        }
    }

    inline void AddZmwType(ResultPacket* result)
    {
        if (result->control)
            result->bamRecord.Impl().AddTag("sz", scrapsZmwControl_);
        else if (result->rejected)
            result->bamRecord.Impl().AddTag("sz", scrapsZmwSentinel_);
        else
            result->bamRecord.Impl().AddTag("sz", scrapsZmwNormal_);
    }

private: // Clean up

    /// Cleans after the WriteToDiskLoop
    void CleanLoop();

private: // Create output files and open streams

    void CreateDataSet();

    void WriteDataset();

    gzFile CreateGZFile(const char* const cString) const;

    void CreateFastqFiles();

    void CreateFastaFiles();

    /// Creates all of the BAMS associated with this dataset, including
    /// the main bam file and the scraps bam.
    /// \param programsInfos - a list of program names that created the BAM file
    /// \param localRgType -
    /// \param localBamInfix - ?
    /// \param additionalExternalResources - path names to random files that are to be  logically attached to the
    ///        bam file. This is used in CCS
    /// \param localRG - output that points to created ReadGroupINfo
    /// \param localBam - output BamProducer created by this function
    /// \returns the filename of the BAM file just created
    std::string CreateBam(const std::vector<PacBio::BAM::ProgramInfo>& programInfos,
                    const std::string& localRgType,
                    const std::string& localBamInfix,
                    const std::vector<BamCommentSideband::ExternalResource>& additionalExternalResources,
                    PacBio::BAM::ReadGroupInfo* localRG,
                    std::unique_ptr<BamProducer>* localBam);

    /// Creates all the bam files
    void CreateBams(const std::vector<PacBio::BAM::ProgramInfo>& programInfos);

    PacBio::BAM::ExternalResource CreateResourceBam(const std::string& bamFileName,
            const std::string& convention,
            const std::string& metatype);

    void CreateAndAddResourceStsXml(PacBio::BAM::ExternalResource* parent);

    std::string RelativizePath(const std::string& originalFilePath);

    void CreateAndAddResourceAdapterFile(PacBio::BAM::ExternalResource* parent);

    void AddResourceFastx(const std::string& convention, const std::string& filename);

    std::vector<PacBio::BAM::ProgramInfo> CreateProgramInfos(const std::vector<PacBio::BAM::ProgramInfo>& apps);

    PacBio::BAM::ReadGroupInfo CreateReadGroupInfo(const std::string& readType);


private: // init
    void PopulateScrapTags();
};

}}} // ::PacBio::Primary::Postprimary
