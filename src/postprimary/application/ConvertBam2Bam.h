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

#include <atomic>
#include <fstream>
#include <memory>
#include <mutex>
#include <thread>

#include <pbbam/BamFile.h>
#include <pbbam/BamRecord.h>
#include <pbbam/ProgramInfo.h>
#include <pbbam/EntireFileQuery.h>
#include <pbbam/DataSet.h>
#include <pbbam/RunMetadata.h>
#include <pbbam/virtual/ZmwReadStitcher.h>
#include <pbbam/virtual/ZmwWhitelistVirtualReader.h>

#include <bazio/file/FileHeader.h>

#include <bazio/PacketFieldName.h>

#include <postprimary/bam/EventData.h>
#include <postprimary/bam/RuntimeMetaData.h>
#include <postprimary/bam/SubreadLabeler.h>
#include <postprimary/stats/ProductivityMetrics.h>

#include "BamSet.h"
#include "UserParameters.h"
#include "PpaAlgoConfig.h"

namespace PacBio {
namespace Primary {
namespace Postprimary {

using namespace PacBio::BAM;

class ResultWriter;

/// ConvertBam2Bam orchestrates BAM reading, ZMW labeling, and result queuing
/// to ResultWriter that have to be written to disk.
class ConvertBam2Bam
{
public:
    using FileHeader = BazIO::FileHeader;
public: // structors
    
    /// Creates instance BC labeler, BazReader, and ResultWriter and 
    /// fills thread pool.
    ConvertBam2Bam(std::shared_ptr<UserParameters>& user);
    // Default constructor
    ConvertBam2Bam() = delete;
    // Move constructor
    ConvertBam2Bam(ConvertBam2Bam&&) = delete;
    // Copy constructor
    ConvertBam2Bam(const ConvertBam2Bam&) = delete;
    // Move assignment operator
    ConvertBam2Bam& operator=(ConvertBam2Bam&&) = delete;
    // Copy assignment operator
    ConvertBam2Bam& operator=(const ConvertBam2Bam&) = delete;
    // Destructor
    ~ConvertBam2Bam();

private:   // data
    const size_t                        batchSize = 10;
    std::atomic_bool                    threadpoolContinue_;
    std::atomic_bool                    loggingContinue_;
    std::atomic_int                     batchId;
    std::atomic_int                     numProcessedZMWs_;
    std::atomic_int                     threadCount_;
    std::mutex                          mutex_;
    std::vector<std::thread>            threadpool_;
    std::vector<BamSet> recordSetList_;
    std::thread                         loggingThread_;

    PacBio::BAM::DataSet                ds_;
    std::shared_ptr<RuntimeMetaData>    rmd_;
    std::unique_ptr<FileHeader>         fh_;
    std::shared_ptr<UserParameters>     user_;
    std::shared_ptr<PpaAlgoConfig>      ppaAlgoConfig_;
    const BAM::DataSet                  dataset_;
    std::unique_ptr<ResultWriter>       resultWriter_;
    std::unique_ptr<SubreadLabeler>     subreadLabeler_;
    std::vector<BAM::BaseFeature>       baseFeatures_;

    std::string                         primaryBam_;
    std::string                         scrapsBam_;

    RecordType                          primaryBamType_;

    std::unique_ptr<BAM::ZmwReadStitcher> vpr_;
    std::unique_ptr<BAM::WhitelistedZmwReadStitcher> wvpr_;

    bool                                whiteList_ = false;

private: // modifying methods
    
    void InitLogToDisk();

    /// Orchestrates reading of BamRecords and group them by hole number.
    void ParseInput();

    void ParseInputSorted();

    void InitPpaAlgoConfig();

    void SetBamFile(const std::string& input);

    RecordType ConvertBamType(const std::string& input);
    
    /// \brief Single thread of thread pool.
    /// \details Responsible for taking a batch of ZMWs of the recordSetList_,
    ///          label and split reads, and queue them for writing to disk.
    void SingleThread();

    int ParseHeader(std::vector<ProgramInfo>* apps);

    void ParseHeader(const BAM::BamHeader& header);

    VirtualZmwBamRecord BamRecordsToPolyRead(BamSet&& bamSet);

    EventData PolyReadToEventData(
        const VirtualZmwBamRecord& polyRead, size_t zmwId);

    double PolyReadToBarcodeScore(
        const VirtualZmwBamRecord& polyRead);

    bool PolyReadToIsControl(
        const VirtualZmwBamRecord& polyRead);

    RegionLabels PolyReadToRegionLabels(
        const VirtualZmwBamRecord& polyRead,
        const EventData& events,
        const std::unordered_map<size_t, size_t> base2frame);

    ProductivityInfo PolyReadToProductivityInfo(
        const VirtualZmwBamRecord& polyRead,
        const RegionLabel& hqregion);

    inline void AddBaseFeature(const BAM::ReadGroupInfo& rg,
                               const BAM::BaseFeature& feature);

    template <typename TOut, typename ContainerIn, typename DefaultVal = uint32_t>
    inline void AppendPacketFields(std::vector<TOut>& dest,
                                   const ContainerIn& data,
                                   const std::vector<uint32_t>& isBase = {},
                                   DefaultVal defaultVal = 0)
    {
        dest.reserve(data.size());
        if (isBase.size() == 0)
        {
            std::copy(data.begin(), data.end(), std::back_inserter(dest));
        }
        else
        {
            for (size_t i = 0, j = 0; i < isBase.size(); ++i)
            {
                dest.push_back(isBase[i] ? data.at(j++) : defaultVal);
            }
            assert(dest.size() == isBase.size());
        }
    }

};

}}} // ::PacBio::Primary::Postprimary


