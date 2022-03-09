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
#include <deque>
#include <memory>
#include <mutex>
#include <thread>


#include <pacbio/logging/Logger.h>

#include <pbbam/RunMetadata.h>

#include <bazio/BazReader.h>
#include <pacbio/primary/ZmwStatsFile.h>

#include <postprimary/bam/RuntimeMetaData.h>
#include <postprimary/bam/ResultWriter.h>
#include <postprimary/bam/SubreadLabeler.h>
#include <postprimary/hqrf/HQRegionFinder.h>
#include <postprimary/insertfinder/InsertFinder.h>
#include <postprimary/stats/ProductivityMetrics.h>

#include "PpaAlgoConfig.h"
#include "PpaProgressMessage.h"
#include "UserParameters.h"

#include "RAIISignalHandler.h"

namespace PacBio {
namespace Primary {
namespace Postprimary {

/// ConvertBaz2Bam orchestrates BAZ reading, ZMW labeling, and result queuing
/// to ResultWriter that have to be written to disk.
class ConvertBaz2Bam
{
public: // structors
    /// Creates instances of AF, BC, CF, and HQ labeler, BazReader, 
    /// and ResultWriter and fills thread pool.
    ConvertBaz2Bam(std::shared_ptr<UserParameters>& user);
    // Default constructor
    ConvertBaz2Bam() = delete;
    // Move constructor
    ConvertBaz2Bam(ConvertBaz2Bam&&) = delete;
    // Copy constructor
    ConvertBaz2Bam(const ConvertBaz2Bam&) = delete;
    // Move assignment operator
    ConvertBaz2Bam& operator=(ConvertBaz2Bam&&) = delete;
    // Copy assignment operator
    ConvertBaz2Bam& operator=(const ConvertBaz2Bam&) = delete;
    // Destructor
    ~ConvertBaz2Bam();

    int Run();

private:
    static constexpr double MinimumHeartbeatTime = 1.0;

private:   // data
    // Helper class, to manage a vector of zmw data that needs
    // to be shared between different threads
    class ReadList
    {
    public:
        size_t NumZmw()
        {
            std::lock_guard<std::mutex> lm(mutex_);
            return data_.size();
        }

        size_t NumBytes()
        {
            std::lock_guard<std::mutex> lm(mutex_);
            return numBytes_;
        }

        bool empty()
        {
            std::lock_guard<std::mutex> lm(mutex_);
            return data_.empty();
        }

        void Append(ZmwByteData&& zmwData)
        {
            std::lock_guard<std::mutex> lm(mutex_);

            numBytes_ += zmwData.NumBytes();

            data_.emplace_back(std::move(zmwData));
        }
        void Append(std::vector<ZmwByteData>&& zmwData)
        {
            std::lock_guard<std::mutex> lm(mutex_);

            for (const auto& zmw : zmwData) numBytes_ += zmw.NumBytes();

            data_.insert(data_.end(),
                         std::make_move_iterator(zmwData.begin()),
                         std::make_move_iterator(zmwData.end()));
        }
        // Extracts up to maxCount elements from the front of the data storage
        // and places them inside `batch`.  Any data currently existing inside `batch`
        // will be destroyed.
        // Function returns a unique `batchId` that monotonicially increases by one for
        // each batch of data that is returned.  If no data is available to populate
        // the batch, a value of -1 will be returned
        int64_t Extract(size_t maxCount, std::vector<ZmwByteData>* batch)
        {
            std::lock_guard<std::mutex> lm(mutex_);
            size_t count = std::min(maxCount, data_.size());

            if (count == 0) return -1;

            *batch = std::vector<ZmwByteData>(
                std::make_move_iterator(data_.begin()),
                std::make_move_iterator(data_.begin() + count));
            int64_t batchId = batchCounter_;
            batchCounter_++;

            data_.erase(data_.begin(), data_.begin() + count);

            size_t removeBytes = 0;
            for (const auto& zmw : *batch)
            {
                removeBytes += zmw.NumBytes();
            }
            assert(numBytes_ >= removeBytes);
            numBytes_ -= removeBytes;
            return batchId;
        }

    private:
        std::mutex mutex_;
        std::deque<ZmwByteData> data_;
        size_t numBytes_ = 0;
        int64_t batchCounter_ = 0;
    };

    std::atomic_bool                                    abortNow_;
    std::atomic_bool                                    threadpoolContinue_;
    std::atomic_bool                                    loggingContinue_;
    std::atomic_int                                     numProcessedZMWs_;
    ///< This is equivalent to sending an "abort" message
    ///< after the number of ZMWs gets above this threshold. It is not a clean
    ///< limit to the number of ZMWs processed because the way slices are processed
    ///< in quantized units. It is only useful for profiling and not intended to
    ///< be used in production runs.
    std::atomic<uint32_t>                               maxNumZmwsToProcess_;
    std::atomic_int                                     threadCount_;
    std::shared_ptr<PpaAlgoConfig>                      ppaAlgoConfig_;
    std::shared_ptr<PacBio::BAM::CollectionMetadata>    cmd_;
    std::shared_ptr<RuntimeMetaData>                    rmd_;
    std::shared_ptr<UserParameters>                     user_;
    std::unique_ptr<BazReader>                          bazReader_;
    std::unique_ptr<SubreadLabeler>                     subreadLabeler_;
    std::unique_ptr<ResultWriter>                       resultWriter_;
    std::unique_ptr<InsertFinder>                       insertFinder_;
    std::unique_ptr<HQRegionFinder>                     hqRegionFinder_;
    std::unique_ptr<ProductivityMetrics>                prodMetrics_;
    std::unique_ptr<PacBio::Primary::ZmwStatsFile>      zmwStatsFile_;
    std::vector<std::thread>                            threadpool_;
    ReadList                                            readList_;
    std::thread                                         loggingThread_;
    bool                                                outputLog = true;
    std::vector<uint32_t>                               whiteListZmwIds_;
    bool                                                performWhiteList_ = false;
    RAIISignalHandler                                   raiiSignalHander_;
    double                                              currentProgress_{0};
    uint32_t                                            numZmwsToProcess_{0};
    std::unique_ptr<PpaProgressMessage>                 progressMessage_;

private: // const methods
    uint32_t NumZmwsToProcess() const
    {
        if (!performWhiteList_)
        {
            return bazReader_->NumZmws();
        } else
        {
            return whiteListZmwIds_.size();
        }
    }

private: // modifying methods
    /// Orchestrates processing batches of ZMW-slices.
    void ParseInput(PpaStageReporter& analyzeRpt);

    void ParseRMD();

    void InitPpaAlgoConfig();

    void StartLoggingThread(PpaStageReporter& analyzeRpt);

    /// \brief Single thread of thread pool.
    /// \details Responsible for taking a batch of ZMWs of the readList_,
    ///          label and split reads, and queue them for writing to disk.
    void SingleThread();

    void CreateZmwStatsFile(const std::string& filename);

    void CreateWhiteList();

    void LogStart();

    void InitLogToDisk();

    void CheckInputFile();

    void PopulateWorkerThreadPool();

    void InitProgressMessage();
};

}}} // ::PacBio::Primary::Postprimary
