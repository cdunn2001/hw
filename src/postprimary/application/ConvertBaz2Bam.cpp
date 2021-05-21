// Copyright (c) 2014-2018, Pacific Biosciences of California, Inc.
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



#include <chrono>
#include <cmath>
#include <memory>
#include <sys/stat.h>
#include <utility>

#include <pbbam/BamRecord.h>
#include <pbbam/ProgramInfo.h>

#include <pacbio/PBException.h>
#include <pacbio/dev/profile/ScopedProfilerChain.h>
#include <pacbio/ipc/JSON.h>
#include <pacbio/logging/Logger.h>

#include <pacbio/primary/ZmwStatsFileData.h>
#include <pacbio/primary/ZmwStatsFile.h>
#include <bazio/Timing.h>
#include <bazio/SmartMemory.h>

#include <postprimary/bam/ResultPacket.h>
#include <postprimary/bam/ComplexResult.h>
#include <postprimary/bam/EventData.h>
#include <postprimary/bam/Platform.h>
#include <postprimary/hqrf/HQRegionFinder.h>
#include <postprimary/hqrf/HQRegionFinderParams.h>
#include <postprimary/insertfinder/InsertFinder.h>
#include <postprimary/stats/ZmwStats.h>

#include <pbcopper/utility/MemoryConsumption.h>

#include "MetadataParser.h"
#include "ConvertBaz2Bam.h"
#include "OomScoreMonitor.h"
#include "ppa-config.h"

#include <git-rev.h>

const std::string versionString = cmakeGitBranch() + "_" + cmakeGitHash();

using namespace PacBio::Primary::Postprimary;

namespace { // anonymous

void PopulateBase2Frame(const EventData& events,
                        RegionLabels& regions)
{
    // We will be modifying base2frame directly. This is the same structure
    // that you would be populating for bam2bam, but only with existing region
    // boundaries.
    auto& base2frame = regions.base2frame;
    const auto& sf = events.StartFrames();
    for (size_t i = 0; i < events.NumBases(); ++i)
        base2frame[i] = sf[events.BaseToPulseIndex(i)];
}

// this only makes sense for Production mode. We assume
// that pws.size() == ipds.size() == NumBasesAll()
void ApproximateBase2Frame(const BlockLevelMetrics& bazMetrics,
                           const EventData& events,
                           RegionLabels& regions)
{
    // We will be modifying base2frame directly. This is the same structure
    // that you would be populating for bam2bam, but only with existing region
    // boundaries.
    auto& base2frame = regions.base2frame;
    const auto& numFrames = bazMetrics.NumFrames().GetRegion(0, bazMetrics.NumFrames().size());
    const auto& numBases = bazMetrics.NumBasesAll().GetRegion(0, bazMetrics.NumBasesAll().size());
    const auto& pws = events.PulseWidths();
    const auto& ipds = events.Ipds();
    size_t basesSeen = 0;
    size_t lastReferenceFrame = 0;
    size_t blocki = 0;
    size_t framei = 0;
    // Iterate over every base and accumulate pws and ipds. BUT, we also track
    // which block we're in, and the boundaries of that block in time. We limit
    // every estimate to those boundaries.
    for (size_t basei = 0; basei < pws.size(); ++basei)
    {
        // update the block boundaries if necessary
        if (basesSeen >= numBases[blocki] && blocki < numFrames.size() - 1)
        {
            lastReferenceFrame += numFrames[blocki];
            ++blocki;
            basesSeen = 0;
        }

        // estimate the current location by adding preceding frames to the
        // previous estimate
        framei += ipds[basei];

        // limit our estimate to the MF block boundaries
        // this does mean that several bases may have the same estimated
        // location
        framei = std::min(std::max(framei, lastReferenceFrame),
                          lastReferenceFrame + static_cast<size_t>(numFrames[blocki]));

        base2frame[basei] = framei;
        // we want the startframe of each base, so this is just added as an offset
        // at the end
        framei += pws[basei];
        ++basesSeen;
    }
}

std::string PeakRSSinGiB()
{
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(3)
       << PacBio::Utility::MemoryConsumption::PeakRss() / 1024.0 / 1024.0 / 1024.0;
    return ss.str();
}

} // anonymous


/**
 * @brief Default ctor.
 * @details Creates instances of different labeler, BazReader, 
 *          and ResultWriter.
 *          Fills thread pool.
 * 
 * @param user unique_pointer with UserParameter from command-line.
 */
ConvertBaz2Bam::ConvertBaz2Bam(std::shared_ptr<UserParameters>& user) 
    : abortNow_(false)
    , threadpoolContinue_(true)
    , ipcReceiverContinue_(true)
    , loggingContinue_(true)
    , numProcessedZMWs_(0)
    , threadCount_(0)
    , ppaAlgoConfig_(std::make_shared<PpaAlgoConfig>())
    , user_(user)
    , raiiSignalHander_(SIGPIPE,abortNow_)
{
    maxNumZmwsToProcess_ = user_->maxNumZmwsToProcess;
}

void ConvertBaz2Bam::ParseRMD()
{
    // TODO: We want to move away from using the MetadataParser to the one
    // supplied by pbbam.
    try
    {
        // Parse RuntimeMetaData
        rmd_ = MetadataParser::ParseRMD(bazReader_->Fileheader(), user_);
    }
    catch (const InvalidSequencingChemistryException&)
    {
        Announce("ppa/error",
                 "{\"acqId\":\"" + rmd_->subreadSet.uniqueId + "\"," +
                 "\"message\":\"INVALID_SEQUENCING_CHEMISTRY_EXCEPTION\"}");
        throw;
    }
}

void ConvertBaz2Bam::InitPpaAlgoConfig()
{
    try
    {
        cmd_ = std::make_shared<PacBio::BAM::CollectionMetadata>
                (PacBio::BAM::RunMetadata::Collection(user_->runtimeMetaDataFilePath));
        ppaAlgoConfig_->SetPlatformDefaults(rmd_->platform);
        if (cmd_->HasPPAConfig())
            throw PBException("Collection metadata should not have PPAConfig block");

        ppaAlgoConfig_->Populate(*cmd_);
        ppaAlgoConfig_->Populate(user_.get());

        PacBio::BAM::PPAConfig ppaConfig;
        ppaConfig.Json(ppaAlgoConfig_->RenderJSON());
        cmd_->PPAConfig(ppaConfig);

        PBLOG_INFO << "PPAAlgoConfig = " << ppaAlgoConfig_->RenderJSON();
    }
    catch (const std::exception& e)
    {
        Announce("ppa/error",
                 "{\"acqId\":\"" + rmd_->subreadSet.uniqueId + "\"" +
                 ",\"message\":\"" + e.what() + "\"}");
        PBLOG_ERROR << e.what();
        throw PBExceptionRethrow("Unable to parse collection metadata",e);
    }
}


int ConvertBaz2Bam::Run()
{

    if (!user_->silent)
    {
        std::cerr << "Version: " << versionString << std::endl;
        std::cerr << "Commit hash: " << cmakeGitHash() << std::endl;
        std::cerr << "Commit date: " << cmakeGitCommitDate() << std::endl;
    }

    InitLogToDisk();

    InitIPC();

    PBLOG_INFO << "Start";
    PBLOG_INFO << user_->originalCommandLine;
    PBLOG_INFO << "commit hash: " << cmakeGitHash();
    PBLOG_INFO << "commit date: " << cmakeGitCommitDate();
    PBLOG_INFO << "Version: " << versionString;

    {
        std::ostringstream message;
        message << "{" <<
                "\"acqId\":\"" << user_->uuid << "\"," <<
                "\"numZmwsSoFar\":0," <<
                "\"numZmwsTotal\":" << maxNumZmwsToProcess_ << // this is not accurate, but an upper bound
                "}";
        Announce("ppa/start", message.str());
    }

    CheckInputFile();

    if (maxNumZmwsToProcess_ != std::numeric_limits<uint32_t>::max())
    {
        PBLOG_INFO << "Roughly processing the first " << maxNumZmwsToProcess_ << " ZMWs, up to a slice boundary...";
    }


    // Parse BAZ.
    bazReader_ = std::unique_ptr<BazReader>(new BazReader(
            user_->inputFilePath, user_->zmwBatchMB, user_->zmwHeaderBatchMB, true,
            [this]() {
                if (!this->abortNow_) this->EmitHeartbeat();
                return this->abortNow_.load();
            }));

    if (!abortNow_)
    {
        numZmwsToProcess_ = NumZmwsToProcess();
    }

    try
    {
        ParseRMD();
        InitPpaAlgoConfig();
        // compare UUIDs, if both are presented. This is a sanity test for running underneath pa-ws, to make sure
        // that both baz2bam and pa-ws are using the same run metadata.
        if (user_->uuid != "" && rmd_->subreadSet.uniqueId != "" &&
            user_->uuid != rmd_->subreadSet.uniqueId)
        {
            throw PBException("Inconsistent UUIDs. RMD UUID:" + rmd_->subreadSet.uniqueId + " --uuid:" + user_->uuid);
        }
    }
    catch (const std::exception& e)
    {
        PBLOG_ERROR << "Exception during ParseRMD or InitPpaAlgoConfig";
        Announce("ppa/error",
                 "{\"acqId\":\"" + rmd_->subreadSet.uniqueId + "\"" +
                 ",\"message\":\"" + e.what() + "\"}");
        throw;
    }

    try
    {
        if (!abortNow_)
        {
            // Labeling, also writes barcoding header info
            subreadLabeler_ = std::unique_ptr<SubreadLabeler>(new SubreadLabeler(user_, rmd_, ppaAlgoConfig_));
        }
    }
    catch (const std::exception& e)
    {
        PBLOG_ERROR << "Exception during SubreadLabel ctor or ChipLayout::Factory";
        Announce("ppa/error",
                 "{\"acqId\":\"" + rmd_->subreadSet.uniqueId + "\"" +
                 ",\"message\":\"" + e.what() + "\"}");
        throw;
    }

    // Prepare output
    ProgramInfo pg;
    pg.Name("baz2bam")
            .Version(versionString)
            .Id("baz2bam")
            .CommandLine(user_->originalCommandLine);

    CreateWhiteList();

    resultWriter_ = std::unique_ptr<ResultWriter>(new ResultWriter(
            user_.get(),
            cmd_.get(),
            rmd_,
            ppaAlgoConfig_.get(),
            bazReader_->Fileheader(),
            {pg},
            !user_->noStats,
            NumZmwsToProcess()));

    if (!abortNow_)
    {
        hqRegionFinder_ = HQRegionFinderFactory(*user_, ppaAlgoConfig_, bazReader_->Fileheader());

        insertFinder_ = InsertFinderFactory(*user_);

        prodMetrics_ = std::make_unique<ProductivityMetrics>(
                ppaAlgoConfig_->inputFilter.minSnr,
                user_->minEmptyTime,
                user_->emptyOutlierTime);

        if (!user_->noStatsH5)
        {
            std::string stsH5Filename = user_->outputPrefix + ".sts.h5";
            CreateZmwStatsFile(stsH5Filename);
        }
    }
    else
    {
        resultWriter_->Abort();
    }

    LogStart();

    PopulateWorkerThreadPool();

    {
        std::ostringstream message;
        message << "{" <<
                "\"acqId\":\"" << rmd_->subreadSet.uniqueId << "\"," <<
                "\"numZmwsSoFar\":0," <<
                "\"numZmwsTotal\":" << NumZmwsToProcess() <<
                "}";
        Announce("ppa/start", message.str());
    }
    StartLoggingThread();

    ParseInput();

    // Wait until all reads have been processed
    while (!readList_.empty() && !abortNow_)
        std::this_thread::sleep_for(std::chrono::seconds(1));

    // Force threadpool to teminate
    threadpoolContinue_ = false;
    for (auto& t : threadpool_)
        t.join();

    // The pbindex-ing can take some time to finish and
    // close and can take a while to complete so
    // we need to keep reporting the progress until it is done
    // especially if it is not running inline. We also
    // account for the closing of the sts.h5 file which can
    // also take a while.
    auto closeFuture = std::async(std::launch::async, [this] {

        Validation valid = Validation::NOT_RUN;
        if (!resultWriter_)
        {
            PBLOG_WARN << "No Result writer created";
        }
        else
        {
            PBLOG_INFO << "Closing BAM files";
            PBLOG_INFO << "Writing PBI files";
            valid = this->resultWriter_->CloseAndValidate();
            PBLOG_INFO << "BAM files closed and PBIs written";
            resultWriter_.reset();
        }

        if (zmwStatsFile_)
        {
            PBLOG_INFO << "Closing sts.h5 file";
            zmwStatsFile_.reset();
            PBLOG_INFO << "sts.h5 file closed and written";
        }

        // Stop logging
        loggingContinue_ = false;
        if (loggingThread_.joinable())
        {
            PBLOG_INFO << "Closing logging";
            loggingThread_.join();
            PBLOG_INFO << "Logging closed";
        }

        PBLOG_INFO << "Finished ZMW processing " << numProcessedZMWs_;

        return valid;
    });

    double iProgress = 0;
    while (closeFuture.wait_for(std::chrono::seconds(1)) != std::future_status::ready)
    {
        if (!abortNow_)
        {
            iProgress += 1.0;
            // pretend that the progress is always (n-1)/n. Ie. the progress sequence within this loop is
            // 1/2, 2/3, 3/4, 4/5, ... Always optimistic that the end is nigh, and monotonically increasing.
            // Then that faction is interpolated between 0.99999 and 1.0000
            double progress = (iProgress / (iProgress + 1.0)) * .00001 + 0.99999;
            std::ostringstream message;
            message << "{" <<
                    "\"progress\":" << std::setprecision(12) << std::fixed << progress << "," <<
                    "\"acqId\":\"" << rmd_->subreadSet.uniqueId << "\"," <<
                    "\"numZmwsSoFar\":" << numProcessedZMWs_ << "," <<
                    "\"numZmwsTotal\":" << NumZmwsToProcess() << "," <<
                    "\"peakRss\":" << PeakRSSinGiB() <<
                    "}";
            PBLOG_INFO << message.str();
            Announce("ppa/progress", message.str());
        }
    }

    Validation valid = Validation::NOT_RUN;
    try
    {
        valid = closeFuture.get();
    }
    catch (const std::exception& ex)
    {
        Announce("ppa/error",
                 "{\"acqId\":\"" + rmd_->subreadSet.uniqueId + "\"" +
                 ",\"message\":\"Exception in closing thread:" + ex.what() + "\"}");
        throw;
    }

    if (valid == Validation::CLOSED_TRUNCATED)
    {
        Announce("ppa/error",
                 "{\"acqId\":\"" + rmd_->subreadSet.uniqueId + "\"" +
                 ",\"message\":\"TRUNCATED_BAM\"}");
        PBLOG_ERROR << "TRUNCATED_BAM";
    }
    else if (valid == Validation::NOT_RUN)
    {
        Announce("ppa/error",
                 "{\"acqId\":\"" + rmd_->subreadSet.uniqueId + "\"" +
                 ",\"message\":\"VALIDATION_NOTRUN\"}");
        PBLOG_ERROR << "VALIDATION_NOTRUN";
    }



#ifdef BUILD_ZMQ
    if (!abortNow_ && rmd_)
    {
        PBLOG_INFO << "Sending progress 1.0";

        std::ostringstream message;
        message << "{" <<
           "\"progress\": " << 1.0 << "," <<
           "\"acqId\":\""  << rmd_->subreadSet.uniqueId << "\"," <<
           "\"numZmwsSoFar\":" << numProcessedZMWs_ << "," <<
           "\"numZmwsTotal\":" << NumZmwsToProcess() << "," <<
           "\"peakRss\":" << PeakRSSinGiB() <<
           "}";

        Announce("ppa/progress", message.str());
        Announce("ppa/complete", "{\"acqId\":\"" + rmd_->subreadSet.uniqueId + "\"}");
        PBLOG_INFO << "Progress 1.0 sent";
    }
    DisconnectIPC();
    PBLOG_INFO << "IPC disconnected";
#endif


    if (abortNow_)
    {
        PBLOG_ERROR << "Aborting";
        return 1;
    } 
    else 
    {
        PBLOG_INFO << "Finished";
        return 0;
    }
}

// The segments we'll profile in the read thread
SMART_ENUM(
        READ_PROFILES,
        TOTAL,
        FULL_QUEUE,
        READ_DATA,
        SEND_DATA
)

using DiskProfiler = PacBio::Dev::Profile::ScopedProfilerChain<READ_PROFILES>;

/**
 * @brief Orchestrates processing batches of ZMW-slices.
 */
void ConvertBaz2Bam::ParseInput()
{
    size_t iterations = 0;
    // Loop as long as file has not been read completely
    while (!abortNow_ && bazReader_->HasNext())
    {
        auto mode = iterations > 5 ? DiskProfiler::Mode::OBSERVE : DiskProfiler::Mode::REPORT;
        DiskProfiler profiler(mode, 3, std::numeric_limits<float>::max());
        auto fullExecution = profiler.CreateScopedProfiler(READ_PROFILES::TOTAL);
        (void)fullExecution;
        // Avoid reading everything in memory, as data processing is the
        // limiting factor. We need to wait until enough reads have been
        // converted to BamAlignments, before we continue.
        if (readList_.NumBytes() > (user_->maxInputQueueMB << 20))
        {
            auto filledQueue = profiler.CreateScopedProfiler(READ_PROFILES::FULL_QUEUE);
            (void)filledQueue;
            while (readList_.NumBytes() > (user_->maxInputQueueMB << 20))
            {
                PBLOG_INFO << "[ParseInput] Input buffer full, waiting: " << readList_.NumZmw();
                std::this_thread::sleep_for(std::chrono::seconds(1));
                if (abortNow_) break;
            }
        }
        
        auto reading = profiler.CreateScopedProfiler(READ_PROFILES::READ_DATA);
        (void)reading;
        if (!performWhiteList_)
        {
            // Get new chunk
            auto readsTmp = bazReader_->NextSlice([this]() {
                if (!this->abortNow_) this->EmitHeartbeat();
                return this->abortNow_.load();
            });
            auto sending = profiler.CreateScopedProfiler(READ_PROFILES::SEND_DATA);
            (void)sending;
            readList_.Append(std::move(readsTmp));
        }
        else
        {
            const auto nextIds = bazReader_->NextZmwIds();
            std::vector<int> intersection;
            std::set_intersection(whiteListZmwIds_.begin(), whiteListZmwIds_.end(),
                                  nextIds.begin(), nextIds.end(),
                                  std::back_inserter(intersection));

            if (intersection.empty())
            {
                bazReader_->SkipNextSlice();
                continue;
            }
            auto readsTmp = bazReader_->NextSlice([this]() {
                if (!this->abortNow_) this->EmitHeartbeat();
                return this->abortNow_.load();
            });

            auto sending = profiler.CreateScopedProfiler(READ_PROFILES::SEND_DATA);
            (void)sending;
            for (auto& zmw : readsTmp)
            {
                if(std::find(whiteListZmwIds_.begin(), whiteListZmwIds_.end(), zmw.ZmwIndex()) != whiteListZmwIds_.end())
                {
                    bool stop = zmw.ZmwIndex() == whiteListZmwIds_.back();
                    readList_.Append(std::move(zmw));
                    if (stop)
                    {
                        return;
                    }
                }
            }
        }
        if(static_cast<uint32_t>(numProcessedZMWs_) >= maxNumZmwsToProcess_)
        {
            abortNow_ = true;
        }
    }
    DiskProfiler::FinalReport();
}

void ConvertBaz2Bam::InitIPC()
{
#ifdef BUILD_ZMQ
    if (user_->zmq)
    {
        std::lock_guard<std::mutex> l(announceMutex_);

        const PacBio::IPC::MessageQueueLabel ppaStatusQueue ("ppa-status" , PacBio::Primary::PORT_STAT_BAZ2BAM);
        PBLOG_INFO << "Initialize IPC MessageSocketPublisher on port "
                   << ppaStatusQueue.ToString();

        try
        {
            publisher_.reset(new IPC::MessageSocketPublisher(ppaStatusQueue));
        }
        catch (...)
        {
            std::string error = "Could not open MessageSocketPublisher on port "
                                + ppaStatusQueue.ToString();
            PBLOG_FATAL << error;
            exit(1);
        }
        PBLOG_INFO << "Initialization MessageSocketPublisher success";
        std::this_thread::sleep_for(std::chrono::seconds(1));
        // Init receiver as a thread
        ipcReceiverThread_ = std::thread([this]()
        {
            const PacBio::IPC::MessageQueueLabel ppaCommandQueue("ppa-command", PacBio::Primary::PORT_CTRL_BAZ2BAM);
            PBLOG_INFO << "Initialize IPC MessageSocketReceiver on port "
                       << ppaCommandQueue.ToString();
            std::unique_ptr<IPC::MessageSocketReceiver> receiveQueue;
            try
            {
                receiveQueue.reset(new IPC::MessageSocketReceiver(ppaCommandQueue, 100));
            }
            catch (...)
            {
                std::string error = "Could not open MessageSocketReceiver on port "
                                    + ppaCommandQueue.ToString();
                PBLOG_FATAL << error;
                exit(1);
            }
            PBLOG_INFO << "Initialization MessageSocketReceiver success";
            while (ipcReceiverContinue_)
            {
                const IPC::SmallMessage rxMsg(receiveQueue->Receive());
                if (rxMsg.IsAvailable())
                {
                    try
                    {
                        rxMsg.CheckValidity();
                        PBLOG_INFO << "ProcessBase::MainEventLoop::mesg: " << rxMsg;
                        if (rxMsg.GetName() == "ppa/command")
                        {
                            if (rxMsg.GetData() == "abort")
                            {
                                abortNow_ = true;
                                if (resultWriter_)
                                {
                                    PBLOG_INFO << "Aborting ResultWriter";
                                    resultWriter_->Abort();
                                }
                                std::cerr << std::endl << "ABORT" << std::endl;
                                Announce("ppa/error",
                                         "{\"acqId\":\"" + user_->uuid + "\"" +
                                         ",\"message\":\"ABORTED_BY_PAWS\"}");
                                break;
                            }
                        }
                    }
                    catch (std::exception& ex)
                    {
                        PBLOG_ERROR << "ProcessBase::MainEventLoop::mesg: exception " << ex.what();
                    }
                }
            }
        });
    }
#endif
}

void ConvertBaz2Bam::StartLoggingThread()
{
    using namespace std::chrono;
    
    uint32_t toDo = NumZmwsToProcess();

    loggingThread_ = std::thread([this,toDo](){
        // Logging
        auto startTime = std::chrono::high_resolution_clock::now();
        auto lastTime = startTime;

        int oldProcessedZmws = this->numProcessedZMWs_;
        int barWidth = 50;
        int iterations = 0;

        double smoothedRate = 0.0;
        double maxDeltaSeconds = 0;
        OomScoreMonitor oomScoreMonitor;
        while(loggingContinue_ && !abortNow_)
        {
            oomScoreMonitor.PollOomScore();

            int numProcessedZMWs = this->numProcessedZMWs_; // latch the value here
            // Only log if there has been an update
            if (numProcessedZMWs > oldProcessedZmws)
            {
                const auto deltaZMW = numProcessedZMWs - oldProcessedZmws;
                const auto now = high_resolution_clock::now();
                const auto deltaT = duration_cast<nanoseconds>(now - lastTime).count();
                const auto fullT = duration_cast<nanoseconds>(now - startTime).count();

                maxDeltaSeconds = std::max(maxDeltaSeconds, deltaT / 1e9);
                oldProcessedZmws = numProcessedZMWs;
                lastTime = now;

                // Compute simplistic ETA in nanoseconds.  This computation is
                // mostly used to bootstrap, but is also preserved in case someone 
                // wants to switch back to it over the smoothed version.
                double fullRate = static_cast<double>(numProcessedZMWs) / fullT;
                //const uint64_t etaSimple = numProcessedZMWs > 0 ?
                //    (double)(toDo - numProcessedZMWs) / fullRate : 0;

                // Compute more nuanced ETA, based on an exponential moving 
                // average.  The resulting ETA may be a bit more noisy (e.g. skips
                // around), but should also be more accurate in the event baz2bam
                // does not run at a uniform speed.
                double marginalRate = static_cast<double>(deltaZMW) / deltaT;
                // Controls how much "memory" is in our average.  We can
                // potentially have gaps where no zmw are completed, and we
                // need to make sure our look back is significantly larger to
                // avoid unnecessary oscilations.  
                double lookback = std::max(333.0, 8*maxDeltaSeconds);
                double seconds = deltaT / 1e9;
                // Need a few iterations to build up our running average before
                // switching to exponetial weighted average.
                if (iterations > 100)
                {
                    // If measurements are at regular intervals, you just have
                    // a simple "smoothedRate = alpha*marginalRate + (1-alpha)*smoothedRate"
                    // However our input are not at regular intervals and we have to use
                    // the sum of a geometric series in order to properly weight
                    // each input. e.g. if our `seconds = 2`, we should have: 
                    // smoothedRate = alpha*marginalRate * alpha ( 1 - alpha) * marginalRate * (1 - alpha)^2 * smoothedRate.
                    // Extrapolating to arbitrary `seconds = k` and simplifying and
                    // we get the below expression.
                    double alpha = 1 / lookback;
                    double beta = 1 - alpha;
                    double betaK = std::pow(beta, seconds);
                    smoothedRate = alpha * marginalRate * (1 - betaK) / (1 - beta) + betaK * smoothedRate;
                } else {
                    // Fallback to simple average if we've not enough history yet.  
                    smoothedRate = fullRate;
                }
                const uint64_t etaMovingAverage = numProcessedZMWs > 0 ?
                    (double)(toDo - numProcessedZMWs) / smoothedRate : 0;

                // Convert ETA to human readable form
                const auto d =  etaMovingAverage / 10000 / 1000 / 1000 / 60 / 60 / 24;
                const auto h = (etaMovingAverage / 1000  / 1000 / 1000 / 60 / 60) % 24;
                const auto m = (etaMovingAverage / 1000  / 1000 / 1000 / 60) % 60;
                const auto s = (etaMovingAverage / 1000  / 1000 / 1000) % 60;
                // Only report reasonable times
                std::stringstream ss;
                if (d > 0) ss << d << "d ";
                if (h > 0) ss << h << "h ";
                if (m > 0 && !d) ss << m << "m ";
                if (s > 0 && !d && !h) ss << s << "s ";
                // Compute progress bar
                double progress = 0.99999 * static_cast<double>(numProcessedZMWs) / toDo;
                if (progress >= 0.99999) progress = 0.99999;
                if (!user_->silent && isatty(fileno(stdout)))
                {
                    std::cout << "[";
                    int pos = static_cast<int>(std::floor(barWidth * progress));
                    for (int i = 0; i < barWidth; ++i) {
                        if (i < pos) std::cout << "=";
                        else if (i == pos) std::cout << ">";
                        else std::cout << " ";
                    }
                    float f = 100.0f * this->numProcessedZMWs_ / toDo;
                    std::cout << "] " << std::fixed << std::setprecision(2) << f
                              << "% (" << smoothedRate * 1e9 << " ZMWs/sec) ETA: " << ss.str() << "\r";
                    std::cout.flush();
                }
       
                iterations++;
                PBLOG_INFO << "Sending progress " + std::to_string(progress);

                // Even though this is a JSON message, it is prettier to format it as a string
                // because of the precision settings. Saving the floating point values using JSON::Value
                // results in truncated decimal points (for progress) and ugly precision for peakRss.
                std::ostringstream message;
                currentProgress_ = progress;
                message << "{" <<
                        "\"progress\":" << std::setprecision(12) << std::fixed << progress << "," <<
                        "\"acqId\":\""  << rmd_->subreadSet.uniqueId << "\"," <<
                        "\"numZmwsSoFar\":" << numProcessedZMWs_ << "," <<
                        "\"numZmwsTotal\":" << NumZmwsToProcess() << "," <<
                        "\"peakRss\":" << PeakRSSinGiB() <<
                        "}";
                Announce("ppa/progress", message.str());
            }
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        if (!user_->silent && !abortNow_ && isatty(fileno(stdout)))
        {
            int pos = barWidth;
            std::cout << "[";
            for (int i = 0; i < barWidth; ++i) {
                if (i < pos) std::cout << "=";
                else if (i == pos) std::cout << ">";
                else std::cout << " ";
            }
            std::cout << "] " << std::fixed << std::setprecision(2) << 100.00
                      << "%, finished." << std::endl;;
            Timing::PrintTime(startTime, "Time elapsed");
            std::cout << "Number of ZMWs\t: " << toDo << std::endl;
        }
    });
}

// The segments we'll profile in the compute thread.
SMART_ENUM(
        COMPUTE_PROFILES,
        REFACTOR,
        DATA_WAIT,
        DATA_MOVE,
        DATA_SEND,
        PARSE_BINARY,
        PROCESS_PACKETS,
        FIND_HQRF,
        FIND_INSERTS,
        PROD_STATS,
        FIND_SUBREAD,
        ZMW_STATS,
        FINALIZE
);
using Profiler = PacBio::Dev::Profile::ScopedProfilerChain<COMPUTE_PROFILES>;


// Destructor
ConvertBaz2Bam::~ConvertBaz2Bam()
{
    Profiler::FinalReport();

    // verify that all threads are joined.
    threadpoolContinue_ = false;
    ipcReceiverContinue_ = false;
    loggingContinue_ = false;
    PBLOG_INFO << "joining thread pool";
    for(auto& t: threadpool_) if (t.joinable()) t.join();

    PBLOG_INFO << "joining loggingThread_";
    if (loggingThread_.joinable()) loggingThread_.join();
    PBLOG_INFO << "joining ipcReceiverThread_";
    if (ipcReceiverThread_.joinable()) ipcReceiverThread_.join();
    PBLOG_INFO << "joining done";

    PBLOG_INFO << "Peak RSS      : " << PeakRSSinGiB() << " GiB";


    PBLOG_INFO << "~ConvertBaz2BamDestroyed";
}

/**
 * @brief Single thread of thread pool.
 * @details Responsible for taking a batch of ZMWs of the readList_,
 *          label and split reads, and queue them for writing to disk.
 */
void ConvertBaz2Bam::SingleThread()
{
    using PacBio::Primary::Postprimary::ZmwStats;

    int zmwCount = 0;

    // Loop until Workflow destructor asks for termination
    while(threadpoolContinue_ && !abortNow_)
    {
        Profiler::Mode mode = Profiler::Mode::REPORT;
        if (zmwCount < 1000) mode = Profiler::Mode::OBSERVE;
        Profiler profiler(mode, 3.0f, std::numeric_limits<float>::max());

        size_t numZmws = 0;
        auto waitProfile = profiler.CreateScopedProfiler(COMPUTE_PROFILES::DATA_WAIT);
        (void)waitProfile;
        std::vector<ZmwByteData> batch;
        int64_t batchId = -1;
        {
            if (readList_.NumZmw() > 0)
            {
                auto dataProfile = profiler.CreateScopedProfiler(COMPUTE_PROFILES::DATA_MOVE);
                (void)dataProfile;

                // Keep track how many threads are working.
                threadCount_++;

                // defaultZmwBatchSize is a request, need to query to find out how much we actually got
                batchId = readList_.Extract(user_->defaultZmwBatchSize, &batch);
                numZmws = batch.size();
            }
            else
            {
                numZmws = 0;
            }
        }
        zmwCount += numZmws;
        if (numZmws > 0)
        {
            assert(batchId >= 0);
            // Result vector
            std::vector<ComplexResult> resultBufferTmp;

            auto& fileHeader = bazReader_->Fileheader();
            auto& fileFooter = bazReader_->Filefooter();

            auto bufferList = zmwStatsFile_->GetZmwStatsBufferList();

            // Split and process each polymeraseread
            size_t startIndex = batch[0].ZmwIndex();
            for (size_t i = 0; i < numZmws; ++i)
            {
                if (abortNow_) break;

                // Lambda just used to avoid having unnecessary invalid variables
                // after doing an `std::move`
                EventData events = [&]()
                {
                    auto parseProfile = profiler.CreateScopedProfiler(COMPUTE_PROFILES::PARSE_BINARY);
                    (void)parseProfile;
                    auto bazEvents = BazEventData(ParsePackets(fileHeader, batch[i]));
                    
                    // Remove bursts:
                    auto insertProfile = profiler.CreateScopedProfiler(COMPUTE_PROFILES::FIND_INSERTS);
                    (void)insertProfile;
                    std::vector<InsertState> insertStates = insertFinder_->ClassifyInserts(bazEvents);

                    auto processProfile = profiler.CreateScopedProfiler(COMPUTE_PROFILES::PROCESS_PACKETS);
                    (void)processProfile;
                    const bool truncated = fileFooter->IsZmwTruncated(batch[i].ZmwIndex());
                    return EventData(fileHeader, batch[i].ZmwIndex(), truncated, std::move(bazEvents), std::move(insertStates));
                }();

                auto parseProfile = profiler.CreateScopedProfiler(COMPUTE_PROFILES::PARSE_BINARY);
                (void)parseProfile;
                auto bazMetrics = ParseMetrics(fileHeader, batch[i], events.Internal());

                auto hqrfProfile = profiler.CreateScopedProfiler(COMPUTE_PROFILES::FIND_HQRF);
                (void)hqrfProfile;
                RegionLabels regions;
                regions.hqregion = hqRegionFinder_->FindAndAnnotateHQRegion(bazMetrics, events);

                auto pstatsProfile = profiler.CreateScopedProfiler(COMPUTE_PROFILES::PROD_STATS);
                (void)pstatsProfile;
                auto prod = prodMetrics_->ComputeProductivityInfo(events.ZmwNumber(), regions.hqregion, bazMetrics);

                if (prod.productivity == ProductivityClass::EMPTY)
                {
                    // Remove the HQ-region if the ZMW is marked empty.
                    regions.hqregion.begin = regions.hqregion.end = 0;
                    regions.hqregion.pulseBegin = regions.hqregion.pulseEnd = 0;
                }

                auto subreadProfile = profiler.CreateScopedProfiler(COMPUTE_PROFILES::FIND_SUBREAD);
                (void)subreadProfile;

                // Label and cut into subreads as ResultPackets

                // "existing' isControl is always false for baz2bam, because
                // we're not pulling a prior from the BAM file
                const auto& controlMetrics = subreadLabeler_->CallControls(
                    prod, events, regions.hqregion, false);
                AdapterMetrics adapterMetrics;
                std::tie(regions.adapters,
                         adapterMetrics,
                         regions.hqregion) = subreadLabeler_->CallAdapters(
                    events, regions.hqregion, controlMetrics.isControl);

                if (!events.Internal())
                    ApproximateBase2Frame(bazMetrics, events, regions);
                else
                    PopulateBase2Frame(events, regions);

                auto cutResults = subreadLabeler_->ReadToResultPacket(
                    prod, events, regions, 0, controlMetrics);

                auto zstatsProfile = profiler.CreateScopedProfiler(COMPUTE_PROFILES::ZMW_STATS);
                (void)zstatsProfile;
                ZmwMetrics zmwMetrics(bazReader_->Fileheader(),
                                      regions.hqregion,
                                      regions.adapters,
                                      bazMetrics,
                                      events,
                                      prod,
                                      controlMetrics,
                                      adapterMetrics);
                // If user wants stats
                if (!user_->noStats)
                {
                    if (zmwStatsFile_)
                    {
                        if (zmwStatsFile_->IsBuffered())
                        {
                            bufferList->emplace_back(zmwStatsFile_->GetZmwStatsTemplate());
                            PacBio::Primary::ZmwStats& zmw = bufferList->back();
                            zmw.index_ = i + startIndex;
                            ZmwStats::FillPerZmwStats(rmd_->platform, fileHeader, regions.hqregion,
                                                      zmwMetrics, events, bazMetrics,
                                                      controlMetrics.isControl, user_->diagStatsH5, zmw);
                        }
                        else
                        {
                            PacBio::Primary::ZmwStats zmwStats{zmwStatsFile_->GetZmwStatsTemplate()};
                            ZmwStats::FillPerZmwStats(rmd_->platform, fileHeader, regions.hqregion,
                                                      zmwMetrics, events, bazMetrics,
                                                      controlMetrics.isControl, user_->diagStatsH5, zmwStats);
                            zmwStatsFile_->Set(i + startIndex, zmwStats);
                        }
                    }
                }

                auto finProfile = profiler.CreateScopedProfiler(COMPUTE_PROFILES::FINALIZE);
                (void)finProfile;
                bool rejected = fileHeader.IsZmwNumberRejected(events.ZmwNumber());
                for (auto& result : cutResults)
                {
                    result.AddTagsToRecord();
                    result.rejected = rejected;
                }
                // resultBufferTmp.emplace_back(std::move(cutResults));
                resultBufferTmp.emplace_back(std::move(cutResults),
                                             std::move(zmwMetrics));


                // Timing::PrintTime(now3, "Label");
            }
            this->numProcessedZMWs_ += numZmws;

            auto sendProfile = profiler.CreateScopedProfiler(COMPUTE_PROFILES::DATA_SEND);
            (void)sendProfile;
            // Timing::PrintTime(now3, "Convert");
            // std::cerr << "ADD" << std::endl;
            // auto now3 = std::chrono::high_resolution_clock::now();
            resultWriter_->AddResultsToBuffer(batchId, std::move(resultBufferTmp));
            // Timing::PrintTime(now3, "Add");
            if (zmwStatsFile_)
            {
                if (zmwStatsFile_->IsBuffered())
                {
                    zmwStatsFile_->WriteBuffers(std::move(bufferList));
                }
            }
            threadCount_--;
        }
        else
        // No reads available. 
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
}

void ConvertBaz2Bam::LogStart()
{
    PBLOG_INFO << "Starting Baz2bam.\t"
               << "MovieName: " <<  bazReader_->Fileheader().MovieName()
               << ",  BaseCallerVersion: " << bazReader_->Fileheader().BaseCallerVersion()
               << ",  BazWriterVersion: " << bazReader_->Fileheader().BazWriterVersion()
               << ",  BazVersion: " << bazReader_->Fileheader().BazVersion()
               << ",  OutputPrefix: " << user_->outputPrefix
               << ",  OutputPrefix: " << user_->outputPrefix
               << ",  Baz2bamVersion: " << versionString;
}

void ConvertBaz2Bam::InitLogToDisk()
{
    // Logging
    std::string logFilePrefix = user_->outputPrefix + ".baz2bam_";
    int logFileCounter = 1;

    bool logFileMissing = false;
    std::string logFileName;
    do {
        logFileName = logFilePrefix + std::to_string(logFileCounter) + ".log";
        struct stat logFilebuffer;   
        logFileMissing = stat (logFileName.c_str(), &logFilebuffer) != 0;
        if (!logFileMissing) logFileCounter++;
        if (logFileCounter == 999)
        {
            std::cerr << "Cannot write log file with prefix " << logFilePrefix << std::endl;
            outputLog = false;
        }
    }
    while(!logFileMissing);

    boost::log::settings settings;
    settings["Sinks.File.Destination"] = "TextFile";
    settings["Sinks.File.Format"] = ">|> %TimeStamp% -|- %Severity% -|- %Channel% -|- %HostName%|P:%PBProcessID%|T:%PBThreadID% -|- %Message%";
    settings["Sinks.File.FileName"] = logFileName;
    settings["Sinks.File.AutoFlush"] = true;
    settings["Sinks.File.RotationSize"] = 100 * 1024 * 1024; // 10 MiB
    Logging::PBLogger::InitPBLogging(settings);
    if (!user_->silent) std::cout << "Writing log to: " << logFileName << std::endl;
}

void ConvertBaz2Bam::CheckInputFile()
{
    // Test if input file exists
    struct stat buffer;   
    if (stat (user_->inputFilePath.c_str(), &buffer) != 0) 
    {
        Announce("ppa/error",
                 "{\"message\":\"INVALID_INPUT_FILE\"}");
        throw PBException("Input file \"" + user_->inputFilePath + "\" does not exist.");
    }
    // Test if input is a directory, if so, die
    if ((buffer.st_mode & S_IFMT) == S_IFDIR)
    {
        Announce("ppa/error",
                 "{\"message\":\"INPUT_FILE_IS_DIRECTORY\"}");
        throw PBException("Input file \"" + user_->inputFilePath + "\" is a directory.");
    }
}

void ConvertBaz2Bam::PopulateWorkerThreadPool()
{
    // Fill thread pool
    PBLOG_INFO << "Spawn " << user_->threads << " workers";
    threadpool_.reserve(user_->threads);
    for (int i = 0; i < user_->threads; ++i)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        threadpool_.emplace_back(std::thread([this,i](){
            try
            {
                SingleThread();
            }
            catch(const std::exception& ex)
            {
                PBLOG_ERROR << "Exception caught in SingleThread " << i << ":" << ex.what();
            }
            catch(const H5::Exception& ex)
            {
                std::cerr << "H5::Exception caught in SingleThread: " << ex.getCDetailMsg() << std::endl;
                PBLOG_FATAL << "H5::Exception caught: " << ex.getCDetailMsg();
            }
            catch(...)
            {
                std::cerr << "Unknown exception caught in SingleThread" << std::endl;
                PBLOG_FATAL << "Unknown exception caught in SingleThread";
            }
        }));
    }
}

void ConvertBaz2Bam::CreateWhiteList()
{
    bool filterNumber = false;
    if (!user_->whiteListZmwIds.empty())
    {
        performWhiteList_ = true;
        whiteListZmwIds_ = std::move(user_->whiteListZmwIds);
    } 
    else if (!user_->whiteListZmwNumbers.empty())
    {
        performWhiteList_ = true;
        filterNumber = true;
        whiteListZmwIds_ = std::move(user_->whiteListZmwNumbers);
    }

    std::sort(whiteListZmwIds_.begin(), whiteListZmwIds_.end());
    auto last = std::unique(whiteListZmwIds_.begin(), whiteListZmwIds_.end());
    whiteListZmwIds_.erase(last, whiteListZmwIds_.end());

    if (filterNumber)
    {
        const auto& fh = bazReader_->Fileheader();
        for (auto& w : whiteListZmwIds_) w = fh.ZmwNumberToId(w);
        std::sort(whiteListZmwIds_.begin(), whiteListZmwIds_.end());
    }
}

void ConvertBaz2Bam::CreateZmwStatsFile(const std::string& filename)
{
    PacBio::Primary::ZmwStatsFile::NewFileConfig config;

    config.numHoles = NumZmwsToProcess();

    config.numAnalogs = 4; // fixme
    switch (rmd_->platform)
    {
        case Platform::SEQUEL:
            config.numFilters = 2;
            break;
        case Platform::SEQUELII:
            config.numFilters = 1;
            break;
        default:
            throw PBException("Only SEQUEL and SEQUELII platforms supported for creating sts.h5");
            break;
    }

    const auto& fh = bazReader_->Fileheader();
    config.binSize = fh.LFMetricFrames();
    config.mfBinSize = fh.MFMetricFrames();
    config.numFrames = fh.MovieLengthFrames();
    config.addDiagnostics = user_->diagStatsH5;

    zmwStatsFile_.reset(new PacBio::Primary::ZmwStatsFile(filename, config, false));

    PBLOG_INFO << "Creating sts.h5 file " << filename << " with parameters numHoles:" << config.numHoles
                << " numAnalogs:" << config.numAnalogs
                << " numFilters:" << config.numFilters
                << " numTimeslices:" << config.NumTimeslices()
                << " LastBinFrames:" << config.LastBinFrames();

    zmwStatsFile_->WriteScanData(fh.ExperimentMetadata());
}

