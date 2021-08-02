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

#include <algorithm>
#include <assert.h>
#include <cctype>
#include <chrono>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <utility>

#include <pacbio/logging/Logger.h>

#include <pbbam/EntireFileQuery.h>
#include <pbbam/RunMetadata.h>
#include <pbbam/virtual/VirtualZmwBamRecord.h>

#include <bazio/BazReader.h>
#include <bazio/Codec.h>
#include <bazio/MetricFieldName.h>
#include <bazio/PacketField.h>
#include <bazio/PacketFieldName.h>
#include <bazio/SmartMemory.h>
#include <bazio/Timing.h>

#include <postprimary/bam/ComplexResult.h>
#include <postprimary/bam/Platform.h>
#include <postprimary/bam/ResultPacket.h>
#include <postprimary/bam/ResultWriter.h>
#include <postprimary/bam/RuntimeMetaData.h>
#include <postprimary/bam/SubreadLabeler.h>
#include <postprimary/stats/ZmwMetrics.h>

#include <pbcopper/utility/MemoryConsumption.h>

#include "ConvertBam2Bam.h"
#include "UserParameters.h"
#include "ppa-config.h"

#include <git-rev.h>

static const std::string versionString = cmakeGitBranch() + "_" + cmakeGitHash();

using namespace PacBio::Primary::Postprimary;

namespace { // anonymous
std::unordered_map<size_t, size_t> BamRecordsToBase2Frame(const BamSet& bamSet)
{
    std::unordered_map<size_t, size_t> base2frame;
    for (const auto& region : bamSet.records)
    {
        if (region.HasQueryStartFrameNumber() && region.HasQueryEndFrameNumber())
        {
            base2frame[region.QueryStart()] = region.QueryStartFrameNumber();
            // bam 0-based intervals are end-exclusive. The QueryEndFrameNumber
            // should actually be the first frame of the base before the index
            // specified in QueryEnd
            if (region.QueryEnd() > region.QueryStart())
                base2frame[region.QueryEnd() - 1] = region.QueryEndFrameNumber();
        }
        else
        {
            // If we run into a bam record without these tags, something is
            // wrong. Don't bother recording existing values for this ZMW
            return std::unordered_map<size_t, size_t>{};
        }
    }
    return base2frame;
}

std::unordered_map<size_t, size_t> StartFramesToBase2Frame(const EventData& events)
{
    std::unordered_map<size_t, size_t> base2frame;
    const auto& sf = events.StartFrames();
    for (size_t i = 0; i < events.NumBases(); ++i)
        base2frame[i] = sf[events.BaseToPulseIndex(i)];
    return base2frame;
}

} // anonymous

ConvertBam2Bam::ConvertBam2Bam(std::shared_ptr<UserParameters>& user) 
    : batchId(0)
    , numProcessedZMWs_(0)
    , threadCount_(0)
    , user_(user)
    , ppaAlgoConfig_(std::make_shared<PpaAlgoConfig>())
    , dataset_(user_->subreadsetFilePath)
{
    if (!user_->silent)
    {
        std::cerr << "Version: " << versionString << std::endl;
        std::cerr << "Commit hash: " << cmakeGitHash() << std::endl;
        std::cerr << "Commit date: " << cmakeGitCommitDate() << std::endl;
    }

    InitLogToDisk();
    PBLOG_INFO << "Start";
    PBLOG_INFO << user_->originalCommandLine;
    PBLOG_INFO << "commit hash: " << cmakeGitHash();
    PBLOG_INFO << "commit date: " << cmakeGitCommitDate();
    PBLOG_INFO << "Version: " << versionString;

    auto SetSubreadAndScrapsFile = [this](const RecordType& recordType,
                                          const ExternalResource& primaryResource,
                                          const BamConfig& bc)
    {
        this->primaryBamType_ = recordType;
        this->primaryBam_ = dataset_.ResolvePath(primaryResource.ResourceId());
        for (const ExternalResource& resource : primaryResource.ExternalResources())
        {
            if (resource.MetaType() == bc.secondaryMetaType)
            {
                this->scrapsBam_ = dataset_.ResolvePath(resource.ResourceId());
                break;
            }
        }
    };

    // Dataset input
    const ExternalResources& resources = dataset_.ExternalResources();
    for (const ExternalResource& resource : resources)
    {
        const auto& metatype = resource.MetaType();
        if (metatype == subreadConfig.primaryMetaType)
        {
            SetSubreadAndScrapsFile(RecordType::SUBREAD, resource, subreadConfig);
            break;
        }
        else if(metatype == hqConfig.primaryMetaType)
        {
            SetSubreadAndScrapsFile(RecordType::HQREGION, resource, hqConfig);
            break;
        }
        else if(metatype == polyConfig.primaryMetaType)
        {
            SetSubreadAndScrapsFile(RecordType::POLYMERASE, resource, polyConfig);
            break;
        }
    }

    if (primaryBam_.empty())
        throw std::runtime_error("Primary BAM file not set (subreads, hqregion, or polymerase)!");
    if (scrapsBam_.empty())
        throw std::runtime_error("Scraps BAM file not set!");

    whiteList_ = !user_->whiteListHoleNumbers.empty();
    if (!whiteList_)
        vpr_ = std::unique_ptr<ZmwReadStitcher>(
                new ZmwReadStitcher(user_->subreadsetFilePath));
    else
    {
        PBLOG_WARN << "Whitelist specified on command-line, ignoring filters in subreadset.xml";
        wvpr_ = std::unique_ptr<WhitelistedZmwReadStitcher>(
                new WhitelistedZmwReadStitcher(user_->whiteListHoleNumbers,
                                               primaryBam_, scrapsBam_));
    }

    // Prepare @PG apps
    std::vector<ProgramInfo> apps;
    // Get meta data from BAM headers
    int nextRunId = ParseHeader(&apps);

    InitPpaAlgoConfig();

    // Set to hqonly if requirements are met
    if (ppaAlgoConfig_->adapterFinding.disableAdapterFinding
        && ppaAlgoConfig_->controlFilter.disableControlFiltering
        && primaryBamType_ == RecordType::HQREGION
        && !user_->polymeraseread)
    {
        user_->hqonly = true;
    }

    // Don't call adapters in polymerase mode.
    if (user_->polymeraseread)
    {
        PBLOG_INFO << "Disabling adapter calling in polymerase mode";
        ppaAlgoConfig_->adapterFinding.disableAdapterFinding = true;
    }

    // Create subread labeler
    subreadLabeler_ = std::unique_ptr<SubreadLabeler>(new SubreadLabeler(user, rmd_, ppaAlgoConfig_));

    ProgramInfo pg;
    pg.Name("bam2bam")
      .Version(versionString)
      .Id("bam2bam_" + std::to_string(nextRunId))
      .CommandLine(user_->originalCommandLine);
    apps.emplace_back(pg);

    // Using the meta data, create output writer
    resultWriter_ = std::make_unique<ResultWriter>(user_.get(),
            nullptr,
            rmd_,
            ppaAlgoConfig_.get(),
            *fh_,
            apps,
            false,
            0 // this value means "don't care". The ResultWriter doesn't need this value.
                           // This value is only set to be passed downstream to CCS.
    );
    resultWriter_->SetCollectionMetadataDataSet(&ds_);

    // Start and fill thread pool
    threadpoolContinue_ = true;
    threadpool_.reserve(user_->threads);
    for (int i = 0; i < user_->threads; ++i)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        threadpool_.emplace_back(std::thread([this](){SingleThread();}));
    }

    // Activate cmd-line progress output
    if (!user_->silent && isatty(fileno(stdout)))
    {
        loggingContinue_ = true;
        numProcessedZMWs_ = 0;
        loggingThread_ = std::thread([this](){ 
            int oldProcessedZmws = -1;
            while(loggingContinue_)
            {
                if (this->numProcessedZMWs_ > oldProcessedZmws)
                {
                    oldProcessedZmws = this->numProcessedZMWs_;
                    std::cout << "Processed ZMWs: " << this->numProcessedZMWs_ << "\r";
                    std::cout.flush();
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
            std::cout << "Processed ZMWs: " << this->numProcessedZMWs_ << "\n";
        });
    }
    ParseInput();
}

void ConvertBam2Bam::InitPpaAlgoConfig()
{
    try
    {
        ds_ = PacBio::BAM::DataSet(user_->subreadsetFilePath);
        ppaAlgoConfig_->SetPlatformDefaults(rmd_->platform);

        if (ds_.Metadata().CollectionMetadata().HasPPAConfig())
        {
            // If the PPAConfig block already exists, we load it from the collection metadata.
            PBLOG_INFO << "Existing PPAConfig block found, loading from collection metadata";
            ppaAlgoConfig_->Load(ds_.Metadata().CollectionMetadata().PPAConfig().Json());
        }
        else
        {
            // No existing PPAConfig block,
            PBLOG_INFO << "No existing PPAConfig block, using collection metadata";
            ppaAlgoConfig_->Populate(ds_.Metadata().CollectionMetadata());
        }

        // Override with any user set command-line options.
        ppaAlgoConfig_->Populate(user_.get());

        PacBio::BAM::PPAConfig ppaConfig;
        ppaConfig.Json(ppaAlgoConfig_->RenderJSON());
        ds_.Metadata().CollectionMetadata().PPAConfig(ppaConfig);

        PBLOG_INFO << "PPAAlgoConfig = " << ppaAlgoConfig_->RenderJSON();
    }
    catch (const std::runtime_error& err)
    {
        PBLOG_ERROR << err.what();
        throw std::runtime_error("Unable to parse collection metadata");
    }
}

void ConvertBam2Bam::InitLogToDisk()
{
    // Logging
    std::string logFilePrefix = user_->outputPrefix + ".bam2bam_";
    int logFileCounter = 1;

    bool logFileMissing = false;
    std::string logFileName;
    do {
        logFileName = logFilePrefix + std::to_string(logFileCounter) + ".log";
        struct stat logFilebuffer;   
        logFileMissing = stat (logFileName.c_str(), &logFilebuffer) != 0;
        if (!logFileMissing) logFileCounter++;
        if (logFileCounter == 999)
            std::cerr << "Cannot write log file with prefix " << logFilePrefix << std::endl;
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

void ConvertBam2Bam::SetBamFile(const std::string& input)
{
    const auto bamType = ConvertBamType(input);
    switch (bamType)
    {
        case RecordType::HQREGION:
            primaryBamType_ = RecordType::HQREGION;
            primaryBam_ = input;
            break;
        case RecordType::SUBREAD:
            primaryBamType_ = RecordType::SUBREAD;
            primaryBam_ = input;
            break;
        case RecordType::SCRAP:
            scrapsBam_ = input;
            break;
        default:
            throw std::runtime_error("ZMW not allowed as input BAM file.");
    }
}

ConvertBam2Bam::~ConvertBam2Bam() 
{
    // Wait until all reads have been processed
    while (!recordSetList_.empty())
        std::this_thread::sleep_for(std::chrono::seconds(1));

    // Force threadpool to teminate
    threadpoolContinue_ = false;
    for (auto& t : threadpool_)
        t.join();

    PBLOG_INFO << "Closing BAM files";

    const Validation valid = resultWriter_->CloseAndValidate();
    if (valid == Validation::CLOSED_TRUNCATED)
        PBLOG_ERROR << "TRUNCATED_BAM";
    
    PBLOG_INFO << "Writing PBI files";
    resultWriter_.reset();

    PBLOG_INFO << "BAM files closed and PBIs written";

    PBLOG_INFO << "Peak RSS      : " << std::fixed << std::setprecision(3)
               << PacBio::Utility::MemoryConsumption::PeakRss() / 1024.0 / 1024.0 / 1024.0 << " GB";

    if (!user_->silent && isatty(fileno(stdout)))
    {
        // Stop logging
        loggingContinue_ = false;
        loggingThread_.join();
    }
}

RecordType ConvertBam2Bam::ConvertBamType(const std::string& input)
{
    BamFile bam(input);
    std::string globalReadType;
    for (const auto& rg : bam.Header().ReadGroups())
    {        

        auto readType = rg.ReadType();
        std::transform(readType.begin(), readType.end(), readType.begin(), ::toupper);
        if (globalReadType.empty())
            globalReadType = readType;
        else if (readType.compare(globalReadType) != 0)
            throw std::runtime_error("Multiple read groups in " +
                                     input +
                                     ". Could not identify bam type.");
    }
    if (globalReadType.compare("SUBREAD") == 0)
        return RecordType::SUBREAD;
    else if (globalReadType.compare("SCRAP") == 0)
        return RecordType::SCRAP;
    else if (globalReadType.compare("POLYMERASE") == 0 || globalReadType.compare("ZMW") == 0)
        return RecordType::ZMW;
    else if (globalReadType.compare("HQREGION") == 0)
        return RecordType::HQREGION;
    else
        throw std::runtime_error("Read type unknown.");
}

void ConvertBam2Bam::ParseInput()
{
    while((whiteList_ && wvpr_->HasNext()) || (!whiteList_ && vpr_->HasNext())) {
        mutex_.lock();
        size_t recordSetListSize = recordSetList_.size();
        mutex_.unlock();

        // Avoid reading everything in memory, as data processing is the
        // limiting factor. We need to wait until enough reads have been
        // converted to ResultPackets, before we continue.
        if (recordSetListSize > 10000) 
        {
            while (recordSetListSize > 5000)
            {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                mutex_.lock();
                recordSetListSize = recordSetList_.size();
                mutex_.unlock();
            }
        }

        // Get 1000 ZMWs per iteration
        std::vector<BamSet> bamrecordVecVec;
        bamrecordVecVec.reserve(1000);

        size_t counter = 0;
        if (!whiteList_)
        {
            while(vpr_->HasNext() && counter++ < 1000)
            {
                const auto h = vpr_->PrimaryHeader();
                bamrecordVecVec.push_back(BamSet(vpr_->NextRaw(), h.DeepCopy()));
            }
        }
        else
        {
            while(wvpr_->HasNext() && counter++ < 1000)
            {
                const auto h = wvpr_->PrimaryHeader();
                bamrecordVecVec.push_back(BamSet(wvpr_->NextRaw(), h.DeepCopy()));
            }
        }

        // Move StitchedZMWs to global recordSetList_
        mutex_.lock();
        MoveAppend(bamrecordVecVec, recordSetList_);
        mutex_.unlock();
    }
}

VirtualZmwBamRecord ConvertBam2Bam::BamRecordsToPolyRead(BamSet&& bamSet)
{
    return VirtualZmwBamRecord(std::move(bamSet.records),
                                      bamSet.header);
}

EventData ConvertBam2Bam::PolyReadToEventData(
    const VirtualZmwBamRecord& polyRead, size_t zmwId)
{
    bool internalMode = polyRead.HasPulseCall() ;

    // // Number of events (bases/pulses)
    uint64_t numEvents = internalMode
                            ? polyRead.PulseCall().size()
                            : polyRead.QueryEnd();

    // Reconstruct the packet info
    using Fields = BazIO::PacketFieldName;
    std::map<Fields, std::vector<uint32_t>> rawIntPackets;
    std::map<Fields, std::vector<float>> rawFloatPackets;

    std::vector<InsertState> insertStates;
    insertStates.reserve(numEvents);
    // Production mode
    if (!internalMode)
    {
        // Bases
        const std::string currentBases = polyRead.Sequence();
        AppendPacketFields(rawIntPackets[Fields::Label], currentBases, {}, '-');

        auto& pws = rawIntPackets[Fields::Pw];
        auto& startFrames = rawIntPackets[Fields::StartFrame];
        std::vector<uint32_t> ipds;

        AppendPacketFields(pws, polyRead.PulseWidthRaw().Data());
        AppendPacketFields(ipds, polyRead.IPDRaw().Data());
        AppendPacketFields(startFrames, polyRead.StartFrame());

        if (startFrames.size() == 0)
        {
            startFrames.reserve(ipds.size());
            // Note: there are still questions about lossy values, both from
            //       a codec that may or may not be applied, as well as from
            //       a saturated conversion to uint16_t that may be applied
            //       when writing to bam
            //       PTSD-568 is a followup story that should address these
            //       questions
            size_t frame = 0;
            for (size_t i = 0; i < ipds.size(); ++i)
            {
                frame += ipds[i];
                startFrames.push_back(frame);
                frame += pws[i];
            }
        }
    }
    else // Internal mode
    {
        // IsBase
        auto pulseCall = polyRead.PulseCall();

        auto& isBase = rawIntPackets[Fields::IsBase];
        isBase.reserve(pulseCall.size());
        for (const auto& c : pulseCall)
        {
            bool isSingleBase = (bool)std::isupper(c);
            isBase.push_back(isSingleBase);
        }

        std::transform(pulseCall.begin(), pulseCall.end(), pulseCall.begin(), ::toupper);
        AppendPacketFields(rawIntPackets[Fields::Label], pulseCall, {}, '-');

        AppendPacketFields(rawIntPackets[Fields::Pw], polyRead.PulseCallWidth().Data());
        AppendPacketFields(rawIntPackets[Fields::StartFrame], polyRead.StartFrame());
        if (polyRead.HasPkmean())
            AppendPacketFields(rawFloatPackets[Fields::Pkmean], polyRead.Pkmean());
        if (polyRead.HasPkmid())
            AppendPacketFields(rawFloatPackets[Fields::Pkmid], polyRead.Pkmid());
        if (polyRead.HasPulseExclusion())
        {
            // Pulse exclusion tag is stored explicitly.
            const auto& pe = polyRead.PulseExclusionReason();
            std::transform(pe.begin(), pe.end(), std::back_inserter(insertStates),
                           [](PacBio::BAM::PulseExclusionReason p)
                                { return static_cast<InsertState>(p); });
        } else {
            throw PBException("Pulse exclusion tag required for processing "
                              "internal mode bam file is missing.  "
                              "This bam file is either too old to be "
                              "supported by this version of bam2bam, or "
                              "otherwise nonconforming");
        }
    }
    for (const auto& vec : rawIntPackets)
    {
        if ((vec.second.size() != 0) && (vec.second.size() != numEvents))
        {
            throw PBException("Error reconstructing baz packets");
        }
    }
    for (const auto& vec : rawFloatPackets)
    {
        if ((vec.second.size() != 0) && (vec.second.size() != numEvents))
        {
            throw PBException("Error reconstructing baz packets");
        }
    }

    auto packets = BazIO::BazEventData(std::move(rawIntPackets), std::move(rawFloatPackets));
    if (insertStates.size() == 0)
    {
        insertStates = std::vector<InsertState>(
            packets.NumEvents(), InsertState::BASE);
    }

    return EventData(zmwId, std::move(packets),
                     std::move(insertStates));
}

double ConvertBam2Bam::PolyReadToBarcodeScore(
    const VirtualZmwBamRecord& polyRead)
{
    if (polyRead.HasBarcodeQuality())
        return polyRead.BarcodeQuality();
    return 0;
}

bool ConvertBam2Bam::PolyReadToIsControl(
    const VirtualZmwBamRecord& polyRead)
{
    if (ppaAlgoConfig_->controlFilter.disableControlFiltering
            && !user_->polymeraseread
            && polyRead.HasScrapZmwType() && user_->saveControls)
        return polyRead.ScrapZmwType() == ZmwType::CONTROL;
    return false;
}

RegionLabels ConvertBam2Bam::PolyReadToRegionLabels(
    const VirtualZmwBamRecord& polyRead,
    const EventData& events,
    const std::unordered_map<size_t, size_t> base2frame)
{
    RegionLabels regions;

    if (events.Internal())
        regions.base2frame = StartFramesToBase2Frame(events);
    else
        regions.base2frame = base2frame;

    bool callAdapters = !ppaAlgoConfig_->adapterFinding.disableAdapterFinding;

    if (!user_->polymeraseread)
    {
        int bcLeft = -1;
        int bcRight = -1;

        if (!callAdapters
            && polyRead.HasVirtualRegionType(VirtualRegionType::SUBREAD))
        {
            const auto subreads = polyRead.VirtualRegionsTable(
                VirtualRegionType::SUBREAD);
            for (const auto& tag : subreads)
                regions.inserts.emplace_back(
                        tag.beginPos,
                        tag.endPos,
                        42,
                        0,
                        RegionLabelType::INSERT);

            bcLeft = subreads[0].barcodeLeft;
            bcRight = subreads[0].barcodeRight;

            if (!callAdapters && !user_->hqonly)
                for (const auto& tag : subreads)
                    regions.cxTags.emplace_back(static_cast<uint8_t>(tag.cxTag));
        }

        if (bcLeft != bcRight)
            user_->scoreMode = BarcodeStrategy::ASYMMETRIC;

        if (!callAdapters
            && !user_->hqonly
            && polyRead.HasVirtualRegionType(VirtualRegionType::ADAPTER))
        {
            const auto adapter = polyRead.VirtualRegionsTable(
                VirtualRegionType::ADAPTER);
            for (const auto& tag : adapter)
                regions.adapters.emplace_back(
                    tag.beginPos,
                    tag.endPos,
                    42,
                    0,
                    RegionLabelType::ADAPTER);
        }

        if (!callAdapters
            && !user_->hqonly
            && polyRead.HasVirtualRegionType(VirtualRegionType::BARCODE))
        {
            const auto barcode = polyRead.VirtualRegionsTable(VirtualRegionType::BARCODE);
            int counter = 0;
            for (const auto& tag : barcode)
                regions.barcodes.emplace_back(
                    tag.beginPos,
                    tag.endPos,
                    1,
                    (counter++ % 2 == 0) ? bcLeft : bcRight,
                    RegionLabelType::BARCODE);
        }

        if (!user_->hqonly
            && polyRead.HasVirtualRegionType(VirtualRegionType::FILTERED))
        {
            const auto hqr = polyRead.VirtualRegionsTable(VirtualRegionType::FILTERED);
            for (const auto& tag : hqr)
                regions.filtered.emplace_back(
                    tag.beginPos,
                    tag.endPos,
                    42,
                    0,
                    RegionLabelType::FILTERED);
        }

        if (!user_->hqonly
            && polyRead.HasVirtualRegionType(VirtualRegionType::LQREGION))
        {
            const auto lq = polyRead.VirtualRegionsTable(VirtualRegionType::LQREGION);
            for (const auto& tag : lq)
                regions.lq.emplace_back(
                        tag.beginPos,
                        tag.endPos,
                        42,
                        0,
                        RegionLabelType::LQREGION);
        }

        if (!user_->hqrf)
        {
            regions.hqregion = RegionLabel(
                0,
                events.NumBases(),
                42,
                0,
                RegionLabelType::HQREGION);
        }
        else if (polyRead.HasVirtualRegionType(VirtualRegionType::HQREGION))
        {
            const auto hqr = polyRead.VirtualRegionsTable(VirtualRegionType::HQREGION);
            for (const auto& tag : hqr)
                regions.hqregion = RegionLabel(
                    tag.beginPos,
                    tag.endPos,
                    42,
                    0,
                    RegionLabelType::HQREGION);
        }
    }

    return regions;
}


ProductivityInfo ConvertBam2Bam::PolyReadToProductivityInfo(
    const VirtualZmwBamRecord& polyRead,
    const RegionLabel& hqregion)
{
    ProductivityInfo pinfo;
    // ReadAccuracy
    if (polyRead.HasReadAccuracy())
    {
        pinfo.readAccuracy = polyRead.ReadAccuracy();
    }

    // SNRs
    if (polyRead.HasSignalToNoise())
    {
        const auto& snrs = polyRead.SignalToNoise();
        pinfo.snr = AnalogMetricData<float>();
        pinfo.snr->A = snrs[0];
        pinfo.snr->C = snrs[1];
        pinfo.snr->G = snrs[2];
        pinfo.snr->T = snrs[3];

        float minSnr = *std::min_element(snrs.cbegin(), snrs.cend());
        pinfo.readAccuracy = minSnr >= ppaAlgoConfig_->inputFilter.minSnr ? 0.8f : 0.0f;
    }

    // Compute if P1
    if (hqregion.begin != hqregion.end)
    {
        const auto accuracy = pinfo.readAccuracy;
        auto prod = ProductivityMetrics::IsP1(hqregion, accuracy);
        // MDS TODO: ProductivityClass::OTHER isn't an option?
        pinfo.productivity = prod ? ProductivityClass::PRODUCTIVE
                                  : ProductivityClass::EMPTY;
    }
    else
    {
        pinfo.productivity = ProductivityClass::EMPTY;
    }
    pinfo.isSequencing = true;
    return pinfo;
}


void ConvertBam2Bam::SingleThread()
{
    // Loop until Workflow destructor asks for termination
    while(threadpoolContinue_)
    {
        mutex_.lock();
        // No reads available. This will only happen in the first seconds.
        if (recordSetList_.size() == 0)
        {
            mutex_.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        else
        {
            // We need to know the batch id to write the BAM 
            // in the correct order
            int currentBatch = batchId.fetch_add(1);

            // Keep track how many threads are working.
            threadCount_++;

            // Timing
            // auto now3 = std::chrono::high_resolution_clock::now();

            // Number of ZMWs in this batch
            size_t numZmws = recordSetList_.size() > batchSize ? 
                                batchSize : 
                                recordSetList_.size();

            // take a batch ZMWs from the list via move semantics
            std::vector<BamSet> batch(
                std::make_move_iterator(recordSetList_.begin()),
                std::make_move_iterator(recordSetList_.begin() + numZmws));

            // remove the dangling pointers
            recordSetList_.erase(recordSetList_.begin(), recordSetList_.begin() + numZmws);

            // Allow next thread to take reads
            mutex_.unlock();

            // Result vector
            std::vector<ComplexResult> resultBufferTmp;

            // Split and process each polymeraseread
            for (size_t i = 0; i < numZmws; ++i)
            {
                // std::this_thread::sleep_for(std::chrono::milliseconds(100));

                // Timing
                // auto now3 = std::chrono::high_resolution_clock::now();

                auto zmwId = batch[i].zmwId;

                // Don't keep ws/we tags if we're changing regions, e.g.
                // calling adapters:
                std::unordered_map<size_t, size_t> base2frame;
                if (!subreadLabeler_->PerformAdapterFinding())
                    base2frame = BamRecordsToBase2Frame(batch[i]);

                const auto& polyRead = BamRecordsToPolyRead(std::move(batch[i]));
                auto events = PolyReadToEventData(polyRead, zmwId);
                const auto barcodeScore = PolyReadToBarcodeScore(polyRead);
                const auto isControl = PolyReadToIsControl(polyRead);
                auto regions = PolyReadToRegionLabels(
                    polyRead, events, base2frame);
                const auto& pinfo = PolyReadToProductivityInfo(
                    polyRead, regions.hqregion);
                // PrintTime(now3, "ZMW");
                // now3 = std::chrono::high_resolution_clock::now();

                // Label and cut into subreads as ResulPackets
                auto controlMetrics = subreadLabeler_->CallControls(
                    pinfo, events, regions.hqregion, isControl);

                AdapterMetrics adapterMetrics;

                if (regions.adapters.size() == 0)
                {
                    std::tie(regions.adapters,
                             adapterMetrics,
                             regions.hqregion) = subreadLabeler_->CallAdapters(
                        events, regions.hqregion, controlMetrics.isControl, true);

                }
                auto cutResults = subreadLabeler_->ReadToResultPacket(
                    pinfo, events, regions, barcodeScore,
                    controlMetrics);

                ZmwMetrics zmwMetrics(regions.hqregion,
                                      regions.adapters,
                                      pinfo,
                                      controlMetrics,
                                      adapterMetrics,
                                      user_->runReport);

                for (auto& re : cutResults)
                    re.AddTagsToRecord();

                resultBufferTmp.emplace_back(std::move(cutResults),
                                             std::move(zmwMetrics));
                // PrintTime(now3, "Label");
            }
            this->numProcessedZMWs_ += numZmws;
            // PrintTime(now3, "Convert");
            // auto now3 = std::chrono::high_resolution_clock::now();
            resultWriter_->AddResultsToBuffer(currentBatch, std::move(resultBufferTmp));
            // PrintTime(now3, "Add");
            threadCount_--;
        }
    }
}

int ConvertBam2Bam::ParseHeader(std::vector<ProgramInfo>* apps)
{
    int nextBam2bamRun = 0;
    // Create new header
    fh_ = std::unique_ptr<FileHeader>(new FileHeader());
    // Creeate new RuntimeMetaData
    rmd_ = std::make_shared<RuntimeMetaData>();

    // Get headers from both bam files
    const auto subreadsHeader = !whiteList_ ? vpr_->PrimaryHeader() : wvpr_->PrimaryHeader();
    const auto scrapsHeader = !whiteList_ ? vpr_->ScrapsHeader() : wvpr_->ScrapsHeader();

    ParseHeader(subreadsHeader);
    ParseHeader(scrapsHeader);
 
    // Get bazwriter, baz2bam, bazFormat versiobns
    for (const auto& app : subreadsHeader.Programs())
    {
        if (app.Name().compare("bazwriter") == 0)
            fh_->BazWriterVersion(app.Version());
        else if (app.Name().compare("bazFormat") == 0)
        {
            // Convert string to major.minor.patch ints
            std::stringstream ss(app.Version());
            std::string item;
            int counter = 0;
            while(std::getline(ss, item, '.')) {
                switch (counter)
                {
                    case 0: fh_->BazMajorVersion(std::stoi(item)); break;
                    case 1: fh_->BazMinorVersion(std::stoi(item)); break;
                    case 2: fh_->BazPatchVersion(std::stoi(item)); break;
                }
                ++counter;
            }
        }
        else if (app.Name().compare("bam2bam") == 0)
        {
            if (app.Id().size() > 8)
            {
                const auto runId = std::stoi(app.Id().substr(8,app.Id().size()));
                nextBam2bamRun = std::max(nextBam2bamRun, 1 + runId);
            }
            apps->emplace_back(app);
        }
        else
            apps->emplace_back(app);
    }

    // Get one read group as a representative
    if (subreadsHeader.ReadGroups().empty())
        throw std::runtime_error("No read group in header");
    const auto rg = subreadsHeader.ReadGroups()[0];
    
    // Determine the different packet fields for the RG description
    for (const auto& feature : baseFeatures_)
    {
        switch(feature)
        {
            case BaseFeature::DELETION_QV:
                fh_->AddPacketField(PacketFieldName::DEL_QV);
                break;
            case BaseFeature::DELETION_TAG:
                fh_->AddPacketField(PacketFieldName::DEL_TAG);
                break;
            case BaseFeature::INSERTION_QV:
                fh_->AddPacketField(PacketFieldName::INS_QV);
                break;
            case BaseFeature::MERGE_QV:
                fh_->AddPacketField(PacketFieldName::MRG_QV);
                break;
            case BaseFeature::SUBSTITUTION_QV:
                fh_->AddPacketField(PacketFieldName::SUB_QV);
                break;
            case BaseFeature::SUBSTITUTION_TAG:
                fh_->AddPacketField(PacketFieldName::SUB_TAG);
                break;
            case BaseFeature::IPD:
                switch(rg.IpdCodec())
                {
                    case FrameCodec::RAW:
                        fh_->AddPacketField(PacketFieldName::IPD_LL);
                        break;
                    case FrameCodec::V1:
                        fh_->AddPacketField(PacketFieldName::IPD_V1);
                        break;
                    default:
                        throw std::runtime_error("No read group in header");
                }
                break;
            case BaseFeature::PULSE_WIDTH:
                switch(rg.PulseWidthCodec())
                {
                    case FrameCodec::RAW:
                        fh_->AddPacketField(PacketFieldName::PW_LL);
                        break;
                    case FrameCodec::V1:
                        fh_->AddPacketField(PacketFieldName::PW_V1);
                        break;
                    default:
                        throw std::runtime_error("No read group in header");
                }
                break;
            case BaseFeature::PKMID:
                fh_->AddPacketField(PacketFieldName::PKMID_LL);
                break;
            case BaseFeature::PKMEAN:
                fh_->AddPacketField(PacketFieldName::PKMEAN_LL);
                break;
            case BaseFeature::PKMID2:
                fh_->AddPacketField(PacketFieldName::PKMID2_LL);
                break;
            case BaseFeature::PKMEAN2:
                fh_->AddPacketField(PacketFieldName::PKMEAN2_LL);
                break;
            case BaseFeature::LABEL:
                fh_->AddPacketField(PacketFieldName::LABEL);
                break;
            case BaseFeature::LABEL_QV:
                fh_->AddPacketField(PacketFieldName::LAB_QV);
                break;
            case BaseFeature::ALT_LABEL:
                fh_->AddPacketField(PacketFieldName::ALT_LABEL);
                break;
            case BaseFeature::ALT_LABEL_QV:
                fh_->AddPacketField(PacketFieldName::ALT_QV);
                break;
            default: break; // Other fields are handled internally.
        }
    }

    return nextBam2bamRun;
}

void ConvertBam2Bam::ParseHeader(const BamHeader& header)
{
    for (const auto& rg : header.ReadGroups())
    {
        // Check Basecaller
        if (!rg.BasecallerVersion().empty())
        {
            if (rmd_->basecallerVersion.empty()) 
                rmd_->basecallerVersion = rg.BasecallerVersion();
            else if (rg.BasecallerVersion().compare(rmd_->basecallerVersion))
                throw std::runtime_error("BasecallerVersion differ!");
        }

        // Check BindingKit
        if (!rg.BindingKit().empty())
        {
            if (rmd_->bindingKit.empty()) 
                rmd_->bindingKit = rg.BindingKit();
            else if (rg.BindingKit().compare(rmd_->bindingKit))
                throw std::runtime_error("BindingKits differ!");
        }

        // Check SequencingKit
        if (!rg.SequencingKit().empty())
        {
            if (rmd_->sequencingKit.empty()) 
                rmd_->sequencingKit = rg.SequencingKit();
            else if (rg.SequencingKit().compare(rmd_->sequencingKit))
                throw std::runtime_error("SequencingKits differ!");
        }

        // Check BasecallerVersion
        if (!rg.BasecallerVersion().empty())
        {
            if (rmd_->basecallerVersion.empty()) 
                rmd_->basecallerVersion = rg.BasecallerVersion();
            else if (rg.BasecallerVersion().compare(rmd_->basecallerVersion))
                throw std::runtime_error("BasecallerVersions differ!");
        }

        // Check Platform
        PacBio::BAM::PlatformModelType pm = rg.PlatformModel();
        Platform pf;
        switch(pm)
        {
        case PlatformModelType::ASTRO: throw std::runtime_error("ASTRO not supported in BAM2BAM");
        case PlatformModelType::RS: pf = Platform::RSII; break;
        case PlatformModelType::SEQUEL: pf = Platform::SEQUEL; break;
        case PlatformModelType::SEQUELII: pf = Platform::SEQUELII; break;
        default:
            throw std::runtime_error("Unsupported source platform");
        }
        if (rmd_->platform == Platform::NONE)
            rmd_->platform = pf;
        else if (pf != rmd_->platform)
            throw std::runtime_error("Platforms differ!");

        // Check MovieName
        if (!rg.MovieName().empty())
        {
            if (rmd_->movieName.empty()) 
                rmd_->movieName = rg.MovieName();
            else if (rg.MovieName().compare(rmd_->movieName))
                throw std::runtime_error("MovieNames differ!");
        }

        // Check FrameRateHz
        if (!rg.FrameRateHz().empty())
        {
            double rgFr = std::stod(rg.FrameRateHz());
            if (fh_->FrameRateHz() == -1) 
                fh_->FrameRateHz(rgFr);
            else if (rgFr != fh_->FrameRateHz())
                throw std::runtime_error("FrameRateHz differ!");
        }

        // Check BasecallerVersion
        if (!rg.BasecallerVersion().empty())
        {
            if (fh_->BaseCallerVersion().empty()) 
                fh_->BaseCallerVersion(rg.BasecallerVersion());
            else if (rg.BasecallerVersion().compare(fh_->BaseCallerVersion()))
                throw std::runtime_error("BasecallerVersion differ!");
        }

        if (rg.HasBarcodeData())
        {
            if (rmd_->numBarcodes == -1)
                rmd_->numBarcodes = rg.BarcodeCount();
            else if (rmd_->numBarcodes != static_cast<int>(rg.BarcodeCount()))
                throw std::runtime_error("BarcodeCounts differ!");

            if (rmd_->barcodeFileName.empty())
                rmd_->barcodeFileName = rg.BarcodeFile();
            else if (rmd_->barcodeFileName.compare(rg.BarcodeFile()))
                throw std::runtime_error("BarcodeFile differ!");

            if (rmd_->barcodeHash.empty())
                rmd_->barcodeHash = rg.BarcodeHash();
            else if (rmd_->barcodeHash.compare(rg.BarcodeHash()))
                throw std::runtime_error("BarcodeHash differ!");

            rmd_->barcodeQualityScore = rg.BarcodeQuality() == BarcodeQualityType::SCORE;
            // Convert the PBBAM::BarcodeModeType to Labeler::Barcode::BarcodeStrategy
            switch (rg.BarcodeMode())
            {
                case BarcodeModeType::SYMMETRIC:
                    rmd_->barcodeScoreMode = BarcodeStrategy::SYMMETRIC;
                    break;
                case BarcodeModeType::ASYMMETRIC:
                    rmd_->barcodeScoreMode = BarcodeStrategy::ASYMMETRIC;
                    break;
                case BarcodeModeType::TAILED:
                    rmd_->barcodeScoreMode = BarcodeStrategy::TAILED;
                    break;
               default:
                    throw std::runtime_error("Invalid BarcodeModeType in BAM file!"); 
            }
        }

        AddBaseFeature(rg, BaseFeature::DELETION_QV);
        AddBaseFeature(rg, BaseFeature::DELETION_TAG);
        AddBaseFeature(rg, BaseFeature::INSERTION_QV);
        AddBaseFeature(rg, BaseFeature::MERGE_QV);
        AddBaseFeature(rg, BaseFeature::SUBSTITUTION_QV);
        AddBaseFeature(rg, BaseFeature::SUBSTITUTION_TAG);
        AddBaseFeature(rg, BaseFeature::IPD);
        AddBaseFeature(rg, BaseFeature::PULSE_WIDTH);
        AddBaseFeature(rg, BaseFeature::PKMID);
        AddBaseFeature(rg, BaseFeature::PKMEAN);
        AddBaseFeature(rg, BaseFeature::PKMID2);
        AddBaseFeature(rg, BaseFeature::PKMEAN2);
        AddBaseFeature(rg, BaseFeature::LABEL);
        AddBaseFeature(rg, BaseFeature::LABEL_QV);
        AddBaseFeature(rg, BaseFeature::ALT_LABEL);
        AddBaseFeature(rg, BaseFeature::ALT_LABEL_QV);
    }
}

void ConvertBam2Bam::AddBaseFeature(const ReadGroupInfo& rg, const BaseFeature& feature)
{
    if (rg.HasBaseFeature(feature)
        && std::find(baseFeatures_.cbegin(), baseFeatures_.cend(), feature) == baseFeatures_.cend())
        baseFeatures_.emplace_back(feature);
}

