// Copyright (c) 2020, Pacific Biosciences of California, Inc.
//
// All rights reserved.
//
// THIS SOFTWARE CONSTITUTES AND EMBODIES PACIFIC BIOSCIENCES' CONFIDENTIAL
// AND PROPRIETARY INFORMATION.
//
// Disclosure, redistribution and use of this software is subject to the
// terms and conditions of the applicable written agreement(s) between you
// and Pacific Biosciences, where "you" refers to you or your company or
// organization, as applicable.  Any other disclosure, redistribution or
// use is prohibited.
//
// THIS SOFTWARE IS PROVIDED BY PACIFIC BIOSCIENCES AND ITS CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL PACIFIC BIOSCIENCES OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
// OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
// ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <appModules/BazWriterBody.h>

#include <tbb/parallel_for.h>

#include <pacbio/logging/Logger.h>

#include <appModules/SmrtBasecallerProgress.h>
#include <bazio/file/FileHeaderBuilder.h>
#include <dataTypes/Pulse.h>
#include <dataTypes/PulseGroups.h>
#include <dataTypes/configs/SmrtBasecallerConfig.h>

using namespace PacBio::Primary;
using namespace PacBio::Mongo;
using namespace PacBio::Mongo::Data;

namespace PacBio {
namespace Application {

BazWriterBody::BazWriterBody(
        const std::string& bazName,
        size_t expectedFrames,
        const BazIO::ZmwInfo& zmwInfo,
        const std::map<uint32_t, BatchDimensions>& poolDims,
        const SmrtBasecallerConfig& basecallerConfig,
        const PacBio::File::ScanData::Data& experimentMetadata,
        SmrtBasecallerStageReporter& reporter)
    : numThreads_(basecallerConfig.system.ioConcurrency)
    , numBatches_(poolDims.size())
    , multipleBazFiles_(basecallerConfig.multipleBazFiles)
    , bazName_(bazName)
{
    const auto metricFrames = basecallerConfig.algorithm.Metrics.framesPerHFMetricBlock;
    const auto& metadata = experimentMetadata.Serialize().toStyledString();
    const auto pulseSerializationConfig = basecallerConfig.internalMode
        ? Mongo::Data::InternalPulses::Params()
        : Mongo::Data::ProductionPulses::Params();

    using FileHeaderBuilder = BazIO::FileHeaderBuilder;
    using ZmwInfo = BazIO::ZmwInfo;

    auto fileHeaderFlags = FileHeaderBuilder::Flags();
    fileHeaderFlags.RealTimeActivityLabels(basecallerConfig.prelimHQ.enablePreHQ);

    if (multipleBazFiles_)
    {
        auto ioStatsAggregator = std::make_shared<BazIO::IOStatsAggregator>(numBatches_);
        auto removeExtension = [](const std::string& fn) {
            size_t lastDot = fn.find_last_of(".");
            if (lastDot == std::string::npos) return fn;
            return fn.substr(0, lastDot);
        };

        PBLOG_INFO << "Opening " << numBatches_ << " BAZ files for writing with filename prefix: "
                   << removeExtension(bazName) << " zmws: " << zmwInfo.NumZmws();

        {
            std::string bazFileListFileName = removeExtension(bazName) + ".bazfilelist.txt";
            PBLOG_INFO << "Generating BAZ file list to file: " << bazFileListFileName;
            std::ofstream ostrm(bazFileListFileName, std::ios::trunc);
            if (ostrm.is_open())
            {
                for (uint32_t b = 0; b < numBatches_; b++)
                {
                    const std::string multiBazName = removeExtension(bazName) + "." + std::to_string(b) + ".baz";
                    ostrm << multiBazName << "\n";
                }
                ostrm.close();
            }
            else
            {
                throw PBException("Unable to write BAZ file list: " + bazFileListFileName);
            }
        }


        std::vector<size_t> batchStartZmw;
        std::vector<size_t> batchNumZmw;
        batchStartZmw.reserve(numBatches_);
        batchNumZmw.reserve(numBatches_);
        size_t startZmwNumber = 0;
        for (const auto& kv : poolDims)
        {
            batchStartZmw.push_back(startZmwNumber);
            batchNumZmw.push_back(kv.second.ZmwsPerBatch());
            startZmwNumber += batchNumZmw.back();
        }

        const auto& zmwNumbers = zmwInfo.HoleNumbers();
        const auto& zmwTypes = zmwInfo.HoleTypes();
        const auto& zmwX = zmwInfo.HoleX();
        const auto& zmwY = zmwInfo.HoleY();
        const auto& zmwFeatures = zmwInfo.HoleFeaturesMask();

        bazWriters_.resize(numBatches_);
        std::atomic<uint32_t> openedFiles = 0;
        tbb::parallel_for((uint32_t) {0}, numBatches_, [&](uint32_t b)
        {
            const auto poolZmwNumbersStart = zmwNumbers.begin() + batchStartZmw[b];
            const auto poolZmwTypesStart = zmwTypes.begin() + batchStartZmw[b];
            const auto poolZmwXStart = zmwX.begin() + batchStartZmw[b];
            const auto poolZmwYStart = zmwY.begin() + batchStartZmw[b];
            const auto poolZmwFeaturesStart = zmwFeatures.begin() + batchStartZmw[b];

            const std::string multiBazName = removeExtension(bazName) + "." + std::to_string(b) + ".baz";
            ZmwInfo zmwInfo(ZmwInfo::Data
                                ( std::vector<uint32_t>(poolZmwNumbersStart, poolZmwNumbersStart + batchNumZmw[b]),
                                  std::vector<uint8_t>(poolZmwTypesStart, poolZmwTypesStart + batchNumZmw[b]),
                                  std::vector<uint16_t>(poolZmwXStart, poolZmwXStart + batchNumZmw[b]),
                                  std::vector<uint16_t>(poolZmwYStart, poolZmwYStart + batchNumZmw[b]),
                                  std::vector<uint32_t>(poolZmwFeaturesStart, poolZmwFeaturesStart + batchNumZmw[b])
                                ));
            FileHeaderBuilder fh(multiBazName,
                                 100.0f,
                                 expectedFrames,
                                 pulseSerializationConfig,
                                 metadata,
                                 basecallerConfig.Serialize().toStyledString(),
                                 zmwInfo,
                                 metricFrames,
                                 fileHeaderFlags);

            // TODO: This needs to be more dynamic (or at least not hard coded in the bowels
            //       of the code) but for now this is necessary to remain compatible with
            //       CCS
            fh.BaseCallerVersion("5.0");

            bazWriters_[b] = std::make_unique<BazIO::BazWriter>(multiBazName, fh, basecallerConfig.bazIO, ioStatsAggregator);
            auto openedSnapshot = ++openedFiles;
            reporter.Update(1);
            if (openedSnapshot % 10 == 0) PBLOG_INFO << "Opened " << openedSnapshot << " baz files so far";
        });
        PBLOG_INFO << "Finished opening a total of " << numBatches_ << " baz files";
        assert(openedFiles == numBatches_);
    }
    else
    {
        PBLOG_INFO << "Opening BAZ file for writing: " << bazName << " zmws: " << zmwInfo.NumZmws();

        FileHeaderBuilder fh(bazName,
                             100.0f,
                             expectedFrames,
                             pulseSerializationConfig,
                             metadata,
                             basecallerConfig.Serialize().toStyledString(),
                             zmwInfo,
                             metricFrames,
                             fileHeaderFlags);
        // TODO: This needs to be more dynamic (or at least not hard coded in the bowels
        //       of the code) but for now this is necessary to remain compatible with
        //       CCS
        fh.BaseCallerVersion("5.0");

        bazWriters_.push_back(std::make_unique<BazIO::BazWriter>(bazName, fh, basecallerConfig.bazIO));
        reporter.Update(numBatches_);
        PBLOG_INFO << "Opened BAZ file for writing: " << bazName << " zmws: " << zmwInfo.NumZmws();
    }
}

BazWriterBody::~BazWriterBody()
{
    PBLOG_INFO << "Closing BAZ file: " << bazName_;
    std::atomic<uint32_t> openedFiles = bazWriters_.size();
    auto statsAggregator = bazWriters_.front()->GetAggregator();
    tbb::parallel_for(size_t{0}, bazWriters_.size(), [&](size_t b)
    {
        bazWriters_[b]->WaitForTermination();
        bazWriters_[b].reset();

        auto openedSnapshot = --openedFiles;
        if (openedSnapshot % 10 == 0) PBLOG_INFO << openedSnapshot << " baz files left to close";
    });
    assert(openedFiles == 0);
    {
        Logging::LogStream ls(Logging::LogLevel::INFO);
        statsAggregator->Summarize(ls);
    }
}

void BazWriterBody::Process(std::unique_ptr<BazIO::BazBuffer> in)
{
    bazWriters_[in->BufferId()]->Flush(std::move(in));
}

}}
