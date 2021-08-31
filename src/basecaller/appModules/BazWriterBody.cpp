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

#include <bazio/file/FileHeaderBuilder.h>
#include <dataTypes/Pulse.h>
#include <dataTypes/PulseGroups.h>
#include <dataTypes/configs/SmrtBasecallerConfig.h>

using namespace PacBio::Primary;
using namespace PacBio::Mongo;
using namespace PacBio::Mongo::Data;

namespace {

inline std::string generateExperimentMetadata(const MovieConfig& movieConfig)
{
    // See PTSD-677
    PBLOG_WARN << "Implementation gap: Generating fake Metadata for baz files";

    std::string basemap;
    for (const auto& analog : movieConfig.analogs)
        basemap.push_back(analog.baseLabel);
    std::ostringstream metadata;
    metadata << "{\"ChipInfo\":{\"LayoutName\":\"";
    metadata << "SequelII";
    metadata << "\"},\"DyeSet\":{\"BaseMap\":\"";
    metadata << basemap;
    metadata << "\",\"RelativeAmp\":";
    metadata << "[";
    std::string sep = "";
    for (const auto& analog : movieConfig.analogs)
    {
        metadata << sep << analog.relAmplitude;
        sep = ",";
    }
    metadata << "]}}";
    return metadata.str();
}

}


namespace PacBio {
namespace Application {

BazWriterBody::BazWriterBody(
        const std::string& bazName,
        size_t expectedFrames,
        const std::vector<uint32_t>& zmwNumbers,
        const std::vector<uint32_t>& zmwFeatures,
        const std::map<uint32_t, BatchDimensions>& poolDims,
        const SmrtBasecallerConfig& basecallerConfig,
        const Mongo::Data::MovieConfig& movieConfig)
    : numThreads_(basecallerConfig.system.ioConcurrency)
    , numBatches_(poolDims.size())
    , multipleBazFiles_(basecallerConfig.multipleBazFiles)
    , bazName_(bazName)
{
    const auto metricFrames = basecallerConfig.algorithm.Metrics.framesPerHFMetricBlock;

    const auto& metadata = generateExperimentMetadata(movieConfig);
    const auto pulseSerializationConfig = basecallerConfig.internalMode
        ? Mongo::Data::InternalPulses::Params()
        : Mongo::Data::ProductionPulses::Params();

    if (multipleBazFiles_)
    {
        auto ioStatsAggregator = std::make_shared<BazIO::BazWriter::IOStatsAggregator>(numBatches_);
        auto removeExtension = [](const std::string& fn) {
            size_t lastDot = fn.find_last_of(".");
            if (lastDot == std::string::npos) return fn;
            return fn.substr(0, lastDot);
        };

        PBLOG_INFO << "Opening " << numBatches_ << " BAZ files for writing with filename prefix: "
                   << removeExtension(bazName) << " zmws: " << zmwNumbers.size();

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

        bazWriters_.resize(numBatches_);
        std::atomic<uint32_t> openedFiles = 0;
        tbb::parallel_for((uint32_t) {0}, numBatches_, [&](uint32_t b)
        {
            auto poolZmwNumbersStart = zmwNumbers.begin() + batchStartZmw[b];
            auto poolZmwFeaturesStart = zmwFeatures.begin() + batchStartZmw[b];
            using FileHeaderBuilder = BazIO::FileHeaderBuilder;
            std::string multiBazName = removeExtension(bazName) + "." + std::to_string(b) + ".baz";
            FileHeaderBuilder fh(multiBazName,
                                 100.0f,
                                 expectedFrames,
                                 pulseSerializationConfig,
                                 SmrtData::MetricsVerbosity::MINIMAL,
                                 metadata,
                                 basecallerConfig.Serialize().toStyledString(),
                                 std::vector<uint32_t>(poolZmwNumbersStart, poolZmwNumbersStart + batchNumZmw[b]),
                                 std::vector<uint32_t>(poolZmwFeaturesStart, poolZmwFeaturesStart + batchNumZmw[b]),
                                 // Hack, until metrics handling can be rewritten
                                 metricFrames,
                                 metricFrames,
                                 metricFrames);

            fh.BaseCallerVersion("0.1");

            bazWriters_[b] = std::make_unique<BazIO::BazWriter>(multiBazName, fh, basecallerConfig.bazio, ioStatsAggregator);
            auto openedSnapshot = ++openedFiles;
            if (openedSnapshot % 10 == 0) PBLOG_INFO << "Opened " << openedSnapshot << " baz files so far";
        });
        PBLOG_INFO << "Finished opening a total of " << numBatches_ << " baz files";
        assert(openedFiles == numBatches_);
    }
    else
    {
        PBLOG_INFO << "Opening BAZ file for writing: " << bazName << " zmws: " << zmwNumbers.size();

        using FileHeaderBuilder = BazIO::FileHeaderBuilder;
        FileHeaderBuilder fh(bazName,
                             100.0f,
                             expectedFrames,
                             pulseSerializationConfig,
                             SmrtData::MetricsVerbosity::MINIMAL,
                             metadata,
                             basecallerConfig.Serialize().toStyledString(),
                             zmwNumbers,
                             zmwFeatures,
                             // Hack, until metrics handling can be rewritten
                             metricFrames,
                             metricFrames,
                             metricFrames);

        fh.BaseCallerVersion("0.1");

        bazWriters_.push_back(std::make_unique<BazIO::BazWriter>(bazName, fh, basecallerConfig.bazio));
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
