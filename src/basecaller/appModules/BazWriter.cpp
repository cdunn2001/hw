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

#include <appModules/BazWriter.h>

#include <pacbio/logging/Logger.h>

#include <bazio/file/FileHeaderBuilder.h>
#include <common/MongoConstants.h>
#include <dataTypes/Pulse.h>
#include <dataTypes/PulseGroups.h>
#include <dataTypes/configs/SmrtBasecallerConfig.h>


using namespace PacBio::Primary;
using namespace PacBio::Mongo;
using namespace PacBio::Mongo::Data;

namespace PacBio {
namespace Application {

struct BazWriterBody::BazWriter
{
    virtual void Flush(std::unique_ptr<BazIO::BazBuffer> buffer) = 0;
    virtual void WaitForTermination() = 0;
};

struct BazWriterBody::SingleBazWriter : public BazWriterBody::BazWriter
{
    SingleBazWriter(const std::string& bazName,
                    size_t expectedFrames,
                    const std::vector<uint32_t>& zmwNumbers,
                    const std::vector<uint32_t>& zmwFeatures,
                    const std::map<uint32_t, BatchDimensions>& poolDims,
                    const SmrtBasecallerConfig& basecallerConfig)
    {
        PBLOG_INFO << "Opening BAZ file for writing: " << bazName << " zmws: " << zmwNumbers.size();

        const auto metricFrames = basecallerConfig.algorithm.Metrics.framesPerHFMetricBlock;

        using FileHeaderBuilder = BazIO::FileHeaderBuilder;
        FileHeaderBuilder fh(bazName,
                             100.0f,
                             expectedFrames,
                             basecallerConfig.internalMode
                             ? Mongo::Data::InternalPulses::Params() : Mongo::Data::ProductionPulses::Params(),
                             SmrtData::MetricsVerbosity::MINIMAL,
                             "",
                             basecallerConfig.Serialize().toStyledString(),
                             zmwNumbers,
                             zmwFeatures,
                             // Hack, until metrics handling can be rewritten
                             metricFrames,
                             metricFrames,
                             metricFrames);

        fh.BaseCallerVersion("0.1");

        bazWriter_ = std::make_unique<BazIO::BazWriter>(bazName, fh, BazIOConfig{});
    }

    void Flush(std::unique_ptr<BazIO::BazBuffer> buffer)
    {
        bazWriter_->Flush(std::move(buffer));
    }

    void WaitForTermination()
    {
        bazWriter_->WaitForTermination();
    }

    ~SingleBazWriter()
    {
        bazWriter_.reset();
    }

    std::unique_ptr<BazIO::BazWriter> bazWriter_;
};

struct BazWriterBody::MultipleBazWriter : public BazWriterBody::BazWriter
{
    MultipleBazWriter(const std::string& bazName,
                      size_t expectedFrames,
                      const std::vector<uint32_t>& zmwNumbers,
                      const std::vector<uint32_t>& zmwFeatures,
                      const std::map<uint32_t, BatchDimensions>& poolDims,
                      const SmrtBasecallerConfig& basecallerConfig)
    {
        PBLOG_INFO << "Opening multiple BAZ files for writing: " << bazName << " zmws: " << zmwNumbers.size();

        const auto metricFrames = basecallerConfig.algorithm.Metrics.framesPerHFMetricBlock;

        for (const auto& kv : poolDims)
        {
            using FileHeaderBuilder = BazIO::FileHeaderBuilder;
            FileHeaderBuilder fh(bazName,
                                 100.0f,
                                 expectedFrames,
                                 basecallerConfig.internalMode
                                 ? Mongo::Data::InternalPulses::Params() : Mongo::Data::ProductionPulses::Params(),
                                 SmrtData::MetricsVerbosity::MINIMAL,
                                 "",
                                 basecallerConfig.Serialize().toStyledString(),
                                 zmwNumbers,
                                 zmwFeatures,
                                 // Hack, until metrics handling can be rewritten
                                 metricFrames,
                                 metricFrames,
                                 metricFrames);

            fh.BaseCallerVersion("0.1");

            bazWriters_[kv.first] = std::make_unique<BazIO::BazWriter>(bazName, fh, BazIOConfig{});
        }
    }

    void Flush(std::unique_ptr<BazIO::BazBuffer> buffer)
    {
        bazWriters_[buffer->PoolId()]->Flush(std::move(buffer));
    }

    void WaitForTermination()
    {
        for (auto& kv : bazWriters_)
            kv.second->WaitForTermination();
    }

    ~MultipleBazWriter()
    {
        for (auto& kv : bazWriters_)
            kv.second.reset();
    }

    std::map<uint32_t, std::unique_ptr<BazIO::BazWriter>> bazWriters_;
};

BazWriterBody::BazWriterBody(
        const std::string& bazName,
        size_t expectedFrames,
        const std::vector<uint32_t>& zmwNumbers,
        const std::vector<uint32_t>& zmwFeatures,
        const std::map<uint32_t, BatchDimensions>& poolDims,
        const SmrtBasecallerConfig& basecallerConfig)
    : bazName_(bazName)
{
    if (basecallerConfig.multipleBazFiles)
        bazWriter_ = std::make_unique<MultipleBazWriter>(bazName, expectedFrames, zmwNumbers,
                                                         zmwFeatures, poolDims, basecallerConfig);
    else
        bazWriter_ = std::make_unique<SingleBazWriter>(bazName, expectedFrames, zmwNumbers,
                                                       zmwFeatures, poolDims, basecallerConfig);
}

void BazWriterBody::Process(std::unique_ptr<BazIO::BazBuffer> in)
{
    bazWriter_->Flush(std::move(in));
}

BazWriterBody::~BazWriterBody()
{
    PBLOG_INFO << "Closing BAZ file: " << bazName_;
    bazWriter_->WaitForTermination();
    bazWriter_.reset();
}

}}
