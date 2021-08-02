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
#include <bazio/FileHeaderBuilder.h>
#include <pacbio/primary/ZmwResultBuffer.h>

#include <common/MongoConstants.h>

#include <dataTypes/configs/SmrtBasecallerConfig.h>
#include <dataTypes/Pulse.h>

using namespace PacBio::Primary;
using namespace PacBio::Mongo;
using namespace PacBio::Mongo::Data;

namespace PacBio {
namespace Application {

BazWriterBody::BazWriterBody(
        const std::string& bazName,
        size_t expectedFrames,
        const std::vector<uint32_t>& zmwNumbers,
        const std::vector<uint32_t>& zmwFeatures,
        const SmrtBasecallerConfig& basecallerConfig)
    : bazName_(bazName)
{
    PBLOG_INFO << "Opening BAZ file for writing: " << bazName_ << " zmws: " << zmwNumbers.size();

    const auto metricFrames = basecallerConfig.algorithm.Metrics.framesPerHFMetricBlock;

    FileHeaderBuilder fh(bazName_,
                         100.0f,
                         expectedFrames,
                         // This is a hack, we need to hand in the new encoding params
                         basecallerConfig.internalMode ? SmrtData::Readout::PULSES : SmrtData::Readout::BASES,
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

    bazWriter_ = std::make_unique<BazIO::BazWriter>(bazName_, fh, BazIOConfig{});
}

void BazWriterBody::Process(std::unique_ptr<BazIO::BazBuffer> in)
{
    bazWriter_->Flush(std::move(in));
}

}}
