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

#ifndef PACBIO_APPLICATION_BAZ_WRITER_H
#define PACBIO_APPLICATION_BAZ_WRITER_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <pacbio/primary/BazWriter.h>

#include <common/graphs/GraphNodeBody.h>

#include <dataTypes/BasecallerConfig.h>
#include <dataTypes/BatchResult.h>

namespace PacBio {
namespace Application {

class NoopBazWriterBody final : public Graphs::LeafBody<Mongo::Data::BatchResult>
{
public:
    size_t ConcurrencyLimit() const override { return 1; }
    float MaxDutyCycle() const override { return 0.01; }

    void Process(Mongo::Data::BatchResult) override
    {
    }
};

class BazWriterBody final : public Graphs::LeafBody<Mongo::Data::BatchResult>
{
public:
    BazWriterBody(const std::string& bazName,
                  size_t expectedFrames,
                  const std::vector<uint32_t>& zmwNumbers,
                  const std::vector<uint32_t>& zmwFeatures,
                  const Mongo::Data::BasecallerConfig& basecallerConfig,
                  size_t outputStride);

    ~BazWriterBody()
    {
        PBLOG_INFO << "Closing BAZ file: " << bazName_;
        bazWriter_->Flush();
        bazWriter_->WaitForTermination();
        bazWriter_.reset();
    }

    size_t ConcurrencyLimit() const override { return 1; }
    float MaxDutyCycle() const override { return 1; }

    void Process(Mongo::Data::BatchResult in) override;

    using BazWriter = PacBio::Primary::BazWriter<Primary::SpiderMetricBlock>;
    std::unique_ptr<BazWriter> bazWriter_;
    std::string bazName_;
    size_t zmwOutputStrideFactor_;
    size_t currFrame_ = 0;
};


}}

#endif //PACBIO_APPLICATION_BAZ_WRITER_H
