// Copyright (c) 2021, Pacific Biosciences of California, Inc.
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

#ifndef PACBIO_APPLICATION_PRELIM_HQ_FILTER_H
#define PACBIO_APPLICATION_PRELIM_HQ_FILTER_H

#include <common/graphs/GraphNodeBody.h>

#include <dataTypes/BatchResult.h>
#include <dataTypes/configs/ConfigForward.h>

#include <bazio/writing/BazBuffer.h>

namespace PacBio {
namespace Application {

class PrelimHQFilterBody final : public Graphs::MultiTransformBody<Mongo::Data::BatchResult, std::unique_ptr<BazIO::BazBuffer>>
{
public:
    PrelimHQFilterBody(size_t numZmws, const std::map<uint32_t, Mongo::Data::BatchDimensions>& poolDims,
                       const Mongo::Data::PrelimHQConfig& config, bool internal, bool multipleBazFiles);
    ~PrelimHQFilterBody();

    size_t ConcurrencyLimit() const override { return 1; }
    float MaxDutyCycle() const override { return 1; }

    void Process(Mongo::Data::BatchResult in) override;
private:
    struct SingleBuffer;
    struct MultipleBuffer;
    struct Impl;
    template <bool internal, bool multipleBazFiles>
    struct ImplChild;
    std::unique_ptr<Impl> impl_;
};


}}

#endif //PACBIO_APPLICATION_PRELIM_HQ_FILTER_H
