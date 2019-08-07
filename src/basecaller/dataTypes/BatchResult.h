#ifndef mongo_dataTypes_BatchResult_H_
#define mongo_dataTypes_BatchResult_H_

// Copyright (c) 2019, Pacific Biosciences of California, Inc.
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
//
//  Description:
//  Defines class BatchResult, a light wrapper around a finished PulseBatch and
//  an accumulated BasecallingMetrics object

#include <dataTypes/PulseBatch.h>
#include <dataTypes/BasecallingMetrics.h>

namespace PacBio {
namespace Mongo {
namespace Data {

struct BatchResult
{

    using PulseBatchT = PulseBatch;
    using MetricsT = Cuda::Memory::UnifiedCudaArray<BasecallingMetrics<laneSize>>;

    BatchResult(PulseBatchT&& pulsesIn)
        : pulses(std::move(pulsesIn))
    {};

    BatchResult(PulseBatchT&& pulsesIn, std::unique_ptr<MetricsT> metricsPtr)
        : pulses(std::move(pulsesIn))
        , metrics(std::move(metricsPtr))
    {};

    PulseBatchT pulses;
    std::unique_ptr<MetricsT> metrics;
};


}}}    // namespace PacBio::Mongo::Data

#endif // mongo_dataTypes_BatchResult_H_
