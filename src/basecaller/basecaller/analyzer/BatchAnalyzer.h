
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
//  Defines class BatchAnalyzer.

#include <dataTypes/BasecallBatch.h>
#include <dataTypes/TraceBatch.h>
#include <dataTypes/ConfigForward.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

/// A function-like type that performs trace analysis for a particular batch
/// of ZMWs.
class BatchAnalyzer
{
public:     // Types
    using InputType = PacBio::Mongo::Data::TraceBatch<int16_t>;
    using OutputType = PacBio::Mongo::Data::BasecallBatch;

public:     // Structors & assignment operators
    BatchAnalyzer(uint32_t batchId,
                  const Data::BasecallerAlgorithmConfig& bcConfig,
                  const Data::MovieConfig& movConfig);

    BatchAnalyzer(const BatchAnalyzer&) = delete;
    BatchAnalyzer(BatchAnalyzer&&) = default;

    BatchAnalyzer& operator=(const BatchAnalyzer&) = delete;
    BatchAnalyzer& operator=(BatchAnalyzer&&) = default;

    ~BatchAnalyzer() = default;

public:
    /// Call operator is non-reentrant and will throw if a trace batch is
    /// received for the wrong ZMW batch or is out of chronological order.
    PacBio::Mongo::Data::BasecallBatch
    operator()(PacBio::Mongo::Data::TraceBatch<int16_t> tbatch);

private:
    uint32_t batchId_;
    uint32_t nextFrameId_ = 0;
};

}}}     // namespace PacBio::Mongo::Basecaller
