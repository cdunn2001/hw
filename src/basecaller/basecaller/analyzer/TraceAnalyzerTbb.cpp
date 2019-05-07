
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
//  Defines members of class TraceAnalyzerTbb.

#include "TraceAnalyzerTbb.h"

#include <boost/numeric/conversion/cast.hpp>
#include <tbb/parallel_for.h>

#include <pacbio/PBException.h>

#include <dataTypes/BasecallerConfig.h>
#include <dataTypes/MovieConfig.h>

using std::vector;

namespace PacBio {
namespace Mongo {
namespace Basecaller {

TraceAnalyzerTbb::TraceAnalyzerTbb(unsigned int numPools,
                                   const Data::BasecallerAlgorithmConfig& bcConfig,
                                   const Data::MovieConfig movConfig)
{
    bAnalyzer_.reserve(numPools);
    // TODO: Should be able to parallelize construction of batch analyzers.
    for (unsigned int poolId = 0; poolId < numPools; ++poolId)
    {
        bAnalyzer_.emplace_back(poolId, bcConfig, movConfig);
    }
}

// The number of worker threads used by this analyzer.
unsigned int TraceAnalyzerTbb::NumWorkerThreads() const
{
    // TODO
    return 0;
}

// Sets the number of worker threads requested.
// To choose the default value for the platform, specify 0.
void TraceAnalyzerTbb::NumWorkerThreads(unsigned int)
{
    // TODO
    throw PBException("Not yet implemented.");
}

// The number of ZMW pools supported by this analyzer.
unsigned int TraceAnalyzerTbb::NumZmwPools() const
{
    return boost::numeric_cast<unsigned int>(bAnalyzer_.size());
}


vector<Data::BasecallBatch>
TraceAnalyzerTbb::Analyze(vector<Data::TraceBatch<int16_t>> input)
{
    const size_t n = input.size();
    assert(input.size() < bAnalyzer_.size());

    vector<Data::BasecallBatch> output (n);

    // TODO: Customize optional parameters of parallel_for.
    tbb::parallel_for(size_t(0), n, [&](size_t i)
    {
        const auto pid = input[i].GetMeta().PoolId();
        output[i] = bAnalyzer_[pid](std::move(input[i]));
    });

    return output;
}

}}}     // namespace PacBio::Mongo::Basecaller
