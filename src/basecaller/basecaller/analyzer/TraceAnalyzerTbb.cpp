
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

#include <basecaller/traceAnalysis/DetectionModelEstimator.h>
#include <basecaller/traceAnalysis/TraceHistogramAccumulator.h>

using std::vector;

namespace PacBio {
namespace Mongo {
namespace Basecaller {

TraceAnalyzerTbb::TraceAnalyzerTbb(unsigned int numPools,
                                   const Data::BasecallerConfig& bcConfig,
                                   const Data::MovieConfig& movConfig)
    : algoFactory_ (bcConfig.algorithm)
    , init_(bcConfig.init.numWorkerThreads)
{
    // TODO: Check logic of worker threads
    NumWorkerThreads(bcConfig.init.numWorkerThreads);

    algoFactory_.Configure(bcConfig.algorithm, movConfig);

    // TODO: If algoFactory_::Configure is handling configuration of the
    // various algorithms, is there a reason to still have a
    // BatchAnalyzer::Configure?
    BatchAnalyzer::Configure(bcConfig.algorithm, movConfig);

    bAnalyzer_.reserve(numPools);
    // TODO: Should be able to parallelize construction of batch analyzers.
    const bool staticAnalysis = bcConfig.algorithm.staticAnalysis;
    for (unsigned int poolId = 0; poolId < numPools; ++poolId)
    {
        bAnalyzer_.emplace_back(poolId, algoFactory_, staticAnalysis);
    }
}

unsigned int TraceAnalyzerTbb::NumWorkerThreads() const
{
    if (numWorkerThreads == 0)
        return tbb::task_scheduler_init::default_num_threads();
    else
        return numWorkerThreads;
}

void TraceAnalyzerTbb::NumWorkerThreads(unsigned int n)
{
    numWorkerThreads = (n == 0
        ? tbb::task_scheduler_init::default_num_threads()
        : n);


}

// The number of ZMW pools supported by this analyzer.
unsigned int TraceAnalyzerTbb::NumZmwPools() const
{
    return boost::numeric_cast<unsigned int>(bAnalyzer_.size());
}


vector<std::unique_ptr<Data::BasecallBatch>>
TraceAnalyzerTbb::Analyze(vector<Data::TraceBatch<int16_t>> input)
{
    const size_t n = input.size();
    assert(input.size() <= bAnalyzer_.size());

    vector<std::unique_ptr<Data::BasecallBatch>> output;
    output.reserve(n);

    tbb::task_scheduler_init init(NumWorkerThreads());
    // TODO: Customize optional parameters of parallel_for.
    tbb::parallel_for(size_t(0), n, [&](size_t i)
    {
        const auto pid = input[i].GetMeta().PoolId();
        output[i] = std::make_unique<Data::BasecallBatch>(bAnalyzer_[pid](std::move(input[i])));
    });

    return output;
}

}}}     // namespace PacBio::Mongo::Basecaller
