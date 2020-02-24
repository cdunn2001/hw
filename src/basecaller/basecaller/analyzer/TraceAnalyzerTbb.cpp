
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
#include <tbb/flow_graph.h>

#include <pacbio/PBException.h>

#include <dataTypes/BasecallerConfig.h>
#include <dataTypes/MovieConfig.h>

#include <basecaller/traceAnalysis/DetectionModelEstimator.h>
#include <basecaller/traceAnalysis/TraceHistogramAccumulator.h>

using std::vector;

namespace PacBio {
namespace Mongo {
namespace Basecaller {

TraceAnalyzerTbb::TraceAnalyzerTbb(const vector<uint32_t>& poolIds,
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

    const bool staticAnalysis = bcConfig.algorithm.staticAnalysis;
    for (unsigned int poolId : poolIds)
    {
        const auto it = bAnalyzer_.find(poolId);
        if (it != bAnalyzer_.cend()) continue;  // Ignore duplicate ids.

        auto batchAnalyzer = BatchAnalyzer(poolId, algoFactory_);
        if (staticAnalysis)
        {
            batchAnalyzer.SetupStaticModel(bcConfig.algorithm.staticDetModelConfig, movConfig);
        }

        bAnalyzer_.emplace(poolId, std::move(batchAnalyzer));
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


vector<std::unique_ptr<BatchAnalyzer::OutputType>>
TraceAnalyzerTbb::Analyze(vector<Data::TraceBatch<int16_t>> input)
{
    const size_t n = input.size();
    assert(input.size() <= bAnalyzer_.size());

    vector<std::unique_ptr<BatchAnalyzer::OutputType>> output(n);

    tbb::task_scheduler_init init(NumWorkerThreads());

    // This is what is desirable for cuda.  3 Threads is minimum,
    // so that one thread can upload while another does compute while
    // the last does download.  One extra thread seems to be beneficial
    // to smooth out scheduling issues.  Anything beyond that had
    // no significant impact on performance, so we'll save any other
    // available host threads for nested parallelism in host filter stages
    static constexpr size_t numTopLevelThreads=4;
    tbb::flow::graph g;
    tbb::flow::function_node<size_t> filter(g, numTopLevelThreads, [&](size_t i){
            // An analyzer for this pool should already exist in bAnalyzer_.
            // Use std::map::at instead of operator[] to ensure that we don't
            // inadvertently insert an element in a multithreaded context.
            auto& analyzer = bAnalyzer_.at(input[i].GetMeta().PoolId());
            output[i] = std::make_unique<BatchAnalyzer::OutputType>(
                analyzer(input[i]));
    });

    // All this graph buisiness is to try and limit concurrency at this top level loop,
    // while not limiting concurrency on any nested loops in individual filter stages.
    // Maybe I missed an easy way to do this, so here are the rejected approaches and why:
    //
    // - Limiting the grain size so that each item of work the scheduler sees is 1/4 of the data:
    //     While this will probably work, it potentially has issues with load balancing.  For
    //     a homogenous set of zmws then each "grain" will probably process in the same time, but
    //     if there happen to be differences then we won't be able to even things out at all.
    // - Limiting parallelism via `task_arena`:
    //     Unless I'm misreading the documentation, this will constrain *both* the outter *and*
    //     inner loops.  I can't tell the outter to use 4 threads and let the inner use 40.
    // - Using parallel_pipeline:
    //     This one would work, but has more infrastructure for setting up.  The overall pipeline
    //     needs to accept and return void, meaning one would have to write source and sink filters
    //     to manage iterating over the data.
    //
    //  With openmp this would be a very simple task, though openmp does come with a couple minor
    //  quirks that prevent me from lobying for it just yet.  But right here it would be a simple
    //  pragma as follows...
    //
    //  #pragma omp parallel_for num_threads(4)
    for (size_t i = 0; i < n; ++i)
    {
        filter.try_put(i);
    }
    g.wait_for_all();

    return output;
}

TraceAnalyzerTbb::~TraceAnalyzerTbb()
{
    if (NumWorkerThreads() != 1)
    {
        Logging::LogStream msg(PacBio::Logging::LogLevel::WARN);
        msg << "\nNote: The following detailed report really only makes sense if \n"
            "there is only a single worker thread.  Otherwise the time reported in a \n"
            "given filter stage also includes the time waiting for other threads \n"
            "that are already utilizing the gpu\n";
    }
    BatchAnalyzer::ReportPerformance();
    PBLOG_INFO << "Peak GPU memory usage: " << Cuda::Memory::SmartDeviceAllocation::PeakAllocatedBytes() / static_cast<float>(1<<20) << " MB";
    PBLOG_INFO << "Peak (Managed) Host memory usage: " << PacBio::Memory::SmartAllocation::PeakAllocatedBytes() / static_cast<float>(1<<20) << " MB";

    BatchAnalyzer::Finalize();
}

}}}     // namespace PacBio::Mongo::Basecaller
