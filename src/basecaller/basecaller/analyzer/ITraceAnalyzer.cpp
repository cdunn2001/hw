
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
//  Defines some members of class ITraceAnalyzer.

#include "ITraceAnalyzer.h"

#include <algorithm>
#include <map>
#include <numeric>
#include <type_traits>

#include <dataTypes/MovieConfig.h>

#include "TraceAnalyzerTbb.h"


namespace {

// Returns a vector containing a range of integers.
template <typename IntType>
std::vector<IntType> rangeVector(size_t size, IntType start = 0)
{
    static_assert(std::is_integral<IntType>::value,
                  "Template argument must be integral type.");
    std::vector<IntType> v(size);
    std::iota(v.begin(), v.end(), start);
    return v;   // NRVO
}

}   // anonymous namepsace


namespace PacBio {
namespace Mongo {
namespace Basecaller {

// static
bool ITraceAnalyzer::Initialize(const PacBio::Mongo::Data::BasecallerInitConfig&)
{
    return false;
}


// static
std::unique_ptr<ITraceAnalyzer>
ITraceAnalyzer::Create(unsigned int numPools,
                       const Data::BasecallerConfig& bcConfig,
                       const Data::MovieConfig& movConfig)
{
    const auto poolIds = rangeVector<uint32_t>(numPools);
    return Create(poolIds, bcConfig, movConfig);
}

// static
std::unique_ptr<ITraceAnalyzer>
ITraceAnalyzer::Create(const std::vector<uint32_t> &poolIds,
                       const Data::BasecallerConfig &bcConfig,
                       const Data::MovieConfig &movConfig)
{
    // At this point, there is only one implementation, which uses TBB.
    std::unique_ptr<ITraceAnalyzer> p {new TraceAnalyzerTbb{poolIds, bcConfig, movConfig}};
    return p;
}


// Returns true if the input meets basic contracts.
bool ITraceAnalyzer::IsValid(const std::vector<Data::TraceBatch<int16_t>>& input)
{
    if (input.size() > NumZmwPools()) return false;
    if (input.empty()) return false;
    const auto first = input.front().GetMeta().FirstFrame();
    const auto last = input.front().GetMeta().LastFrame();
    using PidCountMap = std::map<uint32_t, unsigned short>;
    PidCountMap pidCount;
    for (const auto& tb : input)
    {
        if (tb.GetMeta().FirstFrame() != first) return false;
        if (tb.GetMeta().LastFrame() != last) return false;
        ++pidCount[tb.GetMeta().PoolId()];
    }
    if (std::any_of(pidCount.begin(), pidCount.end(),
                    [](const PidCountMap::value_type& kvp){return kvp.second > 1;}))
    {
        return false;
    }

    return true;
}

}}}     // namespace PacBio::Mongo::Basecaller
