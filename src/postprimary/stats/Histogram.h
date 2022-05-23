// Copyright (c) 2015-2016, Pacific Biosciences of California, Inc.
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted (subject to the limitations in the
// disclaimer below) provided that the following conditions are met:
//
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
//  * Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
//  * Neither the name of Pacific Biosciences nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE
// GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY PACIFIC
// BIOSCIENCES AND ITS CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL PACIFIC BIOSCIENCES OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
// USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
// OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGE.

// Programmer: Armin Töpfer

#pragma once

#include <algorithm>
#include <numeric>
#include <ostream>
#include <utility>
#include <vector>

#include <json/json.h>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/min.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/median.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include <pacbio/logging/Logger.h>

namespace PacBio {
namespace Primary {
namespace Postprimary {

struct Histogram
{
public: // constexpr
    static const int defaultNumBins = 30;

public: // ctor
    Histogram(const std::string& md, const float bw, const float numBins);
    Histogram(std::vector<float>& input);
    Histogram(const std::vector<int>& input);
    Histogram(const std::vector<float>& input,
              float binW, float minBinValue,
              int numBins=defaultNumBins,
              float maxBinValue=std::numeric_limits<float>::quiet_NaN());
    Histogram(const std::vector<float>& values,
              const std::string& md);
    Histogram(const std::vector<float>& values,
              const std::string& md,
              const float binW,
              const float maxBinValue);
    Histogram(const std::vector<float>& values,
              const std::string& md,
              const float binW,
              const float minBinValue,
              const int numBins);

public: // non-modifying
    friend std::ostream& operator << (std::ostream& stream, const Histogram& h);
    Json::Value ToJson() const;

public: // data
    size_t sampleSize = 0;

    float binWidth = 0;
    float first    = 0;
    float last     = 0;
    float min      = std::numeric_limits<float>::quiet_NaN();
    float max      = std::numeric_limits<float>::quiet_NaN();
    float mean     = 0;
    float median   = 0;
    float perc95th = 0;
    float stddev   = 0;
    float mode     = 0;
    float n50      = 0;

    bool outputN50 = false;
    std::vector<int> bins;
    std::string metricDescription;

private: // modifying
    void Bin(const std::vector<float>& input);

private: // typedef
    using Accu = boost::accumulators::accumulator_set<
        float, 
        boost::accumulators::stats<
            boost::accumulators::tag::mean, 
            boost::accumulators::tag::variance, 
            boost::accumulators::tag::median, 
            boost::accumulators::tag::min, 
            boost::accumulators::tag::max
    >>;
};

struct DiscreteHistogram
{
public:
    DiscreteHistogram(const std::vector<float>& values,
                      const std::vector<std::string>& bls,
                      const std::string& md);

public: // non-modifying
    friend std::ostream& operator << (std::ostream& stream, const DiscreteHistogram& h);

public:
    std::vector<int> bins;
    std::vector<std::string> binLabels;
    std::string metricDescription;
private:
    void Bin(const std::vector<float>& input);
};

}}}
