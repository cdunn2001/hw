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
    Histogram(const std::string& md, const float bw, const float numBins)
    : binWidth(bw), metricDescription(md)
    {
        // "Empty" distribution
        sampleSize = 0;

        const float defaultEmptyVal = 0;

        // Set summary metrics to NaN.
        mean     = defaultEmptyVal;
        median   = defaultEmptyVal;
        perc95th = defaultEmptyVal;
        stddev   = defaultEmptyVal;
        mode     = defaultEmptyVal;

        // No outliers or min/max.
        first    = defaultEmptyVal;
        last     = defaultEmptyVal;
        min      = defaultEmptyVal;
        max      = defaultEmptyVal;

        // Zero out bins.
        bins.resize(numBins);
        for (size_t i = 0; i < bins.size(); i++)
            bins[i] = 0;
    }

    Histogram(std::vector<float>& input)
    {
        bins.resize(defaultNumBins);
        Bin(input);
    }

    Histogram(const std::vector<int>& input)
    {
        std::vector<float> floats;
        floats.reserve(input.size());
        for (const auto& i : input)
            floats.push_back(i);
        bins.resize(defaultNumBins);
        Bin(floats);
    }

    Histogram(const std::vector<float>& input,
              float binW, float minBinValue,
              int numBins=defaultNumBins,
              float maxBinValue=std::numeric_limits<float>::quiet_NaN())
        : binWidth(binW), min(minBinValue), max(maxBinValue)
    {
        bins.resize(numBins);
        Bin(input);
    }

    Histogram(const std::vector<float>& values,
              const std::string& md)
        : metricDescription(md)
    {
        bins.resize(defaultNumBins);
        Bin(values);
    }

    Histogram(const std::vector<float>& values,
              const std::string& md,
              const float binW,
              const float maxBinValue)
        : binWidth(binW),
          max(maxBinValue),
          metricDescription(md)
    {
        Bin(values);
    }

    Histogram(const std::vector<float>& values,
              const std::string& md,
              const float binW,
              const float minBinValue,
              const int numBins)
        : binWidth(binW),
          min(minBinValue),
          metricDescription(md)
    {
        bins.resize(numBins);
        Bin(values);
    }


public: // non-modifying
    friend std::ostream& operator << (std::ostream& stream, const Histogram& h)
    {
        auto NaNString = [&stream](const std::string& tag, float v)
        {
            // FIXME: JSON serialization doesn't support NaN, so we use zero.
            stream << "<" << tag << ">";
            if (std::isnan(v))
                stream << 0;
            else
            {
                try
                {
                    float f = boost::numeric_cast<float>(v);
                    stream << f;
                }
                catch (boost::numeric::bad_numeric_cast& e)
                {
                    PBLOG_WARN << "Unable to convert " << tag << " float=" << v << " to float due to: " << e.what();
                    stream << 0;
                }
            }
            stream << "</" << tag << ">";
        };

        stream << "<ns:SampleSize>" << h.sampleSize << "</ns:SampleSize>";

        NaNString("ns:SampleMean", h.mean);
        NaNString("ns:SampleMed", h.median);
        NaNString("ns:SampleMode", h.mode);
        NaNString("ns:SampleStd", h.stddev);
        NaNString("ns:Sample95thPct", h.perc95th);
        if (h.outputN50)
        {
            NaNString("ns:SampleN50", h.n50);
        }

        stream << "<ns:NumBins>" << h.bins.size() << "</ns:NumBins>";

        stream << "<ns:BinCounts>";
        for (const auto& d : h.bins)
            stream << "<ns:BinCount>" << d << "</ns:BinCount>";
        stream << "</ns:BinCounts>";

        NaNString("ns:BinWidth", h.binWidth);
        NaNString("ns:MinOutlierValue", h.first);
        NaNString("ns:MinBinValue", h.min);
        NaNString("ns:MaxBinValue", h.max);
        NaNString("ns:MaxOutlierValue", h.last);

        stream << "<ns:MetricDescription>" << h.metricDescription << "</ns:MetricDescription>";

        return stream;
    }

    Json::Value ToJson() const
    {
        Json::Value root;
        root["SAMPLE_SIZE"] = (Json::UInt64)sampleSize;
        root["SAMPLE_MEAN"] = mean;
        root["SAMPLE_MED"] = median;
        root["SAMPLE_MODE"] = mode;
        root["SAMPLE_STD"] = stddev;
        root["SAMPLE_95TH_PCT"] = perc95th;
        root["SAMPLE_N50"] = n50;
        root["NUM_BINS"] = (Json::UInt64)bins.size();
        auto& binCounts = root["BIN_COUNTS"];
        for (const auto& d : bins)
            binCounts.append(d);
        root["BIN_WIDTH"] = binWidth;
        root["MIN_OUTLIER_VALUE"] = first;
        root["MIN_BIN_VALUE"] = min;
        root["MAX_BIN_VALUE"] = max;
        root["MAX_OUTLIER_VALUE"] = last;
        return root;
    }

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
    void Bin(const std::vector<float>& input)
    {
        // Remove NaNs.
        std::vector<float> inputNoNans;
        std::copy_if(input.begin(), input.end(), std::back_inserter(inputNoNans),
                     [](float v) { return !std::isnan(v); });

        sampleSize = inputNoNans.size();

        // stop if there is no data available
        if (sampleSize == 0) return;

        // accumulate
        Accu acc;
        std::for_each(inputNoNans.begin(), inputNoNans.end(), [&acc](const float d){acc(d);});
        std::vector<float> inputCopy;
        std::for_each(inputNoNans.begin(), inputNoNans.end(), [&inputCopy](const float d){inputCopy.push_back(d);});
        std::sort(inputCopy.begin(), inputCopy.end());

        // Summary stats are on the the full data.

        // range sample
        first    = boost::accumulators::min(acc);
        last     = boost::accumulators::max(acc);

        // median, take the center element
        mean     = boost::accumulators::mean(acc);
        median   = inputCopy[inputCopy.size()/2];
        stddev   = std::sqrt(boost::accumulators::variance(acc));

        // Compute 95th percentile directly from data.
        unsigned int pos = std::floor(std::max<float>(0, std::min<float>(inputCopy.size()-1, (95/100.0) * inputCopy.size())));
        perc95th = inputCopy[pos];

        double halfLength = std::accumulate(inputCopy.begin(), inputCopy.end(), 0.0) / 2.0;
        double total = 0;
        for (const auto& l : inputCopy)
        {
            if (total < halfLength)
            {
                n50 = l;
                total += l;
            }
            else
                break;
        }

        if (bins.empty())
        {
            // Number of bins not specified, determine from data.
            min = 0;
            if (last < max)
            {
                max = last;

                // We should have no outliers.
                first = std::numeric_limits<float>::quiet_NaN();
                last = std::numeric_limits<float>::quiet_NaN();
            }

            max = std::ceil(max/binWidth) * binWidth;
            int numBins = (max - min) / binWidth;
            bins.resize(numBins);
        }
        else
        {
            if (!std::isnan(min))
            {
                // If a minimum bin value is specified, use that
                // as well as the set bin width. Both must be present.
                if (std::isnan(max))
                {
                    // Max is computed with respect to min value specified, unless it is specified.
                    max = min + (binWidth * bins.size());
                }

                // If the sample range is within the min/max, then indicate we have no outliers.
                if (min <= first) first = std::numeric_limits<float>::quiet_NaN();
                if (max > last) last = std::numeric_limits<float>::quiet_NaN();
            }
            else
            {
                // range 5 std dev
                min = median - (5 * stddev);
                if (min < first) min = first;
                max = median + (5 * stddev);
                if (max > last) max = last;

                if (inputNoNans.size() <= 2)
                {
                    mode = inputNoNans.front();
                    return;
                }
            }
        }

        assert(bins.size() > 0);
        binWidth = (max - min) / bins.size();

        if (binWidth > 0)
        {
            // Bin
            int numOutliers = 0;
            std::for_each(inputNoNans.begin(), inputNoNans.end(),
                          [&](const float d)
                          {
                              if (d >=min && d < max)
                              {
                                  unsigned int binHit = std::floor((d - min) / binWidth);
                                  // possible floating point round-up might cause this next corner case,
                                  // it has been seen in SE-338.
                                  if (binHit == bins.size()) binHit--;
                                  if (binHit < bins.size())
                                  {
                                      bins[binHit]++;
                                  }
                              }
                              else
                              {
                                  numOutliers++;
                              }
                          }
            );

            // Check on the outlier fraction.
            float maxOutlierFraction = 0.25;
            float outlierFraction = numOutliers/(float)inputNoNans.size();
            if (outlierFraction >= maxOutlierFraction)
            {
                PBLOG_WARN << "High outlier fraction: " <<  outlierFraction << " detected for metric: " << metricDescription;
            }

            // Compute mode as centroid of max bin
            size_t maxIdx = std::distance(bins.begin(), std::max_element(bins.begin(), bins.end()));
            mode = min + ((maxIdx * binWidth) + (binWidth/2));
        }
        else
        {
            mode = inputNoNans.front();
            binWidth = 0;
        }
    }
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
                      const std::string& md)
        : binLabels(bls),
          metricDescription(md)
    {
        bins.resize(binLabels.size());
        Bin(values);
    }
public: // non-modifying
    friend std::ostream& operator << (std::ostream& stream, const DiscreteHistogram& h)
    {
        stream << "<ns:NumBins>" << h.bins.size() << "</ns:NumBins>";

        stream << "<ns:BinCounts>";
        for (const auto& d : h.bins)
            stream << "<ns:BinCount>" << d << "</ns:BinCount>";
        stream << "</ns:BinCounts>";

        stream << "<ns:MetricDescription>" << h.metricDescription << "</ns:MetricDescription>";

        stream << "<ns:BinLabels>";
        for (const auto& d : h.binLabels)
            stream << "<ns:BinLabel>" << d << "</ns:BinLabel>";
        stream << "</ns:BinLabels>";

        return stream;
    }

public:
    std::vector<int> bins;
    std::vector<std::string> binLabels;
    std::string metricDescription;
private:
    void Bin(const std::vector<float>& input)
    {
        for (const auto& v : input)
        {
            if (0 <= v && v < binLabels.size())
            {
                bins[v]++;
            }
            else if (v >= binLabels.size())
            {
                bins[bins.size()-1]++;
            }
            else
            {
                PBLOG_WARN << metricDescription << " negative bin label specified: " << v;
            }
        }
    }
};

}}}


