// Copyright (c) 2022, Pacific Biosciences of California, Inc.
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

#ifndef PACBIO_COMMON_UTILITY_STATUSMESSAGE_H
#define PACBIO_COMMON_UTILITY_STATUSMESSAGE_H

#include <numeric>

#include <pacbio/configuration/PBConfig.h>
#include <pacbio/utilities/ISO8601.h>

#include <boost/iostreams/device/file_descriptor.hpp>
#include <boost/iostreams/stream.hpp>

namespace PacBio::Utility
{

template <typename Stages>
class ProgressMessage
{
    static_assert(std::is_base_of<SmartEnumBase, Stages>::value, "Stages must be a SMART_ENUM!");
public:
    struct StageInfo
    {
        bool ready;
        int stageNumber;
        int stageWeight;
    };
    using Table = std::map<std::string, StageInfo>;

public:
    ProgressMessage(const Table& stages, const std::string& header, int statusFd)
        : stages_(stages)
        , header_(header)
        , stream_(boost::iostreams::stream<boost::iostreams::file_descriptor_sink>(statusFd,
                                                                                   boost::iostreams::never_close_handle))
    {
        const auto& asv = Stages::allValuesAsStrings();
        if (stages_.size() != asv.size())
        {
            throw PBException("Number of stages doesn't match number of values in stage enum!");
        }

        for (const auto& s : Stages::allValuesAsStrings())
        {
            const auto& st = stages_.find(s);
            if (st == stages_.end())
            {
                throw PBException("Stage name not found in stage enum!");
            }
        }
        std::vector<int> stageNumbers;
        std::vector<int> sw;
        for (const auto& kv : stages_)
        {
            stageNumbers.push_back(kv.second.stageNumber);
            sw.push_back(kv.second.stageWeight);
        }
        std::vector<int> sn = stageNumbers;
        std::sort(sn.begin(), sn.end());
        auto last = std::unique(sn.begin(), sn.end());
        if (std::distance(sn.begin(), last) != static_cast<int>(stageNumbers.size()))
        {
            throw PBException("Stage numbers not unique!");
        }
        std::vector<size_t> idx(sw.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return stageNumbers[a] < stageNumbers[b]; });
        for (const auto& i : idx) stageWeights.push_back(sw[i]);
    }

public:
    ProgressMessage() = delete;
    ProgressMessage(ProgressMessage&&) = default;
    ProgressMessage(const ProgressMessage&) = default;
    ProgressMessage& operator=(ProgressMessage&&) = default;
    ProgressMessage& operator=(ProgressMessage&) = delete;
    ~ProgressMessage() = default;

public:
    void Message(const Stages& s, uint64_t counter, uint64_t counterMax, double timeoutForNextStatus)
    {
        Output o;
        const auto& it = stages_.find(Stages::toString(s));
        o.ready = it->second.ready;
        o.stageNumber = it->second.stageNumber;
        o.stageName = it->first;
        o.stageWeights = stageWeights;
        o.counter = counter;
        o.counterMax = counterMax;
        o.timeoutForNextStatus = timeoutForNextStatus;
        o.timeStamp = Utilities::ISO8601::TimeString();
        stream_ << header_ << " " << o.Serialize() << std::endl;
    }

    void Exception(const std::string& what)
    {
        ExceptionOutput o;
        o.message = what;
        o.timeStamp = Utilities::ISO8601::TimeString();
        stream_ << header_ << " " << o.Serialize() << std::endl;
    }

private:
    struct Output : Configuration::PBConfig<Output>
    {
        PB_CONFIG(Output);
        PB_CONFIG_PARAM(bool, ready, false);
        PB_CONFIG_PARAM(int, stageNumber, 0);
        PB_CONFIG_PARAM(std::string, stageName, "");
        PB_CONFIG_PARAM(std::vector<int>, stageWeights, std::vector<int>{});
        PB_CONFIG_PARAM(uint64_t, counter, 0);
        PB_CONFIG_PARAM(uint64_t, counterMax, 1);
        PB_CONFIG_PARAM(double, timeoutForNextStatus, 0);
        PB_CONFIG_PARAM(std::string, timeStamp, "");
        PB_CONFIG_PARAM(std::string, state, "progress");
    };

    struct ExceptionOutput : Configuration::PBConfig<ExceptionOutput>
    {
        PB_CONFIG(ExceptionOutput);
        PB_CONFIG_PARAM(std::string, message, "");
        PB_CONFIG_PARAM(std::string, timeStamp, "");
        PB_CONFIG_PARAM(std::string, state, "exception");
    };
private:
    Table stages_;
    std::vector<int> stageWeights;
    std::string header_;
    boost::iostreams::stream<boost::iostreams::file_descriptor_sink> stream_;
};

} // namespace PacBio::Utility

#endif // PACBIO_COMMON_UTILITY_STATUSMESSAGE_H

