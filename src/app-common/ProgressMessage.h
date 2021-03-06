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

#ifndef APP_COMMON_STATUSMESSAGE_H
#define APP_COMMON_STATUSMESSAGE_H

#include <numeric>
#include <mutex>
#include <sstream>

#include <pacbio/configuration/PBConfig.h>
#include <pacbio/dev/AutoTimer.h>
#include <pacbio/utilities/ISO8601.h>

#include <boost/iostreams/device/file_descriptor.hpp>
#include <boost/iostreams/stream.hpp>

namespace PacBio {
namespace IPC {

template <typename Stages>
class ProgressMessage
{
    static_assert(std::is_base_of<SmartEnumBase, Stages>::value, "Stages must be a SMART_ENUM!");

    Json::StreamWriterBuilder GetStreamWriterBuilder()
    {
        Json::StreamWriterBuilder builder;
        builder.settings_["commentStyle"] = "None";
        builder.settings_["indentation"] = "";
        return builder;
    }
public:
    struct StageInfo
    {
        bool ready;
        int stageNumber;
        int stageWeight;
    };
    using Table = std::map<std::string, StageInfo>;

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
        PB_CONFIG_PARAM(Json::Value, metrics, Json::objectValue);
    };

public:
    ProgressMessage(const Table& stages, const std::string& header, int statusFd)
        : stageInfo_(stages)
        , header_(header)
        , stream_(boost::iostreams::stream<boost::iostreams::file_descriptor_sink>(statusFd, boost::iostreams::close_handle))
        , jsonWriter_(GetStreamWriterBuilder().newStreamWriter())
    {
        {
            // Validate the SMART_ENUM against the stage names.
            const auto& asv = Stages::allValuesAsStrings();
            if (stageInfo_.size() != asv.size())
            {
                throw PBException("Number of stages doesn't match number of values in stage enum!");
            }

            for (const auto& s : asv)
            {
                const auto& st = stageInfo_.find(s);
                if (st == stageInfo_.end())
                {
                    throw PBException("Stage name not found in stage enum!");
                }
            }
        }

        {
            // Stages are stored in the table as std::map
            // so we have to pull out the stage weights and make sure
            // they are ordered based on the stage numbers.
            std::vector<int> stageNumbers;
            std::vector<int> sw;
            for (const auto& kv: stageInfo_)
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
            for (const auto& i: idx) stageWeights_.push_back(sw[i]);
        }
    }

public:
    ProgressMessage() = delete;
    ProgressMessage(ProgressMessage&&) = default;
    ProgressMessage(const ProgressMessage&) = default;
    ProgressMessage& operator=(ProgressMessage&&) = default;
    ProgressMessage& operator=(ProgressMessage&) = delete;
    ~ProgressMessage() = default;

public:
    /// Prepares the `stage` object with the defaults from the defaults table
    /// This does not send a message; SendMessage(stage) if you want to immediate send the message.
    void Initialize(const Stages& s, Output& stage)
    {
        const auto& it = stageInfo_.find(Stages::toString(s));
        // No need to check it, we've already validated things in the constructor.
        stage.stageName = it->first;
        stage.stageNumber = it->second.stageNumber;
        stage.ready = it->second.ready;
        stage.stageWeights = stageWeights_;
    }

    void SendMessage(Output& stage)
    {
        std::ostringstream sout;
        stage.timeStamp = Utilities::ISO8601::TimeString();
        sout << header_ << " ";
        jsonWriter_->write(stage.Serialize(), &sout);
        sout << std::endl;
        stream_ << sout.str();
        stream_.flush();
    }

    void Exception(const Json::Value& j)
    {
        std::ostringstream sout;
        Json::Value jOut = j;
        jOut["timeStamp"] = Utilities::ISO8601::TimeString();
        jOut["state"] = "exception";
        sout << header_ << " ";
        jsonWriter_->write(jOut, &sout);
        sout << std::endl;
        stream_ << sout.str();
        stream_.flush();
    }
    void Exception(const std::string& message)
    {
        Json::Value j = Json::objectValue;
        j["message"] = message;
        Exception(j);
    }
    void Exception(const char* message)
    {
        Json::Value j = Json::objectValue;
        j["message"] = message;
        Exception(j);
    }

    class StageReporter
    {
    public:
        StageReporter(ProgressMessage* pm, const Stages& stage, uint64_t counterMax, double timeoutForNextStatus,
            std::function<void(Json::Value&)> metricsCallback = nullptr
        )
            : pm_(pm), metricsCallback_(metricsCallback)
        {
            currentStage_.counterMax = counterMax;
            currentStage_.timeoutForNextStatus = timeoutForNextStatus;
            pm_->Initialize(stage, currentStage_);
            Flush(true);
        }

        void Flush(bool force = false)
        {
            if (force || timeSinceOutput_.GetElapsedMilliseconds() >= minimumReportInterval || forceNext_)
            {
                if (metricsCallback_)
                {
                    metricsCallback_(GetMetrics());
                }
                pm_->SendMessage(currentStage_);
                timeSinceOutput_.Restart();
                forceNext_ = false;
            }
        }

        /// \param delta Increments the counter by the delta amount. The counter will not exceed the counterMax
        void Update(uint64_t delta)
        {
            std::lock_guard<std::mutex> lock(reportMutex_);
            currentStage_.counter = std::min(currentStage_.counter + delta, currentStage_.counterMax);
            Flush();
        }

        void Update(uint16_t counter, double timeoutForNextStatus)
        {
            currentStage_.timeoutForNextStatus = timeoutForNextStatus;
            Update(counter);
        }
        /// \returns a reference to the metrics JSON object. The user is free
        /// to set key/value values inside the metrics object, which will be
        /// reported to pa-ws.
        Json::Value& GetMetrics()
        {
            return currentStage_.metrics;
        }
        void SetMinimumInterval(uint32_t x)
        {
            minimumReportInterval = x;
        }
        void ForceNextUpdate()
        {
            forceNext_ = true;
        }
    private:
        std::mutex reportMutex_;
        PacBio::Dev::QuietAutoTimer timeSinceOutput_;
        ProgressMessage* pm_;
        Output currentStage_;
        std::function<void(Json::Value& metrics)> metricsCallback_;
        uint32_t minimumReportInterval = 1000;
        bool forceNext_ = false;
    };

public:
    class ThreadSafeStageReporter
    {
    public:
        /// \param pm The instance that actually sends the message
        /// \param s The stage that this report is bound to
        /// \param counterMax the maximum number associated with the counter, to be used to calculate a fractional progress
        /// \param timeoutForNextStatus the maximum time expected for the next progress message.
        /// \param metricsCallback An optional callback that can be used to refresh some metrics that will be bundled in the
        ///        progress message, under the "metrics" JSON object. The argument to the callback is the Json::Value& of the
        ///        "metrics" object.   NB: the callback must be threadsafe.
        ThreadSafeStageReporter(ProgressMessage* pm, const Stages& s, uint64_t counterMax, double timeoutForNextStatus,
            std::function<void(Json::Value& metrics)> metricsCallback = nullptr
        )
        : sr_(pm, s, counterMax, timeoutForNextStatus, metricsCallback)
        { 
           
        }
        void SetMinimumInterval(uint32_t x)
        {
            sr_.SetMinimumInterval(x);
        }
        void Flush()
        {
            std::lock_guard<std::mutex> lock(mutex_);
            sr_.Flush();
        }

        void Update(uint64_t counter)
        {
            std::lock_guard<std::mutex> lock(mutex_);
            sr_.Update(counter);
        }

        void Update(uint16_t counter, double timeoutForNextStatus)
        {
            std::lock_guard<std::mutex> lock(mutex_);
            sr_.Update(counter, timeoutForNextStatus);
        }
    private:
        std::mutex mutex_;
        StageReporter sr_;
    };

private:
    Table stageInfo_;
    std::vector<int> stageWeights_;
    std::string header_;
    boost::iostreams::stream<boost::iostreams::file_descriptor_sink> stream_;
    std::unique_ptr<Json::StreamWriter> jsonWriter_;
};

}} // namespace PacBio::IPC

#endif // APP_COMMON_STATUSMESSAGE_H

