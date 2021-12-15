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

#include "bazio/BazEventData.h"
#include <algorithm>
#include <boost/multi_array.hpp>
#include <boost/variant.hpp>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>

#include <json/json.h>
#include <json/reader.h>

#include <pacbio/PBException.h>
#include <pacbio/process/OptionParser.h>

#include <bazio/file/FileHeader.h>

#include <bazio/BazCore.h>
#include <bazio/BazReader.h>
#include <BazVersion.h>

#include <pacbio/logging/Logger.h>
#include <pacbio/primary/HDFMultiArrayIO.h>

using namespace PacBio;
using namespace PacBio::Primary;

template <typename T>
using Array2DPtr = std::unique_ptr<boost::multi_array<T, 2>>;
using AnyArray = boost::variant<Array2DPtr<int>, Array2DPtr<float>, Array2DPtr<uint>>;
using DataFrame = std::map<std::string, AnyArray>;

// Class to gather metrics for HDF5 output
class MetricsAggregator {

private:

    // Contains multidimensional arrays (zmw x metric_block) for each metric,
    // type erased to handle both floating and integral data
    DataFrame hfMetrics_;
    DataFrame mfMetrics_;
    DataFrame lfMetrics_;

    std::vector<int> zmwNumbers_;
    const BazReader::FileHeaders* fh_;
    size_t numZmw_;

public:
    MetricsAggregator(const BazReader::FileHeaders* fh)
        : fh_(fh)
        , numZmw_(fh->MaxNumZMWs())
    {
        for (const auto metric : fh_->MetricFields())
        {
            const auto fieldName = metric.fieldName;
            if (metric.fieldScalingFactor != 1)
                hfMetrics_[fieldName.toString()] = Array2DPtr<float>{};
            else
            {
                if (metric.fieldSigned)
                    hfMetrics_[fieldName.toString()] = Array2DPtr<int>{};
                else
                    hfMetrics_[fieldName.toString()] = Array2DPtr<uint>{};
            }
        }

        for (const auto metric : fh_->MetricFields())
        {
            const auto fieldName = metric.fieldName;
            if (metric.fieldScalingFactor != 1)
                mfMetrics_[fieldName.toString()] = Array2DPtr<float>{};
            else
            {
                if (metric.fieldSigned)
                    mfMetrics_[fieldName.toString()] = Array2DPtr<int>{};
                else
                    mfMetrics_[fieldName.toString()] = Array2DPtr<uint>{};
            }
        }

        for (const auto metric : fh_->MetricFields())
        {
            const auto fieldName = metric.fieldName;
            if (metric.fieldScalingFactor != 1)
                lfMetrics_[fieldName.toString()] = Array2DPtr<float>{};
            else
            {
                if (metric.fieldSigned)
                    lfMetrics_[fieldName.toString()] = Array2DPtr<int>{};
                else
                    lfMetrics_[fieldName.toString()] = Array2DPtr<uint>{};
            }
        }
    }

    template<typename T>
    struct FillVisitor : boost::static_visitor<>
    {
        FillVisitor(size_t zmwIdx, size_t numZmw, const std::vector<T>& data)
              : zmwIdx_(zmwIdx)
              , numZmw_(numZmw)
              , data_(data)
        {}

        void operator()(Array2DPtr<T>& dest) const
        {
            if (!dest)
                dest.reset(new boost::multi_array<T, 2>(boost::extents[numZmw_][data_.size()]));

            assert(dest->shape()[0] == numZmw_);
            assert(dest->shape()[1] == data_.size());
            std::copy(data_.begin(), data_.end(), (*dest)[zmwIdx_].begin());
        }
        template <typename Wrong>
        void operator()(Array2DPtr<Wrong>&) const
        {
            throw PBException("Data type mismatch in MetricsAggregator");
        }

        size_t zmwIdx_;
        size_t numZmw_;
        const std::vector<T>& data_;
    };

    template <typename T>
    void Fill(const std::vector<T>& data, size_t zmwIdx, AnyArray& dest)
    {
        boost::apply_visitor(FillVisitor<T>(zmwIdx, numZmw_, data), dest);
    }

    /// Record all metrics values from this ZMW
    void Fill(const ZmwByteData& data)
    {
        assert(data.ZmwIndex() == zmwNumbers_.size());
        zmwNumbers_.push_back(fh_->ZmwIndexToNumber(data.ZmwIndex()));

        auto FillMetric = [this](const RawMetricData& rawData, const MetricField& metric,
                                 const ZmwByteData& data, DataFrame& df)
        {
            const auto fieldName = metric.fieldName;
            if (metric.fieldScalingFactor == 1)
            {
                if (metric.fieldSigned)
                    Fill(rawData.IntMetric(fieldName), data.ZmwIndex(), df[fieldName.toString()]);
                else
                    Fill(rawData.UIntMetric(fieldName), data.ZmwIndex(), df[fieldName.toString()]);
            }
            else
            {
                Fill(rawData.FloatMetric(fieldName), data.ZmwIndex(), df[fieldName.toString()]);
            }
        };

        {
            const auto& rawData = ParseMetricFields(fh_->MetricFields(), data.hFMByteStream());
            for (const auto& metric : fh_->MetricFields())
            {
                FillMetric(rawData, metric, data, hfMetrics_);
            }
        }

        {
            const auto& rawData = ParseMetricFields(fh_->MetricFields(), data.mFMByteStream());
            for (const auto& metric : fh_->MetricFields())
            {
                FillMetric(rawData, metric, data, mfMetrics_);
            }
        }

        {
            const auto& rawData = ParseMetricFields(fh_->MetricFields(), data.lFMByteStream());
            for (const auto& metric : fh_->MetricFields())
            {
                FillMetric(rawData, metric, data, lfMetrics_);
            }
        }
    }

    struct WriteVisitor : boost::static_visitor<>
    {
        WriteVisitor(HDFMultiArrayIO& io, const std::string& name)
              : io_(io)
              , name_(name)
        {}

        template <typename T>
        void operator()(const Array2DPtr<T>& data) const
        {
            assert(data);
            io_.Write(name_, *data);
        }

        HDFMultiArrayIO& io_;
        std::string name_;
    };
    void DumpToFile(const std::string& filename)
    {
        HDFMultiArrayIO h(filename, HDFMultiArrayIO::WriteOver);

        h.Write("ZmwNumbers", zmwNumbers_);

        h.Write("ExperimentMetadata", std::vector<std::string>{Json::writeString(Json::StreamWriterBuilder{}, fh_->ExperimentMetadata())});

        h.CreateGroup("/HFMetrics");
        for (const auto metric : fh_->MetricFields())
        {
            const auto fieldName = metric.fieldName;
            boost::apply_visitor(WriteVisitor(h, "/HFMetrics/" + fieldName.toString()), hfMetrics_[fieldName.toString()]);
        }

        h.CreateGroup("/MFMetrics");
        for (const auto metric : fh_->MetricFields())
        {
            const auto fieldName = metric.fieldName;
            boost::apply_visitor(WriteVisitor(h, "/MFMetrics/" + fieldName.toString()), mfMetrics_[fieldName.toString()]);
        }

        h.CreateGroup("/LFMetrics");
        for (const auto metric : fh_->MetricFields())
        {
            const auto fieldName = metric.fieldName;
            boost::apply_visitor(WriteVisitor(h, "/LFMetrics/" + fieldName.toString()), lfMetrics_[fieldName.toString()]);
        }
    }
};

uint32_t ConvertString2Int(const std::string& str)
{
    std::stringstream ss(str);
    uint32_t x;
    if (! (ss >> x))
    {
        std::cerr << "Error converting " << str << " to integer" << std::endl;
        abort();
    }
    return x;
}

std::vector<std::string> SplitStringToArray(const std::string& str, char splitter)
{
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string temp;
    while (std::getline(ss, temp, splitter)) // split into new "lines" based on character
    {
        tokens.push_back(temp);
    }
    return tokens;
}

std::vector<uint32_t> ParseData(const std::string& data)
{
    std::vector<std::string> tokens = SplitStringToArray(data, ',');

    std::vector<uint32_t> result;
    for (std::vector<std::string>::const_iterator it = tokens.begin(), end_it = tokens.end(); it != end_it; ++it)
    {
        const std::string& token = *it;
        std::vector<std::string> range = SplitStringToArray(token, '-');
        if (range.size() == 1)
        {
            result.push_back(ConvertString2Int(range[0]));
        }
        else if (range.size() == 2)
        {
            uint32_t start = ConvertString2Int(range[0]);
            uint32_t stop = ConvertString2Int(range[1]);
            for (uint32_t i = start; i <= stop; i++)
            {
                result.push_back(i);
            }
        }
        else
        {
            std::cerr << "Error parsing token " << token << std::endl;
            abort();
        }
    }

    return result;
}

using Jval = Json::Value;

static void FillMetrics(const BazReader::FileHeaders* fh, const ZmwByteData& data, const bool internal, const size_t frameLimit, Json::Value& single)
{
    single["ZMW_ID"]     = data.ZmwIndex();
    single["ZMW_NUMBER"] = fh->ZmwIndexToNumber(data.ZmwIndex());
    single["INTERNAL"]   = internal;

    auto floatToJson = [](Jval& jVal, float v) {
        // The current jsoncpp version we are using doesn't support
        // NaN and Inf so we have to explicitly check for it here
        // where we use string representations.
        switch(std::fpclassify(v)) {
            case FP_INFINITE:
                jVal = (v == std::numeric_limits<float>::infinity()) ? "Inf" : "-Inf";
                break;
            case FP_NAN:
                jVal = "NaN";
                break;
            case FP_NORMAL:
            case FP_SUBNORMAL:
            case FP_ZERO:
                jVal = v;
                break;
            default:
                jVal = "unknown";
        }
    };

    auto FillJson = [floatToJson](Jval& jVal, const MetricField& metric, const RawMetricData& rawMetrics, size_t idx)
    {
        const auto fieldName = metric.fieldName;
        if (metric.fieldScalingFactor == 1)
        {
            if (metric.fieldSigned)
                jVal[fieldName.toString()] = rawMetrics.IntMetric(fieldName)[idx];
            else
                jVal[fieldName.toString()] = rawMetrics.UIntMetric(fieldName)[idx];
        }
        else
        {
            floatToJson(jVal[fieldName.toString()], rawMetrics.FloatMetric(fieldName)[idx]);
        }
    };

    auto rawMetrics = ParseMetricFields(fh->MetricFields(), data.hFMByteStream());
    size_t limit = std::min(data.Sizes().numHFMBs, frameLimit / fh->MetricFrames() + 1);
    if (fh->MetricFields().size() == 0) limit = 0;
    for (size_t i = 0; i < limit; ++i)
    {
        Jval x;
        x["HF_ID"] = (int)i;
        for (const auto metric : fh->MetricFields())
        {
            FillJson(x, metric, rawMetrics, i);
        }
        single["HF_METRICS"].append(x);
    }

    rawMetrics = ParseMetricFields(fh->MetricFields(), data.mFMByteStream());
    limit = std::min(data.Sizes().numMFMBs, frameLimit / fh->MetricFrames() + 1);
    if (fh->MetricFields().size() == 0) limit = 0;
    for (size_t i = 0; i < limit; ++i)
    {
        Jval x;
        x["MF_ID"] = (int)i;
        for (const auto metric : fh->MetricFields())
        {
            FillJson(x, metric, rawMetrics, i);
        }
        single["MF_METRICS"].append(x);
    }

    rawMetrics = ParseMetricFields(fh->MetricFields(), data.lFMByteStream());
    limit = std::min(data.Sizes().numLFMBs, frameLimit / fh->MetricFrames() + 1);
    if (fh->MetricFields().size() == 0) limit = 0;
    for (size_t i = 0; i < limit; ++i)
    {
        Jval x;
        x["LF_ID"] = (int)i;
        for (const auto metric : fh->MetricFields())
        {
            FillJson(x, metric, rawMetrics, i);
        }
        single["LF_METRICS"].append(x);
    }
}


// Entry point
int main(int argc, char* argv[])
{
    try
    {
      auto parser =
          PacBio::Process::OptionParser()
              .description("BazViewer is an instrument to inspect a BAZ file.")
              .usage("bazviewer input.baz");

        parser.add_option("-v", "--version").dest("version").action_store_true().help("Print the tool version and exit");
        auto groupOpt = PacBio::Process::OptionGroup(parser, "Optional parameters");
        groupOpt.add_option("-s", "--sChunksToZMWs").action_store_true().help("Mapping of super-chunks to ZMWs");
        groupOpt.add_option("-d", "--detail").action_store_true().help("Per readout and metric information");
        groupOpt.add_option("-m", "--metric").action_store_true().help("Metric only information");
        groupOpt.add_option("-l", "--list").action_store_true().help("List ZMW numbers");
        groupOpt.add_option("-n").dest("number").action_store().metavar("RANGES").type_string().help("Only print given ZMW NUMBERs");
        groupOpt.add_option("-i").dest("id").action_store().metavar("RANGES").type_string().help("Only print given ZMW IDs");
        groupOpt.add_option("-f").dest("first").action_store_true().help("Only print first ZMW");
        groupOpt.add_option("--frames").type_long().set_default(0x7FFFFFFF).help("Range of frames to process. Default is all frames.");
        groupOpt.add_option("--summary").action_store_true().help("Generate full file summary");
        groupOpt.add_option("--checksums").action_store_true().help("Dump index,number,checksum for every ZMW to stdout."
                                                                    "Checksums are summed over all LFM blocks, so there is one checksum for the entire ZMW.");
        groupOpt.add_option("--blockchecksums").action_store_true().help("Dump index,number,checksum for every block of every ZMW to stdout.");
        groupOpt.add_option("--silent").action_store_true().help("No logging output.");

        groupOpt.add_option("-H", "--hdf5Output").dest("hdf5Output").metavar("HDF5_FILENAME").action_store()
            .help("Output metrics to specifiied HDF5 file.  Suppresses JSON output.");

        parser.add_option_group(groupOpt);
        PacBio::Process::Values options = parser.parse_args(argc, argv);
        std::vector<std::string> args = parser.args();


        // Print version
        if (options.get("version"))
        {
            std::cerr << "bazviewer version: " << BAZIO_VERSION << std::endl;
            return 0;
        }

        // Check input
        if (args.size() != 1)
        {
            std::cerr << "Please provide one BAZ file" << std::endl;
            return 1;
        }

        const bool silent = options.get("silent");
        if (silent)
        {
            Logging::PBLogger::SetMinimumSeverityLevel(Logging::LogLevel::ERROR);
        }
        BazReader reader({ args[0] }, 1, 1000, silent);

        // HDF5 mode is only compatible with metrics output
        if (!options["hdf5Output"].empty()) {
            if (options.get("sChunksToZMWs") ||
                options.get("detail"))
            {
                std::cerr << "Only metrics output is supported in HDF5 mode" << std::endl;
                return 1;
            }
        }

        // Whiteliste processing
        if (!options["number"].empty() && !options["id"].empty())
        {
            std::cerr << "Options -i and -n are mutually exclusive" << std::endl;
            return 1;
        }
        bool filterNumber = !options["number"].empty();
        bool filterId = !options["id"].empty();
        bool filterFirst = options.get("first");
        uint32_t lastFrame = options.get("frames");
        bool filter = filterNumber || filterId || filterFirst;
        std::vector<uint32_t> whiteList;
        auto& fh = reader.Fileheader();
        if (filter)
        {
            if (filterFirst)
            {
                whiteList.push_back(0);
            }
            else
            {
                if (filterNumber)
                    whiteList = ParseData(options["number"]);
                else
                    whiteList = ParseData(options["id"]);

                std::sort(whiteList.begin(), whiteList.end());
                auto last = std::unique(whiteList.begin(), whiteList.end());
                whiteList.erase(last, whiteList.end());

                if (filterNumber)
                {
                    for (auto& w : whiteList) w = fh.ZmwNumberToIndex(w);
                    std::sort(whiteList.begin(), whiteList.end());
                }
            }
        }

        if(options.get("list"))
        {
            auto& header = reader.Fileheader();
            for (const auto& num : header.ZmwNumbers())
            {
                std::cout << num << "\n";
            }
            std::cout << std::endl;
            return 0;
        }
        if (options.get("checksums"))
        {
            auto& header = reader.Fileheader();

            while (reader.HasNext())
            {
                auto zmwData = reader.NextSlice();
                for (const auto& data : zmwData)
                {
                    uint32_t number = header.ZmwNumbers()[data.ZmwIndex()];
                    const auto& metrics = ParseMetrics(fh.MetricFields(),
                                                       fh.MetricFrames(),
                                                       fh.FrameRateHz(),
                                                       fh.RelativeAmplitudes(),
                                                       fh.BaseMap(),
                                                       data, false);
                    auto checksums = metrics.PixelChecksum().data();
                    int64_t checksum = 0;
                    for (auto x : checksums)
                    {
                        checksum += x;
                    }
                    std::cout << data.ZmwIndex() << "\t" << number << "\t" << checksum << "\n";
                }
            }
            std::cout << std::flush;
            return 0;
        }
        if (options.get("blockchecksums"))
        {
            auto& header = reader.Fileheader();

            std::cout << "zmw" << "\t" << "block" << "\t" << " blockchecksum" << std::endl;
            while (reader.HasNext())
            {
                auto zmwData = reader.NextSlice();
                for (const auto& data : zmwData)
                {
                    uint32_t number = header.ZmwNumbers()[data.ZmwIndex()];
                    const auto& metrics = ParseMetrics(fh.MetricFields(),
                                                       fh.MetricFrames(),
                                                       fh.FrameRateHz(),
                                                       fh.RelativeAmplitudes(),
                                                       fh.BaseMap(),
                                                       data, false);
                    auto checksums = metrics.PixelChecksum().data();
                    uint32_t block = 0;
                    for (auto checksum : checksums)
                    {
                        std::cout << number << "\t" << block << "\t" << checksum << "\n";
                        block++;
                    }
                }
            }
            std::cout << std::flush;
            return 0;
        }


        Jval overviewJsonFile;
        overviewJsonFile["TYPE"] = "BAZ_OVERVIEW";

        if (options.get("sChunksToZMWs"))
        {
            auto& overviewJson = overviewJsonFile["SCHUNK_TO_ZMWS"];
            const auto& schunkToIds = reader.SuperChunkToZmwHeaders();
            for (size_t i = 0; i < schunkToIds.size(); ++i)
            {
                const auto& schunk = schunkToIds[i];
                Jval thisChunk;
                thisChunk["SCHUNK_ID"] = i;
                Jval field;
                for (const auto& zmw : schunk)
                {

                    Jval jzmw;
                    if (filter && std::find(whiteList.begin(), whiteList.end(), zmw.zmwIndex) == whiteList.end())
                        continue;
                    jzmw["ZMW_ID"]     = zmw.zmwIndex;
                    jzmw["ZMW_NUMBER"] = fh.ZmwIndexToNumber(zmw.zmwIndex);
                    jzmw["NUM_EVENTS"] = zmw.numEvents;
                    jzmw["NUM_HFMBS"]  = zmw.numHFMBs;
                    jzmw["NUM_MFMBS"]  = zmw.numMFMBs;
                    jzmw["NUM_LFMBS"]  = zmw.numLFMBs;
                    jzmw["PACKET_STREAM_BYTE_SIZE"] = zmw.packetsByteSize;
                    field.append(jzmw);
                    if (filter && std::find(whiteList.begin(), whiteList.end(), zmw.zmwIndex) == whiteList.end())
                        break;
                }
                thisChunk["ZMWS"] = field;
                overviewJson.append(thisChunk);
            }
        }

        if (options.get("detail"))
        {
            auto& stitched = overviewJsonFile["STITCHED"];
            bool stop = false;
            auto packetFieldToBamIDWithOverallQV = PacketFieldMap::packetBaseFieldToBamID;
            packetFieldToBamIDWithOverallQV[PacketFieldName::OVERALL_QV] = std::make_pair("oq", FieldType::CHAR);
            while (reader.HasNext())
            {
                if (filter)
                {
                    const auto nextIds = reader.NextZmwIds();
                    std::vector<int> intersection;
                    std::set_intersection(whiteList.begin(), whiteList.end(),
                                          nextIds.begin(), nextIds.end(),
                                          std::back_inserter(intersection));

                    if (intersection.empty())
                    {
                        reader.SkipNextSlice();
                        continue;
                    }
                }

                Jval single;
                auto zmwData = reader.NextSlice();
                for (const auto& data : zmwData)
                {
                    const auto& eventData = BazIO::BazEventData(ParsePackets(fh.PacketGroups(), fh.PacketFields(), data));
                    if (filter && std::find(whiteList.begin(), whiteList.end(), data.ZmwIndex()) == whiteList.end())
                        continue;
                    for (size_t i = 0; i < eventData.NumEvents(); ++i)
                    {
                        Jval x = eventData.EventToJson(i);
                        x["POS"] = (int)i;

                        single["DATA"].append(x);
                    }

                    FillMetrics(&fh, data, eventData.Internal(), lastFrame, single);

                    stitched.append(single);
                    if (filter && data.ZmwIndex() == whiteList[whiteList.size() - 1])
                    {
                        stop = true;
                        break;
                    }
                }
                if (stop) break;
            }
        }

        if (options.get("summary"))
        {
            uint64_t events = 0;
            uint64_t explicitBases = 0;
            std::map<char,uint32_t> explicitCalls;
            while (reader.HasNext())
            {
                auto zmwData = reader.NextSlice();
                for (const auto& data: zmwData)
                {
                    const auto& eventData = BazIO::BazEventData(ParsePackets(fh.PacketGroups(), fh.PacketFields(), data));
                    const bool hasIsBase  = !eventData.IsBase().empty();
                    for (size_t i = 0; i < eventData.NumEvents(); ++i)
                    {
                        auto b = eventData.Readouts()[i];
                        explicitCalls[b]++;
                        if (hasIsBase && eventData.IsBase(i)) explicitBases++;
                        events++;
                    }
                }
            }
            std::cout << "events:" << events << std::endl;
            std::cout << "bases:"  << explicitBases << std::endl;
            for(const auto& kv : explicitCalls)
            {
                std::cout << "call " << kv.first << ":" << kv.second << std::endl;
            }
        }

        bool hdf5Mode = false;
        hdf5Mode = !options["hdf5Output"].empty();
        MetricsAggregator metricsForHDF5(&fh);

        if (options.get("metric") || hdf5Mode)
        {
            auto& stitched = overviewJsonFile["METRICS"];

            bool stop = false;
            while (reader.HasNext())
            {
                if (filter)
                {
                    const auto nextIds = reader.NextZmwIds();
                    std::vector<int> intersection;
                    std::set_intersection(whiteList.begin(), whiteList.end(),
                                          nextIds.begin(), nextIds.end(),
                                          std::back_inserter(intersection));

                    if (intersection.empty())
                    {
                        reader.SkipNextSlice();
                        continue;
                    }
                }

                auto zmwData = reader.NextSlice();
                for (const auto& data: zmwData)
                {
                    const auto& eventData = BazIO::BazEventData(ParsePackets(fh.PacketGroups(), fh.PacketFields(), data));
                    if (filter && std::find(whiteList.begin(), whiteList.end(), data.ZmwIndex()) == whiteList.end())
                        continue;

                    if (hdf5Mode)
                    {
                        metricsForHDF5.Fill(data);
                    }
                    else
                    {
                        Jval single;
                        FillMetrics(&fh, data, eventData.Internal(), lastFrame, single);
                        stitched.append(single);
                    }

                    if (filter && data.ZmwIndex() == whiteList[whiteList.size() - 1])
                    {
                        stop = true;
                        break;
                    }
                }
                if (stop) break;
            }
        }

        if (hdf5Mode)
        {
            metricsForHDF5.DumpToFile(options["hdf5Output"]);
        }
        else
        {
            std::cout << overviewJsonFile << std::endl;
        }

        return 0; // normal return
    }
    catch(const std::exception& ex)
    {
        //PBLOG_FATAL << "Exception caught: " << ex.what();
        std::cerr << "Exception caught: " << ex.what() << std::endl;
    }

    return 1; // error return
}
