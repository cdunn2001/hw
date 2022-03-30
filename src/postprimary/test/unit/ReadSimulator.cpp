
#include "ReadSimulator.h"
#include <postprimary/bam/SubreadLabelerMetrics.h>

using namespace PacBio;

RegionLabel GenerateEmpyHQ()
{
    return RegionLabel(0, 0, 0, RegionLabelType::HQREGION);
}

BlockLevelMetrics SimulateMetrics(const ReadConfig& config)
{
    int numBases = config.numBases;

    int metricFrames = 0;
    int numMetricBlocks = 0;
    std::vector<MetricField> fields;

    const auto& fh = config.GenerateHeader();
    if (!fh.MetricFields().empty())
    {
        fields = fh.MetricFields();
        metricFrames = fh.MetricFrames();
        numMetricBlocks = config.NumberMetricBlocks();
    }
    else
    {
        assert(false);
    }

    // we want to round the start down to the nearest block and the end up (it
    // is an exclusive boundary)
    int seqStartBlock = config.seqstart / metricFrames;
    int seqEndBlock = static_cast<int>(std::ceil(config.seqend/static_cast<double>(metricFrames)));

    int numBasePerMBlock = numBases / (seqEndBlock - seqStartBlock);

    int totBases = 0;

    // Add high-frequency metrics.
    RawMetricData rawMetrics(fields);
    for (int i = 0; i < numMetricBlocks; ++i)
    {
        rawMetrics.FloatMetric(MetricFieldName::BASELINE_MEAN).push_back(20);
        rawMetrics.FloatMetric(MetricFieldName::BASELINE_SD).push_back(40);

        rawMetrics.UIntMetric(MetricFieldName::PKMID_FRAMES_A).push_back(100);
        rawMetrics.UIntMetric(MetricFieldName::PKMID_FRAMES_C).push_back(200);
        rawMetrics.UIntMetric(MetricFieldName::PKMID_FRAMES_G).push_back(300);
        rawMetrics.UIntMetric(MetricFieldName::PKMID_FRAMES_T).push_back(400);


        if (i >= seqStartBlock && i < seqEndBlock)
        {
            if (i == seqEndBlock - 1 && config.dumpBasesInLastBlock) numBasePerMBlock = numBases - totBases;
            rawMetrics.UIntMetric(MetricFieldName::NUM_PULSES).push_back(numBasePerMBlock);
            rawMetrics.UIntMetric(MetricFieldName::NUM_BASES_A).push_back(numBasePerMBlock/4);
            rawMetrics.UIntMetric(MetricFieldName::NUM_BASES_C).push_back(numBasePerMBlock/4);
            rawMetrics.UIntMetric(MetricFieldName::NUM_BASES_G).push_back(numBasePerMBlock/4);
            rawMetrics.UIntMetric(MetricFieldName::NUM_BASES_T).push_back(numBasePerMBlock - numBasePerMBlock/4 * 3);
            rawMetrics.UIntMetric(MetricFieldName::BASE_WIDTH).push_back(400);
            rawMetrics.UIntMetric(MetricFieldName::PULSE_WIDTH).push_back(400);

            rawMetrics.FloatMetric(MetricFieldName::PKMID_A).push_back(config.PKMID_A);
            rawMetrics.FloatMetric(MetricFieldName::PKMID_C).push_back(config.PKMID_C);
            rawMetrics.FloatMetric(MetricFieldName::PKMID_G).push_back(config.PKMID_G);
            rawMetrics.FloatMetric(MetricFieldName::PKMID_T).push_back(config.PKMID_T);
            rawMetrics.FloatMetric(MetricFieldName::PKMAX_A).push_back(config.PKMAX_A);
            rawMetrics.FloatMetric(MetricFieldName::PKMAX_C).push_back(config.PKMAX_C);
            rawMetrics.FloatMetric(MetricFieldName::PKMAX_G).push_back(config.PKMAX_G);
            rawMetrics.FloatMetric(MetricFieldName::PKMAX_T).push_back(config.PKMAX_T);

            rawMetrics.FloatMetric(MetricFieldName::PKZVAR_A).push_back(config.PKZVAR_A);
            rawMetrics.FloatMetric(MetricFieldName::PKZVAR_C).push_back(config.PKZVAR_C);
            rawMetrics.FloatMetric(MetricFieldName::PKZVAR_G).push_back(config.PKZVAR_G);
            rawMetrics.FloatMetric(MetricFieldName::PKZVAR_T).push_back(config.PKZVAR_T);
            rawMetrics.FloatMetric(MetricFieldName::BPZVAR_A).push_back(config.BPZVAR_A);
            rawMetrics.FloatMetric(MetricFieldName::BPZVAR_C).push_back(config.BPZVAR_C);
            rawMetrics.FloatMetric(MetricFieldName::BPZVAR_G).push_back(config.BPZVAR_G);
            rawMetrics.FloatMetric(MetricFieldName::BPZVAR_T).push_back(config.BPZVAR_T);

            rawMetrics.UIntMetric(MetricFieldName::DME_STATUS).push_back(1);

            rawMetrics.UIntMetric(MetricFieldName::NUM_HALF_SANDWICHES).push_back(numBasePerMBlock * 0.017);
            rawMetrics.UIntMetric(MetricFieldName::NUM_SANDWICHES).push_back(6);
            rawMetrics.UIntMetric(MetricFieldName::NUM_PULSE_LABEL_STUTTERS).push_back(numBasePerMBlock * 0.4);
            rawMetrics.FloatMetric(MetricFieldName::TRACE_AUTOCORR).push_back(5200);
            rawMetrics.FloatMetric(MetricFieldName::PULSE_DETECTION_SCORE).push_back(-5);

            totBases += numBasePerMBlock;
        } else {
            rawMetrics.UIntMetric(MetricFieldName::NUM_PULSES).push_back(0);
            rawMetrics.UIntMetric(MetricFieldName::NUM_BASES_A).push_back(0);
            rawMetrics.UIntMetric(MetricFieldName::NUM_BASES_C).push_back(0);
            rawMetrics.UIntMetric(MetricFieldName::NUM_BASES_G).push_back(0);
            rawMetrics.UIntMetric(MetricFieldName::NUM_BASES_T).push_back(0);
            rawMetrics.UIntMetric(MetricFieldName::PULSE_WIDTH).push_back(0);
            rawMetrics.UIntMetric(MetricFieldName::BASE_WIDTH).push_back(0);

            rawMetrics.FloatMetric(MetricFieldName::PKMID_A).push_back(0);
            rawMetrics.FloatMetric(MetricFieldName::PKMID_C).push_back(0);
            rawMetrics.FloatMetric(MetricFieldName::PKMID_G).push_back(0);
            rawMetrics.FloatMetric(MetricFieldName::PKMID_T).push_back(0);
            rawMetrics.FloatMetric(MetricFieldName::PKMAX_A).push_back(0);
            rawMetrics.FloatMetric(MetricFieldName::PKMAX_C).push_back(0);
            rawMetrics.FloatMetric(MetricFieldName::PKMAX_G).push_back(0);
            rawMetrics.FloatMetric(MetricFieldName::PKMAX_T).push_back(0);

            rawMetrics.FloatMetric(MetricFieldName::PKZVAR_A).push_back(0);
            rawMetrics.FloatMetric(MetricFieldName::PKZVAR_C).push_back(0);
            rawMetrics.FloatMetric(MetricFieldName::PKZVAR_G).push_back(0);
            rawMetrics.FloatMetric(MetricFieldName::PKZVAR_T).push_back(0);
            rawMetrics.FloatMetric(MetricFieldName::BPZVAR_A).push_back(0);
            rawMetrics.FloatMetric(MetricFieldName::BPZVAR_C).push_back(0);
            rawMetrics.FloatMetric(MetricFieldName::BPZVAR_G).push_back(0);
            rawMetrics.FloatMetric(MetricFieldName::BPZVAR_T).push_back(0);

            rawMetrics.UIntMetric(MetricFieldName::DME_STATUS).push_back(0);

            rawMetrics.UIntMetric(MetricFieldName::NUM_HALF_SANDWICHES).push_back(0);
            rawMetrics.UIntMetric(MetricFieldName::NUM_SANDWICHES).push_back(0);
            rawMetrics.UIntMetric(MetricFieldName::NUM_PULSE_LABEL_STUTTERS).push_back(0);
            rawMetrics.FloatMetric(MetricFieldName::TRACE_AUTOCORR).push_back(0);
            rawMetrics.FloatMetric(MetricFieldName::PULSE_DETECTION_SCORE).push_back(-7);
        }
        rawMetrics.UIntMetric(MetricFieldName::NUM_FRAMES).push_back(metricFrames);
    }

    return BlockLevelMetrics(rawMetrics,
                             metricFrames, fh.FrameRateHz(), fh.RelativeAmplitudes(), fh.BaseMap(),
                             true);
}
// gcc 6.3.0 was starting to have ICE errors relating to default
// arguments to function parameters
BlockLevelMetrics SimulateMetrics() { return SimulateMetrics(ReadConfig{}); }

EventData SimulateEventData(const ReadConfig& config)
{
    bool exclude = config.excludePulses;
    int numBases = config.numBases;

    static const std::array<char, 4> baseMap = {'A', 'C', 'T', 'G'};

    // Add bases.
    std::map<BazIO::PacketFieldName, std::vector<uint32_t>> fields;
    std::vector<InsertState> states;
    const int fixedIPD = 5;
    const int fixedPW = 6;
    size_t frame = fixedIPD;
    for (int i = 0, j = 0, k = 0; i < numBases; ++i, ++j, ++k)
    {
        if (j == 4) j = 0;
        if (k == 5) k = 0;
        fields[BazIO::PacketFieldName::Label].push_back(baseMap[j]);
        fields[BazIO::PacketFieldName::StartFrame].push_back(frame);
        fields[BazIO::PacketFieldName::PulseWidth].push_back(fixedPW);
        frame += fixedIPD + fixedPW;
        if (exclude && i % 10 == 0)
        {
            fields[BazIO::PacketFieldName::IsBase].push_back(0);
            states.push_back(InsertState::EX_SHORT_PULSE);
        } else {
            fields[BazIO::PacketFieldName::IsBase].push_back(1);
            states.push_back(InsertState::BASE);
        }
    }

    const auto& fh = config.GenerateHeader();
    EventData::Meta meta;
    meta.truncated = false;
    meta.xPos = fh.ZmwInformation().HoleY()[0];
    meta.yPos = fh.ZmwInformation().HoleY()[0];
    meta.zmwIdx = 0;
    meta.zmwNum = fh.ZmwInformation().ZmwIndexToNumber(0);
    meta.features = fh.ZmwInformation().HoleFeaturesMask()[0];
    meta.holeType = 0;
    return EventData(
                     meta,
                     BazIO::BazEventData(fields, {}),
                     std::move(states));
}
// gcc 6.3.0 was starting to have ICE errors relating to default
// arguments to function parameters
EventData SimulateEventData() { return SimulateEventData(ReadConfig{}); }

// Called by test
ZmwMetrics RunMetrics(const EventData& events,
                      const BlockLevelMetrics& metrics,
                      const RegionLabel& hqRegion,
                      const ReadConfig& config)
{
    // Create file header.
    const auto& fh = config.GenerateHeader();

    ProductivityMetrics prodClassifier(4, minEmptyTime, emptyOutlierTime);
    auto prod = prodClassifier.ComputeProductivityInfo(hqRegion, metrics, true);

    return ZmwMetrics(fh.MovieTimeInHrs(),
                      fh.FrameRateHz(),
                      hqRegion,
                      std::vector<RegionLabel>{},
                      metrics,
                      events,
                      prod,
                      ControlMetrics{},
                      AdapterMetrics{});
}


// Called by stats if necessary. Produces the ZmwStats object that holds many
// results from ZmwMetrics functions, but has some internal computation of its
// own.
// TODO: Factor out the computation in the ZmwStats object into a public
// interface so that it can be more easily tested
std::tuple<PacBio::Primary::ZmwStats, std::unique_ptr<PacBio::BazIO::FileHeader>> fillstats(
        const EventData& events,
        const BlockLevelMetrics& metrics,
        const RegionLabel& hqRegion,
        const ReadConfig& readconfig)
{
    auto fh = std::make_unique<PacBio::BazIO::FileHeader>(readconfig.GenerateHeader());
    const auto& zmwMetrics = RunMetrics(events, metrics, hqRegion, readconfig);
    PacBio::Primary::ZmwStats zmw{readconfig.numAnalogs, readconfig.numFilters, readconfig.NumberMetricBlocks()};
    using Platform = PacBio::Primary::Postprimary::Platform;
    Postprimary::ZmwStats::FillPerZmwStats(Platform::SEQUEL, hqRegion, zmwMetrics, events, metrics,
                                           false, false, zmw);
    return std::make_tuple(zmw, std::move(fh));
}

long long printTime(const std::chrono::high_resolution_clock::time_point& t0,
                    std::string prefix)
{
    auto t = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now() - t0).count();

    auto d = t / 10000 / 1000 / 1000 / 60 / 60 / 24;
    auto h = (t / 1000 / 1000 / 1000 / 60 / 60) % 24;
    auto m = (t / 1000 / 1000 / 1000 / 60) % 60;
    auto s = (t / 1000 / 1000 / 1000) % 60;
    auto ms = (t / 1000 / 1000) % 1000;
    auto us = (t / 1000) % 1000;
    auto ns = t % 1000;
    std::stringstream ss;
    ss << prefix << "\t: ";
    if (d > 0) ss << d << "d ";
    if (h > 0) ss << h << "h ";
    if (m > 0) ss << m << "m ";
    if (s > 0) ss << s << "s ";
    if (ms > 0) ss << ms << "ms ";
    if (us > 0) ss << us << "us ";
    if (ns > 0) ss << ns << "ns ";
    std::cerr << ss.str() << "\t" << std::endl;

    return t;
}
