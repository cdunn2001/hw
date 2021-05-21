//
// Created by jnguyen on 1/29/19.
//

#include "ReadSimulator.h"
#include <postprimary/bam/SubreadLabelerMetrics.h>

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
    MetricFrequency frequency = MetricFrequency::HIGH;

    const FileHeader& fh = config.GenerateHeader();
    if (!fh.HFMetricFields().empty())
    {
        assert(fh.MFMetricFields().empty());
        assert(fh.LFMetricFields().empty());
        fields = fh.HFMetricFields();
        metricFrames = fh.HFMetricFrames();
        numMetricBlocks = config.nhfb();
        frequency = MetricFrequency::HIGH;
    } else if (!fh.MFMetricFields().empty()) {
        assert(fh.HFMetricFields().empty());
        assert(fh.LFMetricFields().empty());
        fields = fh.MFMetricFields();
        metricFrames = fh.MFMetricFrames();
        numMetricBlocks = config.nmfb();
        frequency = MetricFrequency::MEDIUM;
    } else if (!fh.LFMetricFields().empty()) {
        assert(fh.HFMetricFields().empty());
        assert(fh.MFMetricFields().empty());
        fields = fh.LFMetricFields();
        metricFrames = fh.LFMetricFrames();
        numMetricBlocks = config.nlfb();
        frequency = MetricFrequency::LOW;
    } else {
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
        rawMetrics.FloatMetric(MetricFieldName::BASELINE_RED_MEAN).push_back(10);
        rawMetrics.FloatMetric(MetricFieldName::BASELINE_GREEN_MEAN).push_back(20);
        rawMetrics.FloatMetric(MetricFieldName::BASELINE_RED_SD).push_back(30);
        rawMetrics.FloatMetric(MetricFieldName::BASELINE_GREEN_SD).push_back(40);

        rawMetrics.UIntMetric(MetricFieldName::PKMID_FRAMES_A).push_back(100);
        rawMetrics.UIntMetric(MetricFieldName::PKMID_FRAMES_C).push_back(200);
        rawMetrics.UIntMetric(MetricFieldName::PKMID_FRAMES_G).push_back(300);
        rawMetrics.UIntMetric(MetricFieldName::PKMID_FRAMES_T).push_back(400);

        rawMetrics.FloatMetric(MetricFieldName::ANGLE_RED).push_back(10);
        rawMetrics.FloatMetric(MetricFieldName::ANGLE_GREEN).push_back(30);
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

    return BlockLevelMetrics(rawMetrics, fh, frequency, true);
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
    std::vector<std::vector<uint32_t>> packetFields(PacketFieldName::allValues().size());
    std::vector<InsertState> states;
    for (int i = 0, j = 0, k = 0; i < numBases; ++i, ++j, ++k)
    {
        if (j == 4) j = 0;
        if (k == 5) k = 0;
        packetFields[static_cast<uint8_t>(PacketFieldName::READOUT)].push_back(baseMap[j]);
        packetFields[static_cast<uint8_t>(PacketFieldName::SUB_TAG)].push_back(k);
        packetFields[static_cast<uint8_t>(PacketFieldName::DEL_TAG)].push_back(k);
        packetFields[static_cast<uint8_t>(PacketFieldName::SUB_QV)].push_back(1);
        packetFields[static_cast<uint8_t>(PacketFieldName::DEL_QV)].push_back(2);
        packetFields[static_cast<uint8_t>(PacketFieldName::MRG_QV)].push_back(3);
        packetFields[static_cast<uint8_t>(PacketFieldName::INS_QV)].push_back(4);
        packetFields[static_cast<uint8_t>(PacketFieldName::IPD_LL)].push_back(5);
        packetFields[static_cast<uint8_t>(PacketFieldName::PW_LL)].push_back(6);
        if (exclude && i % 10 == 0)
        {
            packetFields[static_cast<uint8_t>(PacketFieldName::IS_BASE)].push_back(0);
            states.push_back(InsertState::EX_SHORT_PULSE);
        } else {
            packetFields[static_cast<uint8_t>(PacketFieldName::IS_BASE)].push_back(1);
            states.push_back(InsertState::BASE);
        }
        if (exclude && i % 10 == 1)
        {
            packetFields[static_cast<uint8_t>(PacketFieldName::IPD_LL)].back() += 11;
        }
        packetFields[static_cast<uint8_t>(PacketFieldName::IS_PULSE)].push_back(1);
    }

    const FileHeader& fh = config.GenerateHeader();
    return EventData(
            fh, 0, false,
            BazEventData(std::move(packetFields)),
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

    return ZmwMetrics(fh, hqRegion, std::vector<RegionLabel>{}, metrics, events, prod, ControlMetrics{}, AdapterMetrics{});
}


// Called by stats if necessary. Produces the ZmwStats object that holds many
// results from ZmwMetrics functions, but has some internal computation of its
// own.
// TODO: Factor out the computation in the ZmwStats object into a public
// interface so that it can be more easily tested
std::tuple<PacBio::Primary::ZmwStats, std::unique_ptr<FileHeader>> fillstats(
        const EventData& events,
        const BlockLevelMetrics& metrics,
        const RegionLabel& hqRegion,
        const ReadConfig& readconfig)
{
    auto fh = std::make_unique<FileHeader>(readconfig.GenerateHeader());
    const auto& zmwMetrics = RunMetrics(events, metrics, hqRegion, readconfig);
    PacBio::Primary::ZmwStats zmw{readconfig.numAnalogs, readconfig.numFilters,
                                  readconfig.nlfb()};
    using Platform = PacBio::Primary::Postprimary::Platform;
    Postprimary::ZmwStats::FillPerZmwStats(Platform::SEQUEL, *fh, hqRegion, zmwMetrics, events, metrics,
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
