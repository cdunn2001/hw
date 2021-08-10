#ifndef TEST_READSIMULATOR_H
#define TEST_READSIMULATOR_H

#include <fstream>

#include <boost/format.hpp>

#include <pacbio/smrtdata/Readout.h>

#include <bazio/file/FileHeader.h>
#include <bazio/file/FileHeaderBuilder.h>

#include <bazio/Simulation.h>

#include <dataTypes/PulseGroups.h>

#include <postprimary/insertfinder/InsertState.h>
#include <postprimary/bam/EventData.h>
#include <postprimary/bam/Platform.h>
#include <postprimary/stats/ProductivityMetrics.h>
#include <postprimary/stats/ZmwStats.h>

using namespace PacBio::Mongo::Data;
using namespace PacBio::Primary;
using namespace PacBio::Primary::Postprimary;

const std::string STS_FILENAME = "test.sts.xml";
const std::string METRICS_FILENAME = "test.sts.h5";
const uint32_t minEmptyTime{60};
const uint32_t emptyOutlierTime{3};

// TODO: move this to a class, take a fileheader as input to set some values,
// make some values getter only
struct ReadConfig
{
    using FileHeader = PacBio::BazIO::FileHeader;
    using FileHeaderBuilder = PacBio::BazIO::FileHeaderBuilder;
    // DONT go to crazy with the local pulserate or you will overflow the int16
    // max on metric blocks (imposed during a conversion to double)
    //
    // In bases
    int32_t numBases = 81072;

    // In frames
    int32_t numFrames = 5242880; 
    int32_t hfnf = 1024;
    int32_t mfnf = 4096;
    int32_t lfnf = 16384;
    int32_t seqstart = 100;
    int32_t seqend = numFrames;

    // In bases
    // Double inclusive, zero indexed:
    int32_t hqstart = 0;
    int32_t hqend = numBases;

    int32_t insertsize = 9900;
    int32_t insertstride = 100;

    int32_t PKMID_A = 10;
    int32_t PKMID_C = 20;
    int32_t PKMID_G = 30;
    int32_t PKMID_T = 40;

    int32_t PKMAX_A = 15;
    int32_t PKMAX_C = 25;
    int32_t PKMAX_G = 35;
    int32_t PKMAX_T = 45;

    int32_t PKZVAR_A = 1000;
    int32_t PKZVAR_C = 1000;
    int32_t PKZVAR_G = 1000;
    int32_t PKZVAR_T = 1000;

    int32_t BPZVAR_A = 7;
    int32_t BPZVAR_C = 7;
    int32_t BPZVAR_G = 7;
    int32_t BPZVAR_T = 7;

    int32_t score = 0;

    bool excludePulses = false;
    bool dumpBasesInLastBlock = true;
    uint32_t numAnalogs = 4;
    uint32_t numFilters = 2;

    uint32_t nhfb() const { return (numFrames  + hfnf - 1) / hfnf; }
    uint32_t nmfb() const { return (numFrames  + mfnf - 1) / mfnf; }
    uint32_t nlfb() const { return (numFrames  + lfnf - 1) / lfnf; }

    FileHeader GenerateHeader() const
    {
        FileHeaderBuilder builder(
                "SimulatedMovie",
                80,
                numFrames,
                ProductionPulses::Params(),
                PacBio::SmrtData::MetricsVerbosity::HIGH,
                generateExperimentMetadata(),
                "{}", // basecaller config
                std::vector<uint32_t>(1, 4194368),
                {}, //const std::vector<uint32_t> zmwUnitFeatures,
                hfnf,
                mfnf,
                lfnf);

        auto charVec = builder.CreateJSONCharVector();
        return FileHeader(charVec.data(), charVec.size());
    }

    RegionLabel GenerateHQRegion() const
    {
        assert(hqend >= hqstart);
        RegionLabel hqRegion(
            hqstart, hqend,
            score, RegionLabelType::HQREGION);
        hqRegion.pulseBegin = hqstart;
        hqRegion.pulseEnd = hqend;
        return hqRegion;
    }
};

RegionLabel GenerateEmpyHQ();

BlockLevelMetrics SimulateMetrics(const ReadConfig& config);

// gcc 6.3.0 was starting to have ICE errors relating to default
// arguments to function parameters
BlockLevelMetrics SimulateMetrics();

EventData SimulateEventData(const ReadConfig& config);

// gcc 6.3.0 was starting to have ICE errors relating to default
// arguments to function parameters
EventData SimulateEventData();

// Called by test
ZmwMetrics RunMetrics(const EventData& events,
                   const BlockLevelMetrics& metrics,
                   const RegionLabel& hqRegion,
                   const ReadConfig& config);


// Called by stats if necessary. Produces the ZmwStats object that holds many
// results from ZmwMetrics functions, but has some internal computation of its
// own.
// TODO: Factor out the computation in the ZmwStats object into a public
// interface so that it can be more easily tested
std::tuple<PacBio::Primary::ZmwStats, std::unique_ptr<PacBio::BazIO::FileHeader>> fillstats(
        const EventData& events,
        const BlockLevelMetrics& metrics,
        const RegionLabel& hqRegion,
        const ReadConfig& readconfig);

long long printTime(const std::chrono::high_resolution_clock::time_point& t0,
                          std::string prefix);

#endif //TEST_READSIMULATOR_H
