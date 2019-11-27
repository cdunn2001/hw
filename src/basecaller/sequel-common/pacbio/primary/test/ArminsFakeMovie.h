//
// Created by mlakata on 10/31/16.
//

#ifndef SEQUELACQUISITION_ARMINSFAKEMOVIE_H
#define SEQUELACQUISITION_ARMINSFAKEMOVIE_H

#include <boost/filesystem.hpp>
#include <pacbio/primary/Simulation.h>

using namespace PacBio::Primary;

inline FileHeaderBuilder ArminsFakeMovie( )
{
    FileHeaderBuilder fhb("ArminsFakeMovie", 100.0, 100.0 * 60 * 60 * 3,
                          PacBio::SmrtData::Readout::BASES,
                          PacBio::SmrtData::MetricsVerbosity::MINIMAL,
                          generateExperimentMetadata(),
                          "{}", {}, {},
                          1024, // hf metric block frames
                          4096, // mf metric block frames
                          16384, // superchunk frames
                          ChipClass::Sequel, false, true, true, false);

    return fhb;
}

inline FileHeaderBuilder ArminsFakeMovieHighVerbosity()
{
    FileHeaderBuilder fhb("ArminsFakeMovie", 100.0, 100.0 * 60 * 60 * 3,
                          PacBio::SmrtData::Readout::PULSES,
                          PacBio::SmrtData::MetricsVerbosity::HIGH,
                          generateExperimentMetadata(),
                          "{}", {}, {},
                          1024, // hf metric block frames
                          4096, // mf metric block frames
                          16384, // superchunk frames
                          ChipClass::Sequel,false,true, true, false);

    return fhb;
}

inline FileHeaderBuilder ArminsFakeNoMetricsMovie( )
{
    FileHeaderBuilder fhb("ArminsFakeMovie", 100.0, 100.0 * 60 * 60 * 3,
                          PacBio::SmrtData::Readout::BASES,
                          PacBio::SmrtData::MetricsVerbosity::NONE,
                          generateExperimentMetadata(),
                          "{}", {}, {},
                          1024, // hf metric block frames
                          4096, // mf metric block frames
                          16384, // superchunk frames
                          ChipClass::Sequel,false, true, true, false);

    return fhb;
}

/// for unit testing only
template <typename MetricBlock_T>
inline void CheckFileSizes(const BazWriter<MetricBlock_T>& writer)
{
#if 0
    writer.Summarize(std::cout);
    writer.GetFileHeaderBuilder().EstimatesSummary(std::cout, baseRate);
#endif
    double baseRate = writer.NumEvents() /
            (writer.GetFileHeaderBuilder().MovieLengthSeconds() *
                    writer.GetFileHeaderBuilder().MaxNumZmws());
    size_t bytesWritten = writer.BytesWritten();
    // require actual file size to be less than estimate, but within 50% of estimate too.
    EXPECT_LE (bytesWritten, writer.GetFileHeaderBuilder().ExpectedFileByteSize(baseRate)) << writer.Summary()
            << "\n" <<  writer.GetFileHeaderBuilder().EstimatesSummary( baseRate);
    EXPECT_LE ((int64_t)writer.GetFileHeaderBuilder().ExpectedFileByteSize(baseRate) - (int64_t)bytesWritten,
               bytesWritten * 0.50 + 10000 ) <<
            writer.Summary() << "\n" <<
            writer.GetFileHeaderBuilder().EstimatesSummary( baseRate);
    auto actualFileSize = boost::filesystem::file_size(writer.FilePath());
    EXPECT_FLOAT_EQ (actualFileSize, bytesWritten);
}

#endif //SEQUELACQUISITION_ARMINSFAKEMOVIE_H
