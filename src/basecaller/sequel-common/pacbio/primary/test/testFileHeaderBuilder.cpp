#include <gtest/gtest.h>

#include <assert.h>
#include <iostream>
#include <memory>
#include <string>
#include <stdio.h>
#include <string.h>
#include <json/json.h>
#include <json/reader.h>

#include <pacbio/dev/TemporaryDirectory.h>

#include <pacbio/smrtdata/MetricsVerbosity.h>
#include <pacbio/smrtdata/Readout.h>

#include <pacbio/primary/BazWriter.h>
#include <pacbio/primary/FileHeader.h>
#include <pacbio/primary/FileHeaderBuilder.h>
#include <pacbio/primary/PacketFieldName.h>
#include <pacbio/primary/Sanity.h>
#include <pacbio/primary/SmartMemory.h>
#include <pacbio/primary/Simulation.h>


using namespace PacBio::Primary;


static const std::string BASECALLER_CONFIG = "{}";
static const std::string EXPERIMENT_METADATA = generateExperimentMetadata();

using MFN = MetricFieldName;
using PFN = PacketFieldName;
using Mb = SequelMetricBlock;

static void WriteProduction(const std::string& filename)
{
    FileHeaderBuilder fhb("ArminsFakeMovie", 80.0, 80.0*60*60*3, Readout::BASES, MetricsVerbosity::MINIMAL,
                          EXPERIMENT_METADATA,
                          BASECALLER_CONFIG, {0},{},1024,4096,16384, ChipClass::Sequel, false, false, true, false);
    // std::cerr << fhb.CreateJSON() << std::endl;
    BazWriter<Mb> writer(filename, fhb, PacBio::Primary::BazIOConfig{}, 1000000);
    writer.WaitForTermination();
}

static FileHeader ReadProduction(const std::string& filename)
{
    auto file = SmartMemory::OpenFile(filename.c_str(), "rb");

    // Seek for SANITY block and last position of the file header
    uint64_t headerSize = Sanity::FindAndVerify(file);
    // Wrap header into smrt pointer
    std::vector<char> header(headerSize + 1);

    // Set position indicator to beginning
    std::rewind(file.get());

    // Read file header
    size_t result = std::fread(header.data(), 1, headerSize, file.get());
    if (result != headerSize)
        throw std::runtime_error("Cannot read file header!");

    FileHeader fh(header.data(), headerSize);

    return fh;
}


TEST(FileHeaderBuilder, NoDataPROD)
{
    const std::string FILENAME_PROD1 = "fileheaderbuilderPROD_test.baz";
    static PacBio::Dev::TemporaryDirectory tmpdir;
    std::string FILENAME_PROD = tmpdir.DirName() + "/" + FILENAME_PROD1;

    WriteProduction(FILENAME_PROD);
    auto fh = ReadProduction(FILENAME_PROD);

    EXPECT_EQ("ArminsFakeMovie", fh.MovieName());
    EXPECT_EQ((1), fh.BazMajorVersion());
    EXPECT_EQ((6), fh.BazMinorVersion());
    EXPECT_EQ((0), fh.BazPatchVersion());
    EXPECT_EQ((80), fh.FrameRateHz());
    EXPECT_EQ((1024), fh.HFMetricFrames());
    EXPECT_EQ((4096), fh.MFMetricFrames());
    EXPECT_EQ((16384), fh.LFMetricFrames());
    EXPECT_EQ((16384), fh.SliceLengthFrames());
    EXPECT_EQ((6), fh.HFMetricByteSize());
    EXPECT_EQ((34), fh.MFMetricByteSize());
    EXPECT_EQ((60), fh.LFMetricByteSize());
    EXPECT_EQ((80*60*60*3), fh.MovieLengthFrames());

    EXPECT_EQ((3), fh.HFMetricFields().size());

    EXPECT_EQ(MFN::NUM_BASES,           fh.HFMetricFields()[0].fieldName);
    EXPECT_EQ((16),                     fh.HFMetricFields()[0].fieldBitSize);
    EXPECT_FALSE(                       fh.HFMetricFields()[0].fieldSigned);
    EXPECT_EQ(1,                        fh.HFMetricFields()[0].fieldScalingFactor);

    EXPECT_EQ(MFN::NUM_PULSES,          fh.HFMetricFields()[1].fieldName);
    EXPECT_EQ((16),                     fh.HFMetricFields()[1].fieldBitSize);
    EXPECT_FALSE(                       fh.HFMetricFields()[1].fieldSigned);
    EXPECT_EQ(1,                        fh.HFMetricFields()[1].fieldScalingFactor);


    EXPECT_EQ((17),                     fh.MFMetricFields().size());

    EXPECT_EQ(MFN::NUM_BASES,           fh.MFMetricFields()[0].fieldName);
    EXPECT_EQ((16),                     fh.MFMetricFields()[0].fieldBitSize);
    EXPECT_FALSE(                       fh.MFMetricFields()[0].fieldSigned);
    EXPECT_EQ(1,                        fh.MFMetricFields()[0].fieldScalingFactor);

    EXPECT_EQ(MFN::NUM_PULSES,          fh.MFMetricFields()[1].fieldName);
    EXPECT_EQ((16),                     fh.MFMetricFields()[1].fieldBitSize);
    EXPECT_FALSE(                       fh.MFMetricFields()[1].fieldSigned);
    EXPECT_EQ(1,                        fh.MFMetricFields()[1].fieldScalingFactor);

    EXPECT_EQ(MFN::NUM_FRAMES,          fh.MFMetricFields()[2].fieldName);
    EXPECT_EQ((16),                     fh.MFMetricFields()[2].fieldBitSize);
    EXPECT_FALSE(                       fh.MFMetricFields()[2].fieldSigned);
    EXPECT_EQ(1,                        fh.MFMetricFields()[2].fieldScalingFactor);

    EXPECT_EQ(MFN::NUM_HALF_SANDWICHES, fh.MFMetricFields()[3].fieldName);
    EXPECT_EQ((16),                     fh.MFMetricFields()[3].fieldBitSize);
    EXPECT_FALSE(                       fh.MFMetricFields()[3].fieldSigned);
    EXPECT_EQ(1,                        fh.MFMetricFields()[3].fieldScalingFactor);

    EXPECT_EQ(MFN::NUM_SANDWICHES,      fh.MFMetricFields()[4].fieldName);
    EXPECT_EQ((16),                     fh.MFMetricFields()[4].fieldBitSize);
    EXPECT_FALSE(                       fh.MFMetricFields()[4].fieldSigned);
    EXPECT_EQ(1,                        fh.MFMetricFields()[4].fieldScalingFactor);

    EXPECT_EQ(MFN::NUM_PULSE_LABEL_STUTTERS, fh.MFMetricFields()[5].fieldName);
    EXPECT_EQ((16),                     fh.MFMetricFields()[5].fieldBitSize);
    EXPECT_FALSE(                       fh.MFMetricFields()[5].fieldSigned);
    EXPECT_EQ(1,                        fh.MFMetricFields()[5].fieldScalingFactor);

    EXPECT_EQ(MFN::PULSE_WIDTH,         fh.MFMetricFields()[6].fieldName);
    EXPECT_EQ((16),                     fh.MFMetricFields()[6].fieldBitSize);
    EXPECT_FALSE(                       fh.MFMetricFields()[6].fieldSigned);
    EXPECT_EQ(1,                        fh.MFMetricFields()[6].fieldScalingFactor);

    EXPECT_EQ(MFN::PKMAX_A,             fh.MFMetricFields()[11].fieldName);
    EXPECT_EQ((16),                     fh.MFMetricFields()[11].fieldBitSize);
    EXPECT_EQ(true,                     fh.MFMetricFields()[11].fieldSigned);
    EXPECT_EQ(0,                       fh.MFMetricFields()[11].fieldScalingFactor);

    EXPECT_EQ(MFN::PKMAX_C,             fh.MFMetricFields()[12].fieldName);
    EXPECT_EQ((16),                     fh.MFMetricFields()[12].fieldBitSize);
    EXPECT_EQ(true,                     fh.MFMetricFields()[12].fieldSigned);
    EXPECT_EQ(0,                       fh.MFMetricFields()[12].fieldScalingFactor);

    EXPECT_EQ(MFN::PKMAX_G,             fh.MFMetricFields()[13].fieldName);
    EXPECT_EQ((16),                     fh.MFMetricFields()[13].fieldBitSize);
    EXPECT_EQ(true,                     fh.MFMetricFields()[13].fieldSigned);
    EXPECT_EQ(0,                       fh.MFMetricFields()[13].fieldScalingFactor);

    EXPECT_EQ(MFN::PKMAX_T,             fh.MFMetricFields()[14].fieldName);
    EXPECT_EQ((16),                     fh.MFMetricFields()[14].fieldBitSize);
    EXPECT_EQ(true,                     fh.MFMetricFields()[14].fieldSigned);
    EXPECT_EQ(0,                       fh.MFMetricFields()[14].fieldScalingFactor);

    EXPECT_EQ(MFN::PULSE_DETECTION_SCORE,fh.MFMetricFields()[15].fieldName);
    EXPECT_EQ(16,                       fh.MFMetricFields()[15].fieldBitSize);
    EXPECT_EQ(true,                     fh.MFMetricFields()[15].fieldSigned);
    EXPECT_EQ(0,                      fh.MFMetricFields()[15].fieldScalingFactor);

    EXPECT_EQ((30),                     fh.LFMetricFields().size());

    EXPECT_EQ(MFN::NUM_BASES,           fh.LFMetricFields()[0].fieldName);
    EXPECT_EQ((16),                     fh.LFMetricFields()[0].fieldBitSize);
    EXPECT_FALSE(                       fh.LFMetricFields()[0].fieldSigned);
    EXPECT_EQ(1,                        fh.LFMetricFields()[0].fieldScalingFactor);

    EXPECT_EQ(MFN::NUM_PULSES,          fh.LFMetricFields()[1].fieldName);
    EXPECT_EQ((16),                     fh.LFMetricFields()[1].fieldBitSize);
    EXPECT_FALSE(                       fh.LFMetricFields()[1].fieldSigned);
    EXPECT_EQ(1,                        fh.LFMetricFields()[1].fieldScalingFactor);

    EXPECT_EQ(MFN::NUM_FRAMES,          fh.LFMetricFields()[2].fieldName);
    EXPECT_EQ((16),                     fh.LFMetricFields()[2].fieldBitSize);
    EXPECT_EQ(false,                    fh.LFMetricFields()[2].fieldSigned);
    EXPECT_EQ(1,                        fh.LFMetricFields()[2].fieldScalingFactor);

    EXPECT_EQ(MFN::PKMID_A,             fh.LFMetricFields()[3].fieldName);
    EXPECT_EQ((16),                     fh.LFMetricFields()[3].fieldBitSize);
    EXPECT_EQ(true,                     fh.LFMetricFields()[3].fieldSigned);
    EXPECT_EQ(0,                       fh.LFMetricFields()[3].fieldScalingFactor);

    EXPECT_EQ(MFN::PKMID_C,             fh.LFMetricFields()[4].fieldName);
    EXPECT_EQ((16),                     fh.LFMetricFields()[4].fieldBitSize);
    EXPECT_EQ(true,                     fh.LFMetricFields()[4].fieldSigned);
    EXPECT_EQ(0,                       fh.LFMetricFields()[4].fieldScalingFactor);

    EXPECT_EQ(MFN::PKMID_G,             fh.LFMetricFields()[5].fieldName);
    EXPECT_EQ((16),                     fh.LFMetricFields()[5].fieldBitSize);
    EXPECT_EQ(true,                     fh.LFMetricFields()[5].fieldSigned);
    EXPECT_EQ(0,                       fh.LFMetricFields()[5].fieldScalingFactor);

    EXPECT_EQ(MFN::PKMID_T,             fh.LFMetricFields()[6].fieldName);
    EXPECT_EQ((16),                     fh.LFMetricFields()[6].fieldBitSize);
    EXPECT_EQ(true,                     fh.LFMetricFields()[6].fieldSigned);
    EXPECT_EQ(0,                       fh.LFMetricFields()[6].fieldScalingFactor);


    EXPECT_EQ(MFN::BASELINE_RED_SD,     fh.LFMetricFields()[11].fieldName);
    EXPECT_EQ((16),                     fh.LFMetricFields()[11].fieldBitSize);
    EXPECT_EQ(true,                     fh.LFMetricFields()[11].fieldSigned);
    EXPECT_EQ(0,                       fh.LFMetricFields()[11].fieldScalingFactor);

    EXPECT_EQ(MFN::BASELINE_GREEN_SD,   fh.LFMetricFields()[12].fieldName);
    EXPECT_EQ((16),                     fh.LFMetricFields()[12].fieldBitSize);
    EXPECT_EQ(true,                     fh.LFMetricFields()[12].fieldSigned);
    EXPECT_EQ(0,                       fh.LFMetricFields()[12].fieldScalingFactor);

    EXPECT_EQ(MFN::BASELINE_RED_MEAN,   fh.LFMetricFields()[13].fieldName);
    EXPECT_EQ((16),                     fh.LFMetricFields()[13].fieldBitSize);
    EXPECT_EQ(true,                     fh.LFMetricFields()[13].fieldSigned);
    EXPECT_EQ(0,                       fh.LFMetricFields()[13].fieldScalingFactor);

    EXPECT_EQ(MFN::BASELINE_GREEN_MEAN, fh.LFMetricFields()[14].fieldName);
    EXPECT_EQ((16),                     fh.LFMetricFields()[14].fieldBitSize);
    EXPECT_EQ(true,                     fh.LFMetricFields()[14].fieldSigned);
    EXPECT_EQ(0,                       fh.LFMetricFields()[14].fieldScalingFactor);

    EXPECT_EQ(MFN::PULSE_WIDTH,           fh.LFMetricFields()[15].fieldName);
    EXPECT_EQ((16),                       fh.LFMetricFields()[15].fieldBitSize);
    EXPECT_FALSE(                         fh.LFMetricFields()[15].fieldSigned);
    EXPECT_EQ(1,                          fh.LFMetricFields()[15].fieldScalingFactor);

    EXPECT_EQ(MFN::BASE_WIDTH,            fh.LFMetricFields()[16].fieldName);
    EXPECT_EQ((16),                       fh.LFMetricFields()[16].fieldBitSize);
    EXPECT_FALSE(                         fh.LFMetricFields()[16].fieldSigned);
    EXPECT_EQ(1,                          fh.LFMetricFields()[16].fieldScalingFactor);

    EXPECT_EQ(MFN::PIXEL_CHECKSUM,        fh.LFMetricFields()[17].fieldName);
    EXPECT_EQ((16),                       fh.LFMetricFields()[17].fieldBitSize);
    EXPECT_EQ(true,                       fh.LFMetricFields()[17].fieldSigned);
    EXPECT_EQ(1,                          fh.LFMetricFields()[17].fieldScalingFactor);

    EXPECT_EQ(MFN::ANGLE_A,               fh.LFMetricFields()[18].fieldName);
    EXPECT_EQ((16),                       fh.LFMetricFields()[18].fieldBitSize);
    EXPECT_EQ(true,                       fh.LFMetricFields()[18].fieldSigned);
    EXPECT_EQ(0,                        fh.LFMetricFields()[18].fieldScalingFactor);

    EXPECT_EQ(MFN::ANGLE_C,               fh.LFMetricFields()[19].fieldName);
    EXPECT_EQ((16),                       fh.LFMetricFields()[19].fieldBitSize);
    EXPECT_EQ(true,                       fh.LFMetricFields()[19].fieldSigned);
    EXPECT_EQ(0,                        fh.LFMetricFields()[19].fieldScalingFactor);

    EXPECT_EQ(MFN::ANGLE_G,               fh.LFMetricFields()[20].fieldName);
    EXPECT_EQ((16),                       fh.LFMetricFields()[20].fieldBitSize);
    EXPECT_EQ(true,                       fh.LFMetricFields()[20].fieldSigned);
    EXPECT_EQ(0,                        fh.LFMetricFields()[20].fieldScalingFactor);

    EXPECT_EQ(MFN::ANGLE_T,               fh.LFMetricFields()[21].fieldName);
    EXPECT_EQ((16),                       fh.LFMetricFields()[21].fieldBitSize);
    EXPECT_EQ(true,                       fh.LFMetricFields()[21].fieldSigned);
    EXPECT_EQ(0,                        fh.LFMetricFields()[21].fieldScalingFactor);


    EXPECT_EQ((5), fh.PacketFields().size());

    EXPECT_EQ(PFN::READOUT,   fh.PacketFields()[0].fieldName);
    EXPECT_EQ((2),            fh.PacketFields()[0].fieldBitSize);
    EXPECT_EQ((0),            fh.PacketFields()[0].hasFieldEscape);
    EXPECT_EQ((0b11000000),   fh.PacketFields()[0].fieldBitMask);
    EXPECT_EQ((6),            fh.PacketFields()[0].fieldBitShift);

    EXPECT_EQ(PFN::OVERALL_QV,fh.PacketFields()[1].fieldName);
    EXPECT_EQ((4),            fh.PacketFields()[1].fieldBitSize);
    EXPECT_EQ((0),            fh.PacketFields()[1].hasFieldEscape);
    EXPECT_EQ((0b00111100),   fh.PacketFields()[1].fieldBitMask);
    EXPECT_EQ((2),            fh.PacketFields()[1].fieldBitShift);

    EXPECT_EQ(PFN::IPD_V1,    fh.PacketFields()[3].fieldName);
    EXPECT_EQ((8),            fh.PacketFields()[3].fieldBitSize);

    EXPECT_EQ(PFN::PW_V1,     fh.PacketFields()[4].fieldName);
    EXPECT_EQ((8),            fh.PacketFields()[4].fieldBitSize);
}

static void WriteInternal(const std::string& filename)
{
    FileHeaderBuilder fhb("ArminsFakeMovie", 80.0, 80.0*60*60*3,Readout::PULSES, MetricsVerbosity::HIGH,
                          EXPERIMENT_METADATA,
                          BASECALLER_CONFIG, {0},{},1024,4096,16384,ChipClass::Sequel,false,false,true,false);
    // std::cerr << fhb.CreateJSON() << std::endl;
    BazWriter<Mb> writer(filename, fhb, PacBio::Primary::BazIOConfig{}, 1000000);
    writer.WaitForTermination();
}

static FileHeader ReadInternal(const std::string& filename)
{
    auto file = SmartMemory::OpenFile(filename.c_str(), "rb");

    // Seek for SANITY block and last position of the file header
    uint64_t headerSize = Sanity::FindAndVerify(file);
    // Wrap header into smrt pointer
    std::vector<char> header(headerSize + 1);

    // Set position indicator to beginning
    std::rewind(file.get());

    // Read file header
    size_t result = std::fread(header.data(), 1, headerSize, file.get());
    if (result != headerSize)
        throw std::runtime_error("Cannot read file header!");

    FileHeader fh(header.data(), headerSize);

    return fh;
}


TEST(FileHeaderBuilder, NoDataINT)
{
    const std::string FILENAME_INT1 = "fileheaderbuilderINT_test.baz";
    static PacBio::Dev::TemporaryDirectory tmpdir;
    std::string FILENAME_INT = tmpdir.DirName() + "/" + FILENAME_INT1;

    WriteInternal(FILENAME_INT);
    auto fh = ReadInternal(FILENAME_INT);

    EXPECT_EQ(fh.MovieName(), "ArminsFakeMovie");
    EXPECT_EQ(1,    fh.BazMajorVersion());
    EXPECT_EQ(6,    fh.BazMinorVersion());
    EXPECT_EQ(0,    fh.BazPatchVersion());
    EXPECT_EQ(80,   fh.FrameRateHz());
    EXPECT_EQ(1024, fh.HFMetricFrames());
    EXPECT_EQ(4096, fh.MFMetricFrames());
    EXPECT_EQ(16384,fh.LFMetricFrames());
    EXPECT_EQ(16384,fh.SliceLengthFrames());
    EXPECT_EQ(6,    fh.HFMetricByteSize());
    EXPECT_EQ(82,   fh.MFMetricByteSize());
    EXPECT_EQ(12,   fh.LFMetricByteSize());
    EXPECT_EQ(80*60*60*3, fh.MovieLengthFrames());

    EXPECT_EQ(3,  fh.HFMetricFields().size());
    EXPECT_EQ(MetricFieldName::NUM_BASES, fh.HFMetricFields()[0].fieldName );
    EXPECT_EQ(16, fh.HFMetricFields()[0].fieldBitSize);
    EXPECT_EQ(MetricFieldName::NUM_PULSES, fh.HFMetricFields()[1].fieldName );
    EXPECT_EQ(16, fh.HFMetricFields()[1].fieldBitSize);
    EXPECT_EQ(MetricFieldName::NUM_FRAMES, fh.HFMetricFields()[2].fieldName);
    EXPECT_EQ(16, fh.HFMetricFields()[2].fieldBitSize);

    EXPECT_EQ(41, fh.MFMetricFields().size());

    EXPECT_EQ(MetricFieldName::NUM_BASES,fh.MFMetricFields()[0].fieldName );
    EXPECT_EQ(16,fh.MFMetricFields()[0].fieldBitSize);
    EXPECT_EQ(MetricFieldName::NUM_PULSES, fh.MFMetricFields()[1].fieldName );
    EXPECT_EQ(16,fh.MFMetricFields()[1].fieldBitSize);
    EXPECT_EQ(MetricFieldName::NUM_FRAMES,fh.MFMetricFields()[2].fieldName );
    EXPECT_EQ(16,fh.MFMetricFields()[2].fieldBitSize);
    EXPECT_EQ(MetricFieldName::NUM_HALF_SANDWICHES,fh.MFMetricFields()[3].fieldName );
    EXPECT_EQ(16,fh.MFMetricFields()[3].fieldBitSize);
    EXPECT_EQ(MetricFieldName::NUM_SANDWICHES,fh.MFMetricFields()[4].fieldName );
    EXPECT_EQ(16,fh.MFMetricFields()[4].fieldBitSize);
    EXPECT_EQ(MetricFieldName::NUM_PULSE_LABEL_STUTTERS,fh.MFMetricFields()[5].fieldName );
    EXPECT_EQ(16,fh.MFMetricFields()[5].fieldBitSize);
    EXPECT_EQ(MetricFieldName::PKMID_A,fh.MFMetricFields()[6].fieldName );
    EXPECT_EQ(16,fh.MFMetricFields()[6].fieldBitSize);
    EXPECT_EQ(MetricFieldName::PKMID_C,fh.MFMetricFields()[7].fieldName );
    EXPECT_EQ(16,fh.MFMetricFields()[7].fieldBitSize);
    EXPECT_EQ(MetricFieldName::PKMID_G,fh.MFMetricFields()[8].fieldName );
    EXPECT_EQ(16,fh.MFMetricFields()[8].fieldBitSize);
    EXPECT_EQ(MetricFieldName::PKMID_T,fh.MFMetricFields()[9].fieldName );
    EXPECT_EQ(16,fh.MFMetricFields()[9].fieldBitSize);
    EXPECT_EQ(MetricFieldName::PKMID_FRAMES_A,fh.MFMetricFields()[10].fieldName );
    EXPECT_EQ(16,fh.MFMetricFields()[10].fieldBitSize);
    EXPECT_EQ(MetricFieldName::PKMID_FRAMES_C,fh.MFMetricFields()[11].fieldName );
    EXPECT_EQ(16,fh.MFMetricFields()[11].fieldBitSize);
    EXPECT_EQ(MetricFieldName::PKMID_FRAMES_G,fh.MFMetricFields()[12].fieldName );
    EXPECT_EQ(16,fh.MFMetricFields()[12].fieldBitSize);
    EXPECT_EQ(MetricFieldName::PKMID_FRAMES_T,fh.MFMetricFields()[13].fieldName );
    EXPECT_EQ(16,fh.MFMetricFields()[13].fieldBitSize);
    EXPECT_EQ(MetricFieldName::PULSE_WIDTH,fh.MFMetricFields()[14].fieldName );
    EXPECT_EQ(16,fh.MFMetricFields()[14].fieldBitSize);
    EXPECT_EQ(MetricFieldName::BASELINE_RED_SD,fh.MFMetricFields()[15].fieldName );
    EXPECT_EQ(16,fh.MFMetricFields()[15].fieldBitSize);
    EXPECT_EQ(MetricFieldName::BASELINE_GREEN_SD,fh.MFMetricFields()[16].fieldName );
    EXPECT_EQ(16,fh.MFMetricFields()[16].fieldBitSize);
    EXPECT_EQ(MetricFieldName::BASELINE_RED_MEAN,fh.MFMetricFields()[17].fieldName );
    EXPECT_EQ(16,fh.MFMetricFields()[17].fieldBitSize);
    EXPECT_EQ(MetricFieldName::BASELINE_GREEN_MEAN,fh.MFMetricFields()[18].fieldName );
    EXPECT_EQ(16,fh.MFMetricFields()[18].fieldBitSize);

    EXPECT_EQ(MetricFieldName::NUM_PULSES_A,fh.MFMetricFields()[19].fieldName );
    EXPECT_EQ(16,fh.MFMetricFields()[19].fieldBitSize);
    EXPECT_EQ(MetricFieldName::NUM_PULSES_C,fh.MFMetricFields()[20].fieldName );
    EXPECT_EQ(16,fh.MFMetricFields()[20].fieldBitSize);
    EXPECT_EQ(MetricFieldName::NUM_PULSES_G,fh.MFMetricFields()[21].fieldName );
    EXPECT_EQ(16,fh.MFMetricFields()[21].fieldBitSize);
    EXPECT_EQ(MetricFieldName::NUM_PULSES_T,fh.MFMetricFields()[22].fieldName );
    EXPECT_EQ(16,fh.MFMetricFields()[22].fieldBitSize);
    EXPECT_EQ(MetricFieldName::ANGLE_A,fh.MFMetricFields()[23].fieldName );
    EXPECT_EQ(16,fh.MFMetricFields()[23].fieldBitSize);
    EXPECT_EQ(MetricFieldName::ANGLE_C,fh.MFMetricFields()[24].fieldName );
    EXPECT_EQ(16,fh.MFMetricFields()[24].fieldBitSize);
    EXPECT_EQ(MetricFieldName::ANGLE_G,fh.MFMetricFields()[25].fieldName );
    EXPECT_EQ(16,fh.MFMetricFields()[25].fieldBitSize);
    EXPECT_EQ(MetricFieldName::ANGLE_T,fh.MFMetricFields()[26].fieldName );
    EXPECT_EQ(16,fh.MFMetricFields()[26].fieldBitSize);
    EXPECT_EQ(MetricFieldName::PKMAX_A,fh.MFMetricFields()[27].fieldName );
    EXPECT_EQ(16,fh.MFMetricFields()[27].fieldBitSize);
    EXPECT_EQ(MetricFieldName::PKMAX_C,fh.MFMetricFields()[28].fieldName );
    EXPECT_EQ(16,fh.MFMetricFields()[28].fieldBitSize);
    EXPECT_EQ(MetricFieldName::PKMAX_G,fh.MFMetricFields()[29].fieldName );
    EXPECT_EQ(16,fh.MFMetricFields()[29].fieldBitSize);
    EXPECT_EQ(MetricFieldName::PKMAX_T,fh.MFMetricFields()[30].fieldName );
    EXPECT_EQ(16,fh.MFMetricFields()[30].fieldBitSize);
    EXPECT_EQ(MetricFieldName::PULSE_DETECTION_SCORE,fh.MFMetricFields()[31].fieldName );
    EXPECT_EQ(16,fh.MFMetricFields()[31].fieldBitSize);

    EXPECT_EQ(6,fh.LFMetricFields().size());
    EXPECT_EQ(MetricFieldName::NUM_BASES,fh.LFMetricFields()[0].fieldName );
    EXPECT_EQ(16,fh.LFMetricFields()[0].fieldBitSize);
    EXPECT_EQ(MetricFieldName::NUM_PULSES,fh.LFMetricFields()[1].fieldName );
    EXPECT_EQ(16,fh.LFMetricFields()[1].fieldBitSize);
    EXPECT_EQ(MetricFieldName::NUM_FRAMES,fh.LFMetricFields()[2].fieldName );
    EXPECT_EQ(16,fh.LFMetricFields()[2].fieldBitSize);
    EXPECT_EQ(MetricFieldName::PULSE_WIDTH,fh.LFMetricFields()[3].fieldName );
    EXPECT_EQ(16,fh.LFMetricFields()[3].fieldBitSize);
    EXPECT_EQ(MetricFieldName::BASE_WIDTH,fh.LFMetricFields()[4].fieldName );
    EXPECT_EQ(16,fh.LFMetricFields()[4].fieldBitSize);

    EXPECT_EQ(17,fh.PacketFields().size());
    EXPECT_EQ(PacketFieldName::READOUT,fh.PacketFields()[0].fieldName );
    EXPECT_EQ(2,fh.PacketFields()[0].fieldBitSize);
    EXPECT_EQ(0,fh.PacketFields()[0].hasFieldEscape);
    EXPECT_EQ(0b11000000,fh.PacketFields()[0].fieldBitMask);
    EXPECT_EQ(6,fh.PacketFields()[0].fieldBitShift);
    EXPECT_EQ(PacketFieldName::DEL_TAG,fh.PacketFields()[1].fieldName );
    EXPECT_EQ(3,fh.PacketFields()[1].fieldBitSize);
    EXPECT_EQ(0,fh.PacketFields()[1].hasFieldEscape);
    EXPECT_EQ(0b00111000,fh.PacketFields()[1].fieldBitMask);
    EXPECT_EQ(3,fh.PacketFields()[1].fieldBitShift);
    EXPECT_EQ(PacketFieldName::SUB_TAG,fh.PacketFields()[2].fieldName );
    EXPECT_EQ(3,fh.PacketFields()[2].fieldBitSize);
    EXPECT_EQ(0,fh.PacketFields()[2].hasFieldEscape);
    EXPECT_EQ(0b00000111,fh.PacketFields()[2].fieldBitMask);
    EXPECT_EQ(0,fh.PacketFields()[2].fieldBitShift);
    EXPECT_EQ(PacketFieldName::DEL_QV,fh.PacketFields()[3].fieldName );
    EXPECT_EQ(4,fh.PacketFields()[3].fieldBitSize);
    EXPECT_EQ(0,fh.PacketFields()[3].hasFieldEscape);
    EXPECT_EQ(0b11110000,fh.PacketFields()[3].fieldBitMask);
    EXPECT_EQ(4,fh.PacketFields()[3].fieldBitShift);
    EXPECT_EQ(PacketFieldName::SUB_QV,fh.PacketFields()[4].fieldName );
    EXPECT_EQ(4,fh.PacketFields()[4].fieldBitSize);
    EXPECT_EQ(0,fh.PacketFields()[4].hasFieldEscape);
    EXPECT_EQ(0b00001111,fh.PacketFields()[4].fieldBitMask);
    EXPECT_EQ(0,fh.PacketFields()[4].fieldBitShift);
    EXPECT_EQ(PacketFieldName::INS_QV,fh.PacketFields()[5].fieldName );
    EXPECT_EQ(4,fh.PacketFields()[5].fieldBitSize);
    EXPECT_EQ(0,fh.PacketFields()[5].hasFieldEscape);
    EXPECT_EQ(0b11110000,fh.PacketFields()[5].fieldBitMask);
    EXPECT_EQ(4,fh.PacketFields()[5].fieldBitShift);
    EXPECT_EQ(PacketFieldName::MRG_QV,fh.PacketFields()[6].fieldName );
    EXPECT_EQ(4,fh.PacketFields()[6].fieldBitSize);
    EXPECT_EQ(0,fh.PacketFields()[6].hasFieldEscape);
    EXPECT_EQ(0b00001111,fh.PacketFields()[6].fieldBitMask);
    EXPECT_EQ(0,fh.PacketFields()[6].fieldBitShift);
    EXPECT_EQ(PacketFieldName::ALT_QV,fh.PacketFields()[7].fieldName );
    EXPECT_EQ(4,fh.PacketFields()[7].fieldBitSize);
    EXPECT_EQ(0,fh.PacketFields()[7].hasFieldEscape);
    EXPECT_EQ(0b11110000,fh.PacketFields()[7].fieldBitMask);
    EXPECT_EQ(4,fh.PacketFields()[7].fieldBitShift);
    EXPECT_EQ(PacketFieldName::LAB_QV,fh.PacketFields()[8].fieldName );
    EXPECT_EQ(4,fh.PacketFields()[8].fieldBitSize);
    EXPECT_EQ(0,fh.PacketFields()[8].hasFieldEscape);
    EXPECT_EQ(0b00001111,fh.PacketFields()[8].fieldBitMask);
    EXPECT_EQ(0,fh.PacketFields()[8].fieldBitShift);
    EXPECT_EQ(PacketFieldName::IS_BASE,fh.PacketFields()[9].fieldName );
    EXPECT_EQ(1,fh.PacketFields()[9].fieldBitSize);
    EXPECT_EQ(0,fh.PacketFields()[9].hasFieldEscape);
    EXPECT_EQ(0b10000000,fh.PacketFields()[9].fieldBitMask);
    EXPECT_EQ(7,fh.PacketFields()[9].fieldBitShift);
    EXPECT_EQ(PacketFieldName::IS_PULSE,fh.PacketFields()[10].fieldName );
    EXPECT_EQ(1,fh.PacketFields()[10].fieldBitSize);
    EXPECT_EQ(0,fh.PacketFields()[10].hasFieldEscape);
    EXPECT_EQ(0b01000000,fh.PacketFields()[10].fieldBitMask);
    EXPECT_EQ(6,fh.PacketFields()[10].fieldBitShift);
    EXPECT_EQ(PacketFieldName::ALT_LABEL,fh.PacketFields()[11].fieldName );
    EXPECT_EQ(3,fh.PacketFields()[11].fieldBitSize);
    EXPECT_EQ(0,fh.PacketFields()[11].hasFieldEscape);
    EXPECT_EQ(0b00111000,fh.PacketFields()[11].fieldBitMask);
    EXPECT_EQ(3,fh.PacketFields()[11].fieldBitShift);
    EXPECT_EQ(PacketFieldName::LABEL,fh.PacketFields()[12].fieldName );
    EXPECT_EQ(3,fh.PacketFields()[12].fieldBitSize);
    EXPECT_EQ(0,fh.PacketFields()[12].hasFieldEscape);
    EXPECT_EQ(0b00000111,fh.PacketFields()[12].fieldBitMask);
    EXPECT_EQ(0,fh.PacketFields()[12].fieldBitShift);
    EXPECT_EQ(PacketFieldName::IPD_LL,fh.PacketFields()[13].fieldName );
    EXPECT_EQ(8,fh.PacketFields()[13].fieldBitSize);
    EXPECT_EQ(1,fh.PacketFields()[13].hasFieldEscape);
    EXPECT_EQ(PacketFieldName::IPD32_LL,fh.PacketFields()[13].extensionName );
    EXPECT_EQ(32,fh.PacketFields()[13].extensionBitSize);
    EXPECT_EQ(PacketFieldName::PW_LL,fh.PacketFields()[14].fieldName );
    EXPECT_EQ(8,fh.PacketFields()[14].fieldBitSize);
    EXPECT_EQ(1,fh.PacketFields()[14].hasFieldEscape);
    EXPECT_EQ(PacketFieldName::PW32_LL,fh.PacketFields()[14].extensionName );
    EXPECT_EQ(32,fh.PacketFields()[14].extensionBitSize);
    EXPECT_EQ(PacketFieldName::PKMEAN_LL,fh.PacketFields()[15].fieldName );
    EXPECT_EQ(8,fh.PacketFields()[15].fieldBitSize);
    EXPECT_EQ(1,fh.PacketFields()[15].hasFieldEscape);
    EXPECT_EQ(PacketFieldName::PKMEAN16_LL,fh.PacketFields()[15].extensionName );
    EXPECT_EQ(16,fh.PacketFields()[15].extensionBitSize);
    EXPECT_EQ(PacketFieldName::PKMID_LL,fh.PacketFields()[16].fieldName );
    EXPECT_EQ(8,fh.PacketFields()[16].fieldBitSize);
    EXPECT_EQ(1,fh.PacketFields()[16].hasFieldEscape);
    EXPECT_EQ(PacketFieldName::PKMID16_LL,fh.PacketFields()[16].extensionName );
    EXPECT_EQ(16,fh.PacketFields()[16].extensionBitSize);
}

class FileHeaderBuilderEx : public FileHeaderBuilder
{
public:
    using FileHeaderBuilder::FileHeaderBuilder;
    using FileHeaderBuilder::HfMetricBlockSize;
    using FileHeaderBuilder::MfMetricBlockSize;
    using FileHeaderBuilder::LfMetricBlockSize;
    using FileHeaderBuilder::OverheadSize;
    using FileHeaderBuilder::HeaderSize;
    using FileHeaderBuilder::PaddingSize;
};

TEST(FileHeaderBuilder, EventsEstimates)
{
    std::vector<uint32_t> zmwNumbers;
    zmwNumbers.push_back(1);

    const float frameRate = 80.0f;
    const uint32_t movieLengthFrames = 50000;

    FileHeaderBuilderEx fhb_bases(
            "/tmp/dummy.baz", //    const std::string& movieName,
            frameRate, //    const float frameRateHz,
            movieLengthFrames, //     const uint32_t movieLengthFrames,
            PacBio::SmrtData::Readout::BASES, //     const SmrtData::Readout readout ,
            PacBio::SmrtData::MetricsVerbosity::NONE, //    const SmrtData::MetricsVerbosity metricsVerbosity,
            EXPERIMENT_METADATA,
            BASECALLER_CONFIG,
            zmwNumbers,
            {},
            1024,4096,16384,
            ChipClass::Sequel,false,false,true,false);

    const uint32_t bytesPerBase = 3;
    EXPECT_FLOAT_EQ(movieLengthFrames/frameRate*bytesPerBase,fhb_bases.EventsByteSize(1.0));


    FileHeaderBuilderEx fhb_pulses(
            "/tmp/dummy.baz", //    const std::string& movieName,
            frameRate, //    const float frameRateHz,
            movieLengthFrames, //     const uint32_t movieLengthFrames,
            PacBio::SmrtData::Readout::PULSES, //     const SmrtData::Readout readout ,
            PacBio::SmrtData::MetricsVerbosity::NONE, //    const SmrtData::MetricsVerbosity metricsVerbosity,
            EXPERIMENT_METADATA,
            BASECALLER_CONFIG,
            zmwNumbers,
            {},
            1024,4096,16384,
            ChipClass::Sequel,false,false,true,false);

    const uint32_t bytesPerPulse = 9;
    // the Event Byte Size is guaranteed to be at least 9 bytes/sec, but will be substantially more
    // in PULSES readout, because of extension bytes.
    EXPECT_LE(movieLengthFrames/frameRate*bytesPerPulse,fhb_pulses.EventsByteSize(1.0));
}


TEST(FileHeaderBuilder, MetricsEstimates)
{
    std::vector<uint32_t> zmwNumbers;
    zmwNumbers.push_back(1);

    // no metrics = 0 bytes size!
    FileHeaderBuilderEx fhb_none(
            "/tmp/dummy.baz", //    const std::string& movieName,
            80.0f, //    const float frameRateHz,
            50000, //     const uint32_t movieLengthFrames,
            PacBio::SmrtData::Readout::BASES, //     const SmrtData::Readout readout ,
            PacBio::SmrtData::MetricsVerbosity::NONE, //    const SmrtData::MetricsVerbosity metricsVerbosity,
            EXPERIMENT_METADATA,
            BASECALLER_CONFIG,
            zmwNumbers,
            {},1024,4096,16384,ChipClass::Sequel,false,false,true,false);
    EXPECT_EQ(0,fhb_none.HfMetricBlockSize());
    EXPECT_EQ(0,fhb_none.MfMetricBlockSize());
    EXPECT_EQ(0,fhb_none.LfMetricBlockSize());
    EXPECT_EQ(0,fhb_none.MetricsByteSize());


    FileHeaderBuilderEx fhb_minimal1(
            "/tmp/dummy.baz", //    const std::string& movieName,
            80.0f, //    const float frameRateHz,
            16384, //     const uint32_t movieLengthFrames,
            PacBio::SmrtData::Readout::BASES, //     const SmrtData::Readout readout ,
            PacBio::SmrtData::MetricsVerbosity::MINIMAL, //    const SmrtData::MetricsVerbosity metricsVerbosity,
            EXPERIMENT_METADATA,
            BASECALLER_CONFIG,
            zmwNumbers,
            {},1024,4096,16384,ChipClass::Sequel,false,false,true,false);
    EXPECT_EQ(6,fhb_minimal1.HfMetricBlockSize());
    EXPECT_EQ(34,fhb_minimal1.MfMetricBlockSize());
    EXPECT_EQ(60,fhb_minimal1.LfMetricBlockSize());
    EXPECT_EQ(60+34*4+6*16,fhb_minimal1.MetricsByteSize());

    zmwNumbers.push_back(2);
    FileHeaderBuilderEx fhb_minimal2(
            "/tmp/dummy.baz", //    const std::string& movieName,
            80.0f, //    const float frameRateHz,
            16384*2, //     const uint32_t movieLengthFrames,
            PacBio::SmrtData::Readout::BASES, //     const SmrtData::Readout readout ,
            PacBio::SmrtData::MetricsVerbosity::MINIMAL, //    const SmrtData::MetricsVerbosity metricsVerbosity,
            EXPERIMENT_METADATA,
            BASECALLER_CONFIG,
            zmwNumbers,
            {},1024,4096,16384,ChipClass::Sequel,false,false,true,false);
    EXPECT_EQ(6,fhb_minimal2.HfMetricBlockSize());
    EXPECT_EQ(34,fhb_minimal2.MfMetricBlockSize());
    EXPECT_EQ(60,fhb_minimal2.LfMetricBlockSize());
    EXPECT_EQ((60+34*4+6*16)*2*2,fhb_minimal2.MetricsByteSize());

    FileHeaderBuilderEx fhb_high(
            "/tmp/dummy.baz", //    const std::string& movieName,
            80.0f, //    const float frameRateHz,
            16384*2, //     const uint32_t movieLengthFrames,
            PacBio::SmrtData::Readout::BASES, //     const SmrtData::Readout readout ,
            PacBio::SmrtData::MetricsVerbosity::HIGH, //    const SmrtData::MetricsVerbosity metricsVerbosity,
            EXPERIMENT_METADATA,
            BASECALLER_CONFIG,
            zmwNumbers,
            {},1024,4096,16384,ChipClass::Sequel,false,false,true,false);
    EXPECT_EQ(6,fhb_high.HfMetricBlockSize());
    EXPECT_EQ(82,fhb_high.MfMetricBlockSize());
    EXPECT_EQ(12,fhb_high.LfMetricBlockSize());
    EXPECT_EQ((12+82*4+6*16)*2*2,fhb_high.MetricsByteSize());
}

TEST(FileHeaderBuilder,OverheadEstimates)
{
    FileHeaderBuilderEx fhb_high(
            "/tmp/dummy.baz", //    const std::string& movieName,
            80.0f, //    const float frameRateHz,
            16384*2, //     const uint32_t movieLengthFrames,
            PacBio::SmrtData::Readout::BASES, //     const SmrtData::Readout readout ,
            PacBio::SmrtData::MetricsVerbosity::HIGH, //    const SmrtData::MetricsVerbosity metricsVerbosity,
            EXPERIMENT_METADATA,
            BASECALLER_CONFIG,
            {1,2},
            {},1024,4096,16384,ChipClass::Sequel,false,false,true,false);
    EXPECT_EQ( 2*(12 + 2*(22+4)),fhb_high.OverheadSize());
}

TEST(FileHeaderBuilder,ExpectedFileByteSize)
{
    FileHeaderBuilderEx fhb_high(
            "/tmp/dummy.baz", //    const std::string& movieName,
            80.0f, //    const float frameRateHz,
            16384*2, //     const uint32_t movieLengthFrames,
            PacBio::SmrtData::Readout::BASES, //     const SmrtData::Readout readout ,
            PacBio::SmrtData::MetricsVerbosity::HIGH, //    const SmrtData::MetricsVerbosity metricsVerbosity,
            EXPERIMENT_METADATA,
            BASECALLER_CONFIG,
            {1,2},
            {},1024,4096,16384,ChipClass::Sequel,false,false,true,false);
    EXPECT_EQ(fhb_high.MetricsByteSize() +
              fhb_high.EventsByteSize(1.0) +
              fhb_high.OverheadSize() +
              fhb_high.HeaderSize() +
              fhb_high.PaddingSize(),
              fhb_high.ExpectedFileByteSize(1.0));
}
