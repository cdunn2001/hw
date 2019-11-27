#include <gtest/gtest.h>

#include <assert.h>
#include <iostream>
#include <memory>
#include <string>
#include <stdio.h>
#include <string.h>
#include "json/json.h"
#include "json/reader.h"

#include <pacbio/dev/TemporaryDirectory.h>
#include <pacbio/smrtdata/Basecall.h>
#include <pacbio/smrtdata/Pulse.h>
#include <pacbio/logging/Logger.h>

#include <pacbio/primary/BazWriter.h>
#include <pacbio/primary/FileHeader.h>
#include <pacbio/primary/FileHeaderValidator.h>
#include <pacbio/primary/PrimaryToBaz.h>
#include <pacbio/primary/SmartMemory.h>
#include <pacbio/primary/Simulation.h>
#include <pacbio/primary/Timing.h>
#include <pacbio/primary/PacketFieldName.h>
#include <pacbio/primary/Sanity.h>


using namespace PacBio::Primary;
using namespace PacBio::Logging;


// Original header
static std::string JSON_HEADER =
    "{\"TYPE\":\"BAZ\", \"HEADER\":"
    "{\"MOVIE_NAME\":\"ArminsFakeMovie\",\"COMPLETE\":1,\"BASE_CALLER_VERSION\":\"1.2\",\"BAZWRITER_VERSION\":\"1.0\",\"BAZ2BAM_VERSION\":\"3.3\",\"BAZ_MAJOR_VERSION\" : 0,\"BAZ_MINOR_VERSION\" : "
    "1,\"BAZ_PATCH_VERSION\" : 0,\"FRAME_RATE_HZ\" : 100,\"MOVIE_LENGTH_FRAMES\" : 1080000,"
    "\"BASECALLER_CONFIG\": \"{}\","
    "\"EXPERIMENT_METADATA\" : \"{\\\"AcqParams\\\":{\\\"AduGain\\\":1.9531248807907104},\\\"ChipInfo\\\":{\\\"AnalogRefSnr\\\":11,\\\"AnalogRefSpectrum\\\":[0.10197792202234268,0.8980221152305603],\\\"CrosstalkFilter\\\":[[0.00011132592771900818,0.00043121803901158273,0.00081435841275379062,0.0010630703764036298,0.00087830686243250966,0.00046748583554290235,0.00012474130198825151],[0.00040487214573659003,-0.00203699991106987,-0.003298585768789053,-0.0051095252856612206,-0.0040731276385486126,-0.0018567438237369061,0.00044959798105992377],[0.0008288127719424665,-0.0025635270867496729,-0.030308688059449196,-0.092926815152168274,-0.030468275770545006,-0.0043287212029099464,0.00080724310828372836],[0.0012212963774800301,-0.011045822873711586,-0.076042637228965759,1.5124297142028809,-0.085587434470653534,-0.0050257253460586071,0.00087140261894091964],[0.00093532569007948041,-0.0050715520046651363,-0.027508752420544624,-0.095789402723312378,-0.027349235489964485,-0.00034580400097183883,0.0005108729237690568],[0.00046027876669541001,-0.0018478382844477892,-0.0040882616303861141,-0.0075986571609973907,-0.0033990705851465464,-0.00060037523508071899,0.00022889320098329335],[0.00012168431567260996,0.00047721769078634679,0.00092700554523617029,0.0012106377398595214,0.00080913538113236427,0.00030304939718917012,5.5799839174142107e-05]],\\\"FilterMap\\\":[1,0],\\\"ImagePsf\\\":[[[-0.0010000000474974513,0.0038000000640749931,0.0066999997943639755,0.0038000000640749931,0.0019000000320374966],[-0.0010000000474974513,0.017200000584125519,0.061999998986721039,0.015300000086426735,0.0038000000640749931],[0.0010000000474974513,0.047699999064207077,0.66920000314712524,0.037200000137090683,0.0076000001281499863],[0.002899999963119626,0.021900000050663948,0.06289999932050705,0.019099999219179153,0.0010000000474974513],[0.0019000000320374966,0.0048000002279877663,0.0048000002279877663,0.0038000000640749931,0.0019000000320374966]],[[0.0030000000260770321,0.0040000001899898052,0.0080000003799796104,0.004999999888241291,0.0020000000949949026],[0.004999999888241291,0.023000000044703484,0.054900001734495163,0.02500000037252903,0.0060000000521540642],[0.0099999997764825821,0.057900000363588333,0.60039997100830078,0.057900000363588333,0.0099999997764825821],[0.0060000000521540642,0.02199999988079071,0.05090000107884407,0.024000000208616257,0.0060000000521540642],[0.0020000000949949026,0.0040000001899898052,0.0070000002160668373,0.0040000001899898052,0.0020000000949949026]]],\\\"LayoutName\\\":\\\"SequEL_4.0_RTO3\\\"},\\\"DyeSet\\\":{\\\"AnalogSpectra\\\":[[-0.052940212190151215,1.0529402494430542],[-0.052940212190151215,1.0529402494430542],[-0.052940212190151215,1.0529402494430542],[-0.052940212190151215,1.0529402494430542]],\\\"BaseMap\\\":\\\"CTAG\\\",\\\"ExcessNoiseCV\\\":[0.10000000149011612,0.10000000149011612,0.10000000149011612,0.10000000149011612],\\\"IpdMean\\\":[0.30656999349594116,0.32460001111030579,0.28850001096725464,0.30656999349594116],\\\"NumAnalogs\\\":4,\\\"PulseWidthMean\\\":[0.21639999747276306,0.14427000284194946,0.18029999732971191,0.21639999747276306],\\\"RelativeAmp\\\":[1,0.63636398315429688,0.43181800842285156,0.22727300226688385]},\\\"basecallerVersion\\\":\\\"?\\\",\\\"bazfile\\\":\\\"\\\",\\\"cameraConfigKey\\\":0,\\\"chipId\\\":\\\"n/a\\\",\\\"dryrun\\\":false,\\\"expectedFrameRate\\\":100,\\\"exposure\\\":0.01,\\\"hdf5output\\\":\\\"\\\",\\\"instrumentName\\\":\\\"n/a\\\",\\\"metricsVerbosity\\\":\\\"MINIMAL\\\",\\\"minSnr\\\":4,\\\"movieContext\\\":\\\"\\\",\\\"noCalibration\\\":false,\\\"numFrames\\\":0,\\\"numPixelLanes\\\":[],\\\"numZmwLanes\\\":[],\\\"numZmws\\\":1171456,\\\"photoelectronSensitivity\\\":1,\\\"readout\\\":\\\"BASES\\\",\\\"refDwsSnr\\\":11,\\\"refSpectrum\\\":[0.10197792202234268,0.8980221152305603],\\\"remapVroiOption\\\":false,\\\"roiMetaData\\\":null}\","
    "\"LF_METRIC\" : {\"FIELDS\" : [[ \"BASELINE_RED_SD\", 8 ], "
    "[\"PULSE_WIDTH\", 16], [\"BASE_WIDTH\", 16], [ "
    "\"BASELINE_GREEN_SD\", 8 ]"
    ",[\"BASELINE_RED_MEAN\",16]"
    ",[\"BASELINE_GREEN_MEAN\",16]"
    ",[ "
    "\"NUM_PULSES\", 16 ],[ \"NUM_BASES\", 16 ]],\"FRAMES\" : "
    "16384},\"MF_METRIC\" : {\"FIELDS\" : [[ \"PKMID_A\", 8 ],[ "
    "\"PKMID_C\", 8 ],[ \"PKMID_G\", 8 ],[ \"PKMID_T\", 8 ]],\"FRAMES\" : "
    "4096},\"PACKET\" : [[ \"READOUT\", 2 ],[ \"DEL_TAG\", 3 ],[ "
    "\"SUB_TAG\", 3 ],[ \"DEL_QV\", 4 ],[ \"SUB_QV\", 4 ],[ \"INS_QV\", 4 "
    "],[ \"MRG_QV\", 4 ],[ \"IPD_LL\", 8, 255, \"IPD16_LL\", 16 "
    "]],\"SLICE_LENGTH_FRAMES\" : 16384}}";

    // "{\"BAZ_MAJOR_VERSION\" : 0,\"BAZ_MINOR_VERSION\" : "
    // "1,\"BAZ_PATCH_VERSION\" : 0,\"FRAME_RATE_HZ\" : 100,\"HF_METRIC\" : "
    // "{\"FIELDS\" : [[ \"PR_0\", 8 ],[ \"PR_1\", 8 ]],\"FRAMES\" : "
    // "512},\"LF_METRIC\" : {\"FIELDS\" : [[ \"BASELINE_LEVEL_RED\", 8 ],[ "
    // "\"BASELINE_LEVEL_GREEN\", 8 ],[ \"BASELINE_STANDARD_STD\", 16 ],[ "
    // "\"NUM_PULSES\", 16 ],[ \"NUM_BASES\", 16 ]],\"FRAMES\" : "
    // "16384},\"MF_METRIC\" : {\"FIELDS\" : [[ \"PKMID_A\", 8 ],[ "
    // "\"PKMID_C\", 8 ],[ \"PKMID_G\", 8 ],[ \"PKMID_T\", 8 ]],\"FRAMES\" : "
    // "4096},\"PACKET\" : [[ \"READOUT\", 2 ],[ \"DEL_TAG\", 3 ],[ "
    // "\"SUB_TAG\", 3 ],[ \"DEL_QV\", 4 ],[ \"SUB_QV\", 4 ],[ \"INS_QV\", 4 "
    // "],[ \"MRG_QV\", 4 ],[ \"IPD_LL\", 8, 255, \"IPD16_LL\", 16 "
    // "]],\"SLICE_LENGTH_FRAMES\" : 16384}";

static void Write(const std::string& filename)
{
    
    // Open new file
    auto file = SmartMemory::OpenFile(filename.c_str(), "wb");
    // Write header
    std::fwrite(JSON_HEADER.c_str(), JSON_HEADER.size(), 1, file.get());
    // Write SANITY block to find end of header
    Sanity::Write(file);
    // 4k (yes, 4000 bytes, not 4096) alignment
    uint64_t curPtr = std::ftell(file.get());
    uint64_t blockSize = 4000;
    if (curPtr % blockSize != 0)
        curPtr += blockSize - curPtr % blockSize;

    fseek(file.get(), curPtr, SEEK_SET);
    Sanity::Write(file);
}

static FileHeader Read(const std::string& filename)
{
    auto file = SmartMemory::OpenFile(filename.c_str(), "rb");

    // Seek for SANITY block and last position of the file header
    uint64_t headerSize = Sanity::FindAndVerify(file);
    EXPECT_EQ(headerSize, static_cast<uint64_t>(JSON_HEADER.size()));
    // Wrap header into smrt pointer
    std::vector<char> header(headerSize + 1);

    // Set position indicator to beginning
    std::rewind(file.get());

    // Read file header
    size_t result = std::fread(header.data(), 1, headerSize, file.get());
    if (result != headerSize)
        throw std::runtime_error("Cannot read file header!");

    return FileHeader(header.data(), headerSize);;
}

TEST(fileheader, NoData)
{
    const std::string FILENAME1 = "fileheader_test.baz";
    static PacBio::Dev::TemporaryDirectory tmpdir;
    std::string FILENAME = tmpdir.DirName() + "/" + FILENAME1;

    Write(FILENAME);
    auto fh = Read(FILENAME);
 
    EXPECT_EQ(fh.MovieName(), "ArminsFakeMovie");
    EXPECT_EQ(fh.BazMajorVersion(), static_cast<uint32_t>(0));
    EXPECT_EQ(fh.BazMinorVersion(), static_cast<uint32_t>(1));
    EXPECT_EQ(fh.BazPatchVersion(), static_cast<uint32_t>(0));
    EXPECT_EQ(fh.FrameRateHz(), static_cast<uint32_t>(100));
    EXPECT_EQ(fh.HFMetricFrames(), static_cast<uint32_t>(0));  // TODO: Armin, is this understood as "no HF metrics"?
    EXPECT_EQ(fh.MFMetricFrames(), static_cast<uint32_t>(4096));
    EXPECT_EQ(fh.LFMetricFrames(), static_cast<uint32_t>(16384));
    EXPECT_EQ(fh.SliceLengthFrames(), static_cast<uint32_t>(16384));
    EXPECT_EQ(fh.HFMetricByteSize(), static_cast<uint32_t>(0));
    EXPECT_EQ(fh.MFMetricByteSize(), static_cast<uint32_t>(4));
    EXPECT_EQ(fh.LFMetricByteSize(), static_cast<uint32_t>(14));
    EXPECT_EQ(fh.MovieLengthFrames(), static_cast<uint32_t>(100*60*60*3));

    EXPECT_EQ(fh.HFMetricFields().size(), static_cast<uint32_t>(0));

    EXPECT_EQ(fh.MFMetricFields().size(), static_cast<uint32_t>(4));
    EXPECT_EQ(fh.MFMetricFields()[0].fieldName == MetricFieldName::PKMID_A, static_cast<uint32_t>(1));
    EXPECT_EQ(fh.MFMetricFields()[0].fieldBitSize, static_cast<uint32_t>(8));
    EXPECT_EQ(fh.MFMetricFields()[1].fieldName == MetricFieldName::PKMID_C, static_cast<uint32_t>(1));
    EXPECT_EQ(fh.MFMetricFields()[1].fieldBitSize, static_cast<uint32_t>(8));
    EXPECT_EQ(fh.MFMetricFields()[2].fieldName == MetricFieldName::PKMID_G, static_cast<uint32_t>(1));
    EXPECT_EQ(fh.MFMetricFields()[2].fieldBitSize, static_cast<uint32_t>(8));
    EXPECT_EQ(fh.MFMetricFields()[3].fieldName == MetricFieldName::PKMID_T, static_cast<uint32_t>(1));
    EXPECT_EQ(fh.MFMetricFields()[3].fieldBitSize, static_cast<uint32_t>(8));

    EXPECT_EQ(fh.LFMetricFields().size(), static_cast<uint32_t>(8));
    EXPECT_EQ(fh.LFMetricFields()[0].fieldName == MetricFieldName::BASELINE_RED_SD, static_cast<uint32_t>(1));
    EXPECT_EQ(fh.LFMetricFields()[0].fieldBitSize, static_cast<uint32_t>(8));
    EXPECT_EQ(fh.LFMetricFields()[1].fieldName == MetricFieldName::PULSE_WIDTH, static_cast<uint32_t>(1));
    EXPECT_EQ(fh.LFMetricFields()[1].fieldBitSize, static_cast<uint32_t>(16));
    EXPECT_EQ(fh.LFMetricFields()[2].fieldName == MetricFieldName::BASE_WIDTH, static_cast<uint32_t>(1));
    EXPECT_EQ(fh.LFMetricFields()[2].fieldBitSize, static_cast<uint32_t>(16));
    EXPECT_EQ(fh.LFMetricFields()[3].fieldName == MetricFieldName::BASELINE_GREEN_SD, static_cast<uint32_t>(1));
    EXPECT_EQ(fh.LFMetricFields()[3].fieldBitSize, static_cast<uint32_t>(8));
    EXPECT_EQ(fh.LFMetricFields()[4].fieldName == MetricFieldName::BASELINE_RED_MEAN, static_cast<uint32_t>(1));
    EXPECT_EQ(fh.LFMetricFields()[4].fieldBitSize, static_cast<uint32_t>(16));
    EXPECT_EQ(fh.LFMetricFields()[5].fieldName == MetricFieldName::BASELINE_GREEN_MEAN, static_cast<uint32_t>(1));
    EXPECT_EQ(fh.LFMetricFields()[5].fieldBitSize, static_cast<uint32_t>(16));
    EXPECT_EQ(fh.LFMetricFields()[6].fieldName == MetricFieldName::NUM_PULSES, static_cast<uint32_t>(1));
    EXPECT_EQ(fh.LFMetricFields()[6].fieldBitSize, static_cast<uint32_t>(16));
    EXPECT_EQ(fh.LFMetricFields()[7].fieldName == MetricFieldName::NUM_BASES, static_cast<uint32_t>(1));
    EXPECT_EQ(fh.LFMetricFields()[7].fieldBitSize, static_cast<uint32_t>(16));

    EXPECT_EQ(fh.PacketFields().size(), static_cast<uint32_t>(8));
    EXPECT_EQ(fh.PacketFields()[0].fieldName == PacketFieldName::READOUT, static_cast<uint32_t>(1));
    EXPECT_EQ(fh.PacketFields()[0].fieldBitSize, static_cast<uint32_t>(2));
    EXPECT_EQ(fh.PacketFields()[0].hasFieldEscape, static_cast<uint32_t>(0));
    EXPECT_EQ(fh.PacketFields()[0].fieldBitMask, static_cast<uint32_t>(0b11000000));
    EXPECT_EQ(fh.PacketFields()[0].fieldBitShift, static_cast<uint32_t>(6));
    EXPECT_EQ(fh.PacketFields()[1].fieldName == PacketFieldName::DEL_TAG, static_cast<uint32_t>(1));
    EXPECT_EQ(fh.PacketFields()[1].fieldBitSize, static_cast<uint32_t>(3));
    EXPECT_EQ(fh.PacketFields()[1].hasFieldEscape, static_cast<uint32_t>(0));
    EXPECT_EQ(fh.PacketFields()[1].fieldBitMask, static_cast<uint32_t>(0b00111000));
    EXPECT_EQ(fh.PacketFields()[1].fieldBitShift, static_cast<uint32_t>(3));
    EXPECT_EQ(fh.PacketFields()[2].fieldName == PacketFieldName::SUB_TAG, static_cast<uint32_t>(1));
    EXPECT_EQ(fh.PacketFields()[2].fieldBitSize, static_cast<uint32_t>(3));
    EXPECT_EQ(fh.PacketFields()[2].hasFieldEscape, static_cast<uint32_t>(0));
    EXPECT_EQ(fh.PacketFields()[2].fieldBitMask, static_cast<uint32_t>(0b00000111));
    EXPECT_EQ(fh.PacketFields()[2].fieldBitShift, static_cast<uint32_t>(0));
    EXPECT_EQ(fh.PacketFields()[3].fieldName == PacketFieldName::DEL_QV, static_cast<uint32_t>(1));
    EXPECT_EQ(fh.PacketFields()[3].fieldBitSize, static_cast<uint32_t>(4));
    EXPECT_EQ(fh.PacketFields()[3].hasFieldEscape, static_cast<uint32_t>(0));
    EXPECT_EQ(fh.PacketFields()[3].fieldBitMask, static_cast<uint32_t>(0b11110000));
    EXPECT_EQ(fh.PacketFields()[3].fieldBitShift, static_cast<uint32_t>(4));
    EXPECT_EQ(fh.PacketFields()[4].fieldName == PacketFieldName::SUB_QV, static_cast<uint32_t>(1));
    EXPECT_EQ(fh.PacketFields()[4].fieldBitSize, static_cast<uint32_t>(4));
    EXPECT_EQ(fh.PacketFields()[4].hasFieldEscape, static_cast<uint32_t>(0));
    EXPECT_EQ(fh.PacketFields()[4].fieldBitMask, static_cast<uint32_t>(0b00001111));
    EXPECT_EQ(fh.PacketFields()[4].fieldBitShift, static_cast<uint32_t>(0));
    EXPECT_EQ(fh.PacketFields()[5].fieldName == PacketFieldName::INS_QV, static_cast<uint32_t>(1));
    EXPECT_EQ(fh.PacketFields()[5].fieldBitSize, static_cast<uint32_t>(4));
    EXPECT_EQ(fh.PacketFields()[5].hasFieldEscape, static_cast<uint32_t>(0));
    EXPECT_EQ(fh.PacketFields()[5].fieldBitMask, static_cast<uint32_t>(0b11110000));
    EXPECT_EQ(fh.PacketFields()[5].fieldBitShift, static_cast<uint32_t>(4));
    EXPECT_EQ(fh.PacketFields()[6].fieldName == PacketFieldName::MRG_QV, static_cast<uint32_t>(1));
    EXPECT_EQ(fh.PacketFields()[6].fieldBitSize, static_cast<uint32_t>(4));
    EXPECT_EQ(fh.PacketFields()[6].hasFieldEscape, static_cast<uint32_t>(0));
    EXPECT_EQ(fh.PacketFields()[6].fieldBitMask, static_cast<uint32_t>(0b00001111));
    EXPECT_EQ(fh.PacketFields()[6].fieldBitShift, static_cast<uint32_t>(0));
    EXPECT_EQ(fh.PacketFields()[7].fieldName == PacketFieldName::IPD_LL, static_cast<uint32_t>(1));
    EXPECT_EQ(fh.PacketFields()[7].fieldBitSize, static_cast<uint32_t>(8));
    EXPECT_EQ(fh.PacketFields()[7].hasFieldEscape, static_cast<uint32_t>(1));
    EXPECT_EQ(fh.PacketFields()[7].extensionName == PacketFieldName::IPD16_LL, static_cast<uint32_t>(1));
    EXPECT_EQ(fh.PacketFields()[7].extensionBitSize, static_cast<uint32_t>(16));

}

TEST(FileHeader, ExperimentMetadata)
{
    LogSeverityContext _(LogLevel::WARN);
    // Hard to test log output, but you can observe the log messages while this
    // runs to make sure they are in the desired order
    std::string defaultMetadata = generateExperimentMetadata();
    Json::Value expected;
    Json::Reader{}.parse(defaultMetadata, expected);

    // Sanity check that the default isn't nonsense:
    EXPECT_EQ("SequEL_4.0_RTO3",
              expected["ChipInfo"]["LayoutName"].asString());

    // missing
    EXPECT_EQ(parseExperimentMetadata(""), expected);
    // empty
    EXPECT_EQ(parseExperimentMetadata("{}"), expected);
    // unparsable
    EXPECT_EQ(parseExperimentMetadata("{\"foo\": 3, \"bar\"}"), expected);
    // incomplete
    EXPECT_EQ(parseExperimentMetadata("{\"foo\": 3, \"bar\": 4}"), expected);
    // complete
    EXPECT_EQ(parseExperimentMetadata(defaultMetadata), expected);

    // Check that it isn't just always returning the default, by modifying the
    // input and reparsing:
    expected["ChipInfo"]["ChipLayoutName"] = "potato";
    Json::Value observed = parseExperimentMetadata(
        Json::writeString(Json::StreamWriterBuilder{}, expected));
    EXPECT_EQ(observed, expected);
    EXPECT_EQ(observed["ChipInfo"]["ChipLayoutName"], "potato");
}

