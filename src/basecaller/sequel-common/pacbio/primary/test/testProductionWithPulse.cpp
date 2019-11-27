#include <gtest/gtest.h>

#include <iostream>
#include <memory>
#include <vector>
#include <thread>
#include <chrono>
#include <random>
#include <sstream>
#include <string>
#include <assert.h>

#include <pacbio/logging/Logger.h>

#include <pacbio/dev/TemporaryDirectory.h>
#include <pacbio/smrtdata/Basecall.h>
#include <pacbio/smrtdata/Pulse.h>

#include <pacbio/primary/BazWriter.h>
#include <pacbio/primary/BazReader.h>
#include <pacbio/primary/BazCore.h>
#include <pacbio/primary/SmartMemory.h>
#include <pacbio/primary/Timing.h>
#include <pacbio/primary/PrimaryToBaz.h>

#include "ArminsFakeMovie.h"

using namespace PacBio::Primary;


const std::vector<NucleotideLabel> BASES = {
    NucleotideLabel::NONE, NucleotideLabel::A, NucleotideLabel::C,
    NucleotideLabel::NONE, NucleotideLabel::G, NucleotideLabel::T,
    NucleotideLabel::A,    NucleotideLabel::C, NucleotideLabel::NONE,
    NucleotideLabel::NONE, NucleotideLabel::G, NucleotideLabel::T};
const std::vector<uint32_t> EXPECTED_BASES = {'A','C','G','T','A','C','G','T'};
const std::vector<uint32_t> EXPECTED_IPDS = {20+13+20,20,20+13+20,20,20,20,20+13+20+13+20,20};

static void write(Basecall& basecall, NucleotideLabel base, int& currentFrames)
{
    int ipd = 20;
    basecall.Base(base);
    if (base == NucleotideLabel::NONE)
        basecall.GetPulse().IsCrfPulse(true);
    basecall.DeletionTag(NucleotideLabel::G);
    basecall.SubstitutionTag(NucleotideLabel::T);
    basecall.DeletionQV(4);
    basecall.SubstitutionQV(5);
    basecall.InsertionQV(6);
    basecall.GetPulse().MergeQV(7);
    basecall.GetPulse().AltLabelQV(8);
    basecall.GetPulse().LabelQV(9);
    basecall.GetPulse().AltLabel(NucleotideLabel::G);
    basecall.GetPulse().Label(NucleotideLabel::T);
    basecall.GetPulse().MeanSignal(14);
    basecall.GetPulse().MidSignal(15);
    basecall.GetPulse().Start(currentFrames + ipd);
    basecall.GetPulse().Width(13);
    currentFrames += ipd + basecall.GetPulse().Width();
}

static void write(const std::string& filename)
{
    int currentFrames = 0;
    FileHeaderBuilder fhb = ArminsFakeMovie();
    fhb.MovieLengthFrames(16384); // 1 chunk, instead of 3 hours

    // LUT
    for (int i = 0; i < 1; ++i)
        fhb.AddZmwNumber(100000 + i);

    BazWriter<SequelMetricBlock> writer(filename, fhb, PacBio::Primary::BazIOConfig{}, 100000);

    int numEvents = BASES.size();
    Basecall* basecall = new Basecall[numEvents];
    for (int i = 0; i < numEvents; ++i)
        write(basecall[i], BASES[i], currentFrames);

    auto r = writer.AddZmwSlice(basecall, numEvents, std::vector<SequelMetricBlock>(), 0);
    EXPECT_TRUE(r);
    r = writer.Flush();
    EXPECT_TRUE(r);
    writer.WaitForTermination();

    CheckFileSizes(writer);
}

TEST(productionWithPulse, test)
{
    PacBio::Logging::LogSeverityContext context(PacBio::Logging::LogLevel::WARN);

    const std::string FILENAME1 = "end2endProductionPulse_test.baz";
    static PacBio::Dev::TemporaryDirectory tmpdir;
    std::string FILENAME = tmpdir.DirName() + "/" + FILENAME1;

    write(FILENAME);
    BazReader reader(FILENAME);
    EXPECT_TRUE(reader.HasNext());
    auto zmwData = reader.NextSlice();
    EXPECT_EQ(zmwData.size(), static_cast<size_t>(1));
    auto& data = zmwData[0];
    const auto& packets = ParsePackets(reader.Fileheader(), data);

    EXPECT_EQ(packets.PacketField(PacketFieldName::READOUT), EXPECTED_BASES);

    auto ipds = packets.PacketField(PacketFieldName::IPD_V1);

    // Need to reverse the encoding to get back actual ipds
    Codec codec;
    for (auto& val : ipds) val = codec.CodeToFrame(static_cast<uint8_t>(val));

    EXPECT_EQ(ipds, EXPECTED_IPDS);
}

