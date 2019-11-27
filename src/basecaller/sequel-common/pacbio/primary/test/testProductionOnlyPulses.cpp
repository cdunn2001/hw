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
using Mb = SequelMetricBlock;

static void write(Basecall& basecall, int& currentFrames)
{
    int ipd = 20;
    basecall.Base(NucleotideLabel::NONE);
    basecall.DeletionTag(NucleotideLabel::G);
    basecall.SubstitutionTag(NucleotideLabel::T);
    basecall.DeletionQV(4);
    basecall.SubstitutionQV(5);
    basecall.InsertionQV(6);
    basecall.GetPulse().IsCrfPulse(false);
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
    int numEvents = 10;
    int currentFrames = 0;
    FileHeaderBuilder fhb= ArminsFakeMovie();

    // LUT
    for (int i = 0; i < numEvents; ++i)
        fhb.AddZmwNumber(100000 + i);
    
    BazWriter<Mb> writer(filename, fhb, PacBio::Primary::BazIOConfig{}, 100000);

    Basecall* basecall = new Basecall[numEvents];
    for (int i = 0; i < numEvents; ++i)
        write(basecall[i], currentFrames);

    auto r = writer.AddZmwSlice(basecall, numEvents, std::vector<Mb>(), 0);
    EXPECT_TRUE(r);
    r = writer.Flush();
    EXPECT_TRUE(r);
    writer.WaitForTermination();
}

TEST(productionOnlyPulses, test)
{
    const std::string FILENAME1 = "prodOnlyPulse_test.baz";
    static PacBio::Dev::TemporaryDirectory tmpdir;
    std::string FILENAME = tmpdir.DirName() + "/" + FILENAME1;

    write(FILENAME);
    BazReader reader(FILENAME);
    EXPECT_TRUE(reader.HasNext());
    // This is actually the terminal SuperChunkMeta
    auto zmws = reader.NextSlice();
    EXPECT_EQ(zmws.size(), static_cast<size_t>(0));
    EXPECT_FALSE(reader.HasNext());
    EXPECT_FALSE(reader.HasNext());
}

