#include <gtest/gtest.h>

#include <pacbio/dev/TemporaryDirectory.h>
#include <pacbio/primary/BazWriter.h>

#include "ArminsFakeMovie.h"

using namespace PacBio::Primary;


TEST(emptyChunk, test)
{
    FileHeaderBuilder fhb = ArminsFakeMovie();
    fhb.AddZmwNumber(0);
    fhb.MovieLengthFrames(1); // minimal movie length

    const std::string FILENAME1 = "empty.baz";
    PacBio::Dev::TemporaryDirectory tmpdir;
    std::string FILENAME = tmpdir.DirName() + "/" + FILENAME1;
    BazWriter<SequelMetricBlock> writer(FILENAME, fhb, PacBio::Primary::BazIOConfig{}, 1000000);
    writer.WaitForTermination();
    EXPECT_FALSE(writer.Flush());

    CheckFileSizes(writer);
}
