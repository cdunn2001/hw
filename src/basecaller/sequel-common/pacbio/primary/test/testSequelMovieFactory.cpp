#include <pacbio/primary/Tile.h>
#include <pacbio/primary/SequelMovieFactory.h>
#include <pacbio/primary/ChipLayoutRTO3.h>
#include <pacbio/primary/SequelMovieFileHDF5.h>
#include <pacbio/primary/SequelTraceFile.h>
#include <pacbio/dev/gtest-extras.h>
#include "gtest/gtest.h"
#include "SequelTestROIs.h"

using namespace std;
using namespace PacBio::Primary;

class SequelMovieFactory_Test : public ::testing::Test
{
    PacBio::Logging::LogSeverityContext loggerContext;

public:
    TempFileManager temps;
    SequelMovieFactory_Test() : loggerContext(PacBio::Logging::LogLevel::FATAL) {}
    ~SequelMovieFactory_Test()
    {
        if (HasFailure()) temps.Keep();
    }
    static ChipLayoutRTO3 chipLayout;
};
ChipLayoutRTO3 SequelMovieFactory_Test::chipLayout;

TEST_F(SequelMovieFactory_Test,Frame)
{
    SequelMovieConfig smconfig;
    smconfig.path = temps.GenerateTempFileName(".h5");
    smconfig.roi.reset(chipLayout.GetFullChipROI().Clone());
    smconfig.chipClass = chipLayout.GetChipClass();
    smconfig.numFrames = 1;
    smconfig.movie.chunking.col = 32;
    smconfig.movie.chunking.row = 364;
    smconfig.movie.chunking.frame = 512;
//    smconfig.movie.compression = acqConfig_.init.moviehdf5.gzipCompressionLevel();

    {
        auto x = SequelMovieFactory::CreateOutput(smconfig);
        EXPECT_EQ(x->TypeName(), "Frame");
        auto y = dynamic_cast<SequelMovieFileHDF5*>(x.get());
        ASSERT_TRUE(y != nullptr);
        EXPECT_EQ(smconfig.movie.chunking.col, y->GetChunking().col);

        smconfig.path = "/dev/null";
        auto z = SequelMovieFactory::CreateOutput(smconfig);
        EXPECT_EQ(z->TypeName(), "Frame");
    }
}

TEST_F(SequelMovieFactory_Test,Trace)
{
    PacBio::Logging::LogSeverityContext _(PacBio::Logging::LogLevel::ERROR);

    SequelMovieConfig smconfig;
    smconfig.path = temps.GenerateTempFileName(".trc.h5");
    smconfig.roi.reset(chipLayout.GetFullChipROI().Clone());
    smconfig.chipClass = chipLayout.GetChipClass();
    smconfig.numFrames = 512;
//    smconfig.movie.chunking = movieChunking;
//    smconfig.movie.compression = acqConfig_.init.moviehdf5.gzipCompressionLevel();

    {
        auto x = SequelMovieFactory::CreateOutput(smconfig);
        EXPECT_EQ(x->TypeName(), "Trace");
        auto y = dynamic_cast<SequelTraceFileHDF5*>(x.get());
        ASSERT_TRUE(y != nullptr);
    }
}

TEST_F(SequelMovieFactory_Test,CRC)
{
    SequelMovieConfig smconfig;
    smconfig.path = temps.GenerateTempFileName(".crc");
    smconfig.roi.reset(chipLayout.GetFullChipROI().Clone());
    smconfig.chipClass = chipLayout.GetChipClass();
    smconfig.numFrames = 512;

    auto x = SequelMovieFactory::CreateOutput(smconfig);

    EXPECT_EQ(x->TypeName(), "CRC");
}

TEST_F(SequelMovieFactory_Test,EstimatedSize)
{
    SequelRectangularROI roi(RowPixels(0), ColPixels(0), RowPixels(1), ColPixels(2048),SequelSensorROI::SequelAlpha());

    EXPECT_EQ(14438891520, SequelMovieFactory::EstimatedSize("foo.trc.h5",roi,80*3600*12));
    EXPECT_EQ(14438891520, SequelMovieFactory::EstimatedSize("foo.mov.h5",roi,80*3600*12));
}




