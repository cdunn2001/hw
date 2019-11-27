#include <cmath>

#include <gtest/gtest.h>
#include <pacbio/dev/gtest-extras.h>
#include <pacbio/dev/AutoTimer.h>

#include <pacbio/primary/ZmwStatsFileData.h>
#include <pacbio/primary/ZmwStatsFile.h>

using namespace PacBio::Primary;
using namespace PacBio::Dev;

uint32_t GetNumZmws()
{
    if (getenv("NUMZMWS"))
    {
        return atoi(getenv("NUMZMWS"));
    } else {
        return 1300;
    }
}
uint32_t NumZmws = GetNumZmws();

TEST(ZmwStats,Constructor)
{
    ZmwStats zs(1,1,1);
    zs.Init();
    EXPECT_EQ(std::numeric_limits<uint32_t>::max() ,zs.NumBases);  // fixme. the Init is broken
    EXPECT_TRUE(std::isnan(zs.BaseFraction[0]));
    zs.NumBases = 5;
    EXPECT_EQ(5,zs.NumBases);
}

/// This is not so useful for the end user, but it is testing an internal
/// feature of the ZmwStatsFile
TEST(ZmwStatDataSet,Constructor)
{
    hsize_t dim0 = 1;
    hsize_t dim1 = 2;
    hsize_t dim2 = 3;
    ZmwStatDataSet zsds1("foo",float32(),"nothung","This is the description",true,dim0);
    EXPECT_FALSE(zsds1.IsTemporal());
    ZmwStatDataSet zsds2("foo",float32(),"nothung","This is the description",true,"nA_",dim0,dim1);
    EXPECT_FALSE(zsds2.IsTemporal());
    ZmwStatDataSet zsds3("foo",float32(),"nothung","This is the description",true,"nMF_",dim0,dim1,dim2);
    EXPECT_TRUE(zsds3.IsTemporal());
}

void Fill(ZmwStats& zmw, hsize_t i)
{
    zmw.index_ = i;
    zmw.NumBases = i;
    zmw.ReadLength = 5;
    zmw.Productivity = ZmwStatsFile::Productivity_t::Productive;
    zmw.MedianInsertLength = 6;
    zmw.InsertReadLength = 7;
    zmw.ReadType =  ZmwStatsFile::ReadType_t::FullHqRead0;
    zmw.ReadScore = 0.5;
    zmw.HQRegionStart = 0;
    zmw.HQRegionEnd = 100;
    zmw.BaseFraction[0] = 0.15;
    zmw.BaseFraction[1] = 0.25;
    zmw.BaseFraction[2] = 0.35;
    zmw.BaseFraction[3] = 0.45;
    zmw.BaselineLevel[0] = 100.0;
    zmw.BaselineLevel[1] = 200.0;
    ///zmw.ClusterDistance[0][0] = 1.0;
    ///zmw.ClusterDistance[0][1] = 1.0;
    ///zmw.ClusterDistance[0][2] = 2.0;
    ///zmw.ClusterDistance[0][3] = 3.0;
    ///zmw.ClusterDistance[0][4] = 4.0;

    for (uint32_t t = 0; t < zmw.nMF_; t++)
    {
        zmw.VsMFTraceAutoCorr[t] = static_cast<float>(4.5 * t);
    }

    zmw.HoleNumber = 1000+i;
    zmw.HoleXY[0] = i / 100;
    zmw.HoleXY[1] = i % 100;
    zmw.UnitFeature = 5;
}

void Validate(std::string filename, const ZmwStatsFile::NewFileConfig& config)
{
    PBLOG_DEBUG << "Opening " << filename;
    ZmwStatsFile file(filename);

    EXPECT_EQ(config.numHoles, file.nH());
    EXPECT_EQ(config.numAnalogs, file.nA());
    EXPECT_EQ(config.numFilters, file.nF());

    // These are disabled as the VsT datasets have been removed.
    //EXPECT_EQ(config.binSize, file.BinSize());
    //EXPECT_EQ(config.NumTimeslices(), file.nT());

    EXPECT_EQ("Number of called bases", file.NumBases.Description());
    EXPECT_FALSE(file.NumBases.HQRegion());
    EXPECT_TRUE(file.ReadLength.HQRegion());
    EXPECT_TRUE(file.LocalBaseRate.HQRegion());
    EXPECT_TRUE(file.NumPulses.HQRegion());
    EXPECT_EQ("bases", file.NumBases.UnitsOrEncoding());
    EXPECT_EQ(16384, file.NumBases.BinSize());

    for (hsize_t i = 0; i < file.nH(); i+= 7)
    {
        ZmwStats zmw = file.Get(i);
        EXPECT_EQ(i, zmw.NumBases);

        EXPECT_EQ(5, zmw.ReadLength);
        EXPECT_EQ(ZmwStatsFile::Productivity_t::Productive, zmw.Productivity);
        EXPECT_EQ(6, zmw.MedianInsertLength);
        EXPECT_EQ(7, zmw.InsertReadLength);
        EXPECT_EQ(ZmwStatsFile::ReadType_t::FullHqRead0, zmw.ReadType);
        EXPECT_FLOAT_EQ(0.5, zmw.ReadScore);
        EXPECT_EQ(0, zmw.HQRegionStart);
        EXPECT_EQ(100, zmw.HQRegionEnd);
        EXPECT_FLOAT_EQ(0.15, zmw.BaseFraction[0]);
        EXPECT_FLOAT_EQ(0.25, zmw.BaseFraction[1]);
        EXPECT_FLOAT_EQ(0.35, zmw.BaseFraction[2]);
        EXPECT_FLOAT_EQ(0.45, zmw.BaseFraction[3]);
        EXPECT_FLOAT_EQ(100.0, zmw.BaselineLevel[0]);
        EXPECT_FLOAT_EQ(200.0, zmw.BaselineLevel[1]);
        ///EXPECT_FLOAT_EQ(1.0, zmw.ClusterDistance[0][0]);
        ///EXPECT_FLOAT_EQ(1.0, zmw.ClusterDistance[0][1]);
        ///EXPECT_FLOAT_EQ(2.0, zmw.ClusterDistance[0][2]);
        ///EXPECT_FLOAT_EQ(3.0, zmw.ClusterDistance[0][3]);
        ///EXPECT_FLOAT_EQ(4.0, zmw.ClusterDistance[0][4]);

        EXPECT_EQ(i+1000, zmw.HoleNumber);
        EXPECT_EQ(i/100 , zmw.HoleXY[0]);
        EXPECT_EQ(i%100 , zmw.HoleXY[1]);
        EXPECT_EQ(5, zmw.UnitFeature);
    }

    EXPECT_EQ( 0, file.GetCoordinate(0).first);
    EXPECT_EQ( 0, file.GetCoordinate(0).second);
    EXPECT_EQ( 0, file.GetCoordinate(1).first);
    EXPECT_EQ( 1, file.GetCoordinate(1).second);
    EXPECT_EQ( 1, file.GetCoordinate(100).first);
    EXPECT_EQ( 0, file.GetCoordinate(100).second);
}



TEST(ZmwStatsFile,WriteReadUnbuffered)
{
   // PacBio::Logging::LogSeverityContext _(PacBio::Logging::LogLevel::DEBUG);

    TempFileManager tfm;
    tfm.Keep();


    std::string filename = tfm.GenerateTempFileName("_unbuf_sts.h5");
    ZmwStatsFile::NewFileConfig config;
    config.numAnalogs = 4;
    config.numFilters = 2;
    config.numFrames = 5*16384;
    config.numHoles = NumZmws;
    config.binSize = 16384;
    config.mfBinSize = 8192;
    {
        ZmwStatsFile::NewFileConfig config1;
        config1.numAnalogs = 4;
        config1.numFilters = 2;
        config1.numFrames = 5*16384;
        config1.numHoles = NumZmws;
        config1.binSize = 0;
        EXPECT_THROW(ZmwStatsFile file(filename, config1);,std::exception);
    }
    {
        PBLOG_DEBUG << "Creating " << filename;
        TEST_COUT << "Creating " << filename << std::endl;
        ZmwStatsFile file(filename, config);

        ZmwStats zmw(file.nA(),file.nF(),file.nMF());

        for(hsize_t i=0;i<file.nH();i++)
        {
            zmw.Init();
            Fill(zmw,i);
            file.Set(i,zmw);
        }
        TEST_COUT << "Closing " << filename << std::endl;
    }
    TEST_COUT << "Closed " << filename << std::endl;

    TEST_COUT << "Validating " << filename << std::endl;

    Validate(filename,config);

    if (! HasFailure()) tfm.DontKeep();
}

TEST(ZmwStatsFile,WriteReadBuffered)
{
    // PacBio::Logging::LogSeverityContext _(PacBio::Logging::LogLevel::DEBUG);

    TempFileManager tfm;
    tfm.Keep();


    std::string filename = tfm.GenerateTempFileName("_buf_sts.h5");
    ZmwStatsFile::NewFileConfig config;
    config.numAnalogs = 4;
    config.numFilters = 2;
    config.numFrames = 5*16384;
    config.numHoles = NumZmws;
    config.binSize = 16384;
    config.mfBinSize = 8192;
    {
        PacBio::Dev::AutoTimer timer("write",NumZmws);
        TEST_COUT << "Creating " << filename << std::endl;
        ZmwStatsFile file(filename, config);

#ifdef OLDWAY

        for(hsize_t i=0;i<file.nH();i++)
        {
            std::unique_ptr<ZmwStats> zmw = std::move(file.GetZmwStatsBuffer());
            zmw->Init();
            Fill(*zmw,i);
            file.SetBuffer(i,std::move(zmw));
        }
#else

        const uint32_t bufSize = 100; // this is what baz2bam uses
        for(hsize_t i=0;i<file.nH();i+=bufSize)
        {
            auto bufferList = file.GetZmwStatsBufferList();
            for (hsize_t j = 0; j < bufSize; j++)
            {
                bufferList->emplace_back(file.GetZmwStatsTemplate());
                ZmwStats& zmw = bufferList->back();
                zmw.Init();
                Fill(zmw, i + j);
            }
            file.WriteBuffers(std::move(bufferList));
        }
#endif
        TEST_COUT << "Closing " << filename << std::endl;
    }
    TEST_COUT << "Closed " << filename << std::endl;

    TEST_COUT << "Validating " << filename << std::endl;

    PacBio::Dev::AutoTimer timer("read",NumZmws);
    Validate(filename,config);

    if (! HasFailure()) tfm.DontKeep();
}


TEST(ZmwStatsFile,FractionalFile)
{
    // PacBio::Logging::LogSeverityContext _(PacBio::Logging::LogLevel::DEBUG);

    TempFileManager tfm;
    tfm.Keep();


    ZmwStatsFile::NewFileConfig config;
    config.numAnalogs = 4;
    config.numFilters = 2;
    config.numFrames = 6144;
    config.numHoles = NumZmws;
    config.binSize = 16384;
    config.mfBinSize = 8192;
    EXPECT_EQ(1,config.NumTimeslices());
    EXPECT_EQ(6144,config.LastBinFrames());

    std::string filename = tfm.GenerateTempFileName("_unbuf_sts.h5");
    {
        PBLOG_DEBUG << "Creating " << filename;
        ZmwStatsFile file(filename, config);

        ZmwStats zmw(file.nA(),file.nF(),file.nMF());

        for(hsize_t i=0;i<file.nH();i++)
        {
            zmw.Init();
            Fill(zmw,i);
            file.Set(i,zmw);
        }
    }

    Validate(filename,config);

    if (! HasFailure()) tfm.DontKeep();
}
