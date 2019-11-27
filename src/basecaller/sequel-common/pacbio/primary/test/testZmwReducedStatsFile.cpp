// Copyright (c) 2014-2016, Pacific Biosciences of California, Inc.
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted (subject to the limitations in the
// disclaimer below) provided that the following conditions are met:
//
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
//  * Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
//  * Neither the name of Pacific Biosciences nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE
// GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY PACIFIC
// BIOSCIENCES AND ITS CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL PACIFIC BIOSCIENCES OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
// USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
// OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGE.

// Programmer: Mark Lakata
//

#include <boost/filesystem.hpp>
#include <cmath>
#include <gtest/gtest.h>
#include <pacbio/dev/gtest-extras.h>
#include <pacbio/primary/ZmwReducedStatsFile.h>
#include <pacbio/primary/ZmwStatsFile.h>
#include <pacbio/primary/ChipLayoutSequel.h>

using namespace PacBio::Dev;
using namespace PacBio::Primary;

static const std::string stsh5 = "/dept/primary/testdata/ppa-reducestats/SimMovContext_16110123115160.sts.h5";
static const std::string realstsh5 = "/dept/primary/testdata/ppa-reducestats/m54011_161116_090326.sts.h5";

static const std::string json =
        R"json(
    {
        "BinCols" : 8,
        "BinRows" : 10,
        "Outputs": [
          { "Input": "/ZMWMetrics/NumBases",        "Algorithm": "Sum",     "Filter": "All"},
        { "Input": "/ZMWMetrics/HQPkmid",           "Algorithm": "Median",  "Filter": "P1"},
        { "Input": "/ZMWMetrics/SnrMean",           "Algorithm": "Median",  "Filter": "Sequencing"},
        { "Input": "/ZMWMetrics/HQRegionStart",     "Algorithm": "Median",  "Filter": "P1"},
        { "Input": "/ZMWMetrics/HQRegionStartTime", "Algorithm": "Median",  "Filter": "P1", "Type": "uint16" },
        { "Input": "/ZMWMetrics/Productivity",      "Algorithm": "Count=1", "Filter": "P1", "Type": "uint8"},
        { "Input": "/ZMWMetrics/BaselineLevel",     "Algorithm": "Median",  "Filter": "Sequencing" , "BinRows": 8}
        ]
    }
    )json";

std::string GetStsTestFile(const std::string& fn)
{
    const std::string roots[] = {
        "/pbi",
        "/home/pbi"    
    };

    for (const std::string& r : roots)
    {
        boost::filesystem::path dir(r);
        boost::filesystem::path file(fn);
        boost::filesystem::path p = dir / file;
        if (boost::filesystem::exists(p))
            return p.string();
    }

    return (boost::filesystem::path(roots[0]) / boost::filesystem::path(fn)).string();
}

TEST(ZmwReducedStatsFile,ReducedStatsConfig)
{
    ReducedStatsConfig config(ChipClass::Sequel);
    config.Load(json);

    EXPECT_EQ("/ZMWMetrics/NumBases",config.Outputs[0].Input());
    EXPECT_EQ(10,config.BinRows());

    ReducedStatsConfig outputConfig(ChipClass::Sequel);
    outputConfig.Copy(config);

    EXPECT_EQ(10,outputConfig.BinRows());
    outputConfig.Load(config.Outputs[6].Json()); // this is a trick to overwrite the parent with the child values IF they are set
    EXPECT_EQ(8,outputConfig.BinRows());

    outputConfig.BinRows = 3;
    outputConfig.BinCols = 16;
    outputConfig.UnitCellOffsetX = 3;
    outputConfig.UnitCellOffsetY = 128;
    outputConfig.UnitCellRows = 7;
    outputConfig.UnitCellCols = 32;

    Reducer::Binning binning(outputConfig);
    EXPECT_EQ(3,binning.BinRows());
    EXPECT_EQ(16,binning.BinCols());
    EXPECT_EQ(3,binning.UnitCellOffsetX());
    EXPECT_EQ(128,binning.UnitCellOffsetY());
    EXPECT_EQ(7,binning.UnitCellRows());
    EXPECT_EQ(32,binning.UnitCellCols());

    EXPECT_EQ(3,binning.NumOutputRows());
    EXPECT_EQ(2,binning.NumOutputCols());

    EXPECT_EQ(3*16,binning.NumBinValues());
    EXPECT_EQ(3*2,binning.NumOutputCount());
    EXPECT_EQ(7*32,binning.NumInputCount());
}

using namespace PacBio::Primary::Reducer;

// The NaN should get filtered out of all computations
static std::vector<uint8_t> filter1 = { 1,1,0,1,1,1};
static std::vector<uint8_t> filter2 = { 0,1,1,1,1,1};
static std::vector<uint8_t> filter3 = { 1,0,1,1,0,1};
static std::vector<uint8_t> filter4 = { 0,0,0,0,0,1};
static std::vector<double>  data1 = { 1,2,3,4,5,std::numeric_limits<float>::quiet_NaN()};


TEST(ZmwReducedStatsFile,Algorithm_count)
{
    Algorithm a("Count=1");
    EXPECT_EQ("Count",a.ShortName());
    EXPECT_EQ("Count=1",a.FullName());
    EXPECT_FLOAT_EQ(1.0, a.Apply(data1, filter1));
    EXPECT_FLOAT_EQ(0.0, a.Apply(data1, filter2));
}

TEST(ZmwReducedStatsFile,Algorithm_sum)
{
    Algorithm a(Algorithm::Algorithm_t::Sum);
    EXPECT_FLOAT_EQ(12.0, a.Apply(data1, filter1));
    EXPECT_TRUE( isnan(a.Apply(data1, filter4)));
}

TEST(ZmwReducedStatsFile,Algorithm_subsample)
{
    Algorithm a(Algorithm::Algorithm_t::Subsample);
    EXPECT_FLOAT_EQ(1.0, a.Apply(data1, filter1));
    EXPECT_FLOAT_EQ(1.0, a.Apply(data1, filter2));
}

TEST(ZmwReducedStatsFile,Algorithm_mean)
{
    Algorithm a(Algorithm::Algorithm_t::Mean);
    EXPECT_FLOAT_EQ(3.0, a.Apply(data1, filter1));
    EXPECT_FLOAT_EQ(8.0/3.0, a.Apply(data1, filter3));
}

TEST(ZmwReducedStatsFile,Algorithm_median)
{
    Algorithm a(Algorithm::Algorithm_t::Median);
    EXPECT_FLOAT_EQ(3.0, a.Apply(data1, filter1));
    EXPECT_FLOAT_EQ(3.5, a.Apply(data1, filter2));
    EXPECT_FLOAT_EQ(3.0, a.Apply(data1, filter3));
}

TEST(ZmwReducedStatsFile,Algorithm_min)
{
    Algorithm a(Algorithm::Algorithm_t::Min);
    EXPECT_FLOAT_EQ(1.0, a.Apply(data1, filter1));
    EXPECT_FLOAT_EQ(2.0, a.Apply(data1, filter2));
}

TEST(ZmwReducedStatsFile,Algorithm_max)
{
    Algorithm a(Algorithm::Algorithm_t::Max);
    EXPECT_FLOAT_EQ(5.0, a.Apply(data1, filter1));
    EXPECT_FLOAT_EQ(4.0, a.Apply(data1, filter3));
}

TEST(ZmwReducedStatsFile,Algorithm_stddev)
{
    EXPECT_THROW(Algorithm a(Algorithm::Algorithm_t::Stddev), std::runtime_error);
  //  EXPECT_FLOAT_EQ(1.5, a.Apply(data1, filter1));
}

TEST(ZmwReducedStatsFile,Algorithm_mad)
{
    EXPECT_THROW(Algorithm a(Algorithm::Algorithm_t::MAD), std::runtime_error);
  //  EXPECT_FLOAT_EQ(0.0, a.Apply(data1, filter1));
}

// static const char* mm= "/pbi/dept/primary/testdata/ppa-reducestats/m54004_161114_225358.sts.h5";
// /home/UNIXHOME/mlakata/m54004_161114_225358.sts.h5";
// # baz2bam /home/UNIXHOME/mlakata/m54004_161114_225358.baz -o /home/UNIXHOME/mlakata/m54004_161114_225358 -m /home/UNIXHOME/mlakata/m54004_161114_225358.run.metadata.xml


class ZmwReducedStatsFileFeatures : public ::testing::Test
{
public:
    TempFileManager tfm;

    std::string filename_;
    void SetUp() override
    {
//tfm.Keep();
        filename_ = tfm.GenerateTempFileName("_sts.h5");
        ZmwStatsFile::NewFileConfig config;
        config.numAnalogs = 4;
        config.numFilters = 2;
        config.numFrames = 6144;
        config.numHoles = 6;
        config.binSize = 16384;
        config.mfBinSize = 1; // to avoid divide by zero error
        EXPECT_EQ(1, config.NumTimeslices());
        EXPECT_EQ(6144, config.LastBinFrames());

        std::vector<uint32_t> features = {
                0,
                1,
                1,
                1,
                0,
                0
        };
        std::vector<ZmwStatsFile::Productivity_t> productivity = {
                ZmwStatsFile::Productivity_t::Productive,
                ZmwStatsFile::Productivity_t::Empty,
                ZmwStatsFile::Productivity_t::Empty,
                ZmwStatsFile::Productivity_t::Empty,
                ZmwStatsFile::Productivity_t::Other,
                ZmwStatsFile::Productivity_t::Empty
        };


        {
            PBLOG_DEBUG << "Creating " << filename_;
            ZmwStatsFile file(filename_, config);

            ZmwStats zmw(file.nA(), file.nF(), file.nMF());

            for (hsize_t i = 0; i < file.nH(); i++)
            {
                zmw.Init();
                zmw.index_ = i;
                zmw.UnitFeature = features.at(i);
                zmw.Productivity = productivity.at(i);
                zmw.HoleNumber = 0;
                file.Set(i, zmw);
            }
        }
    }
};

TEST_F(ZmwReducedStatsFileFeatures,Filter_all)
{
    // PacBio::Logging::LogSeverityContext _(PacBio::Logging::LogLevel::DEBUG);

    Filter f(Filter::Filter_t::All);
    EXPECT_EQ("All",f.Name());
    ZmwStatsFile zsf(filename_);
    f.Load(zsf);
    EXPECT_EQ(1,f.IsSelected(0));
    EXPECT_EQ(1,f.IsSelected(1));
    EXPECT_EQ(1,f.IsSelected(2));
    EXPECT_EQ(1,f.IsSelected(3));
    EXPECT_EQ(1,f.IsSelected(4));
    EXPECT_EQ(1,f.IsSelected(5));
}

TEST_F(ZmwReducedStatsFileFeatures,Filter_Sequencing)
{
    Filter f(Filter::Filter_t::Sequencing);
    EXPECT_EQ("Sequencing",f.Name());
    ZmwStatsFile zsf(filename_);
    f.Load(zsf);
    EXPECT_EQ(0,f.IsSelected(0));
    EXPECT_EQ(0,f.IsSelected(1));
    EXPECT_EQ(0,f.IsSelected(2));
    EXPECT_EQ(0,f.IsSelected(3));
    EXPECT_EQ(0,f.IsSelected(4));
    EXPECT_EQ(0,f.IsSelected(5));
}

TEST_F(ZmwReducedStatsFileFeatures,Filter_NonSequencing)
{
    Filter f(Filter::Filter_t::NonSequencing);
    EXPECT_EQ("NonSequencing", f.Name());
    ZmwStatsFile zsf(filename_);
    f.Load(zsf);
    EXPECT_EQ(1,f.IsSelected(0));
    EXPECT_EQ(1,f.IsSelected(1));
    EXPECT_EQ(1,f.IsSelected(2));
    EXPECT_EQ(1,f.IsSelected(3));
    EXPECT_EQ(1,f.IsSelected(4));
    EXPECT_EQ(1,f.IsSelected(5));
}
TEST_F(ZmwReducedStatsFileFeatures,Filter_P1)
{
    Filter f(Filter::Filter_t::P1);
    EXPECT_EQ("P1",f.Name());
    ZmwStatsFile zsf(filename_);
    f.Load(zsf);
    EXPECT_EQ(1,f.IsSelected(0));
    EXPECT_EQ(0,f.IsSelected(1));
    EXPECT_EQ(0,f.IsSelected(2));
    EXPECT_EQ(0,f.IsSelected(3));
    EXPECT_EQ(0,f.IsSelected(4));
    EXPECT_EQ(0,f.IsSelected(5));
}

#ifdef PB_MIC_COPROCESSOR
TEST(ZmwReducedStatsFile,DISABLED_TestWithFakeData)
#else
TEST(ZmwReducedStatsFile,TestWithFakeData)
#endif
{
    // PacBio::Logging::LogSeverityContext _(PacBio::Logging::LogLevel::DEBUG);

    TempFileManager tfm;
    tfm.Keep();

    std::string filename = tfm.GenerateTempFileName("_rsts.h5");
    ReducedStatsConfig config(ChipClass::Sequel);
    config.Load(json);

    // write
    {
        ZmwReducedStatsFile file(filename, config);
        ZmwStatsSetQuiet();
        ZmwStatsFile zsf(GetStsTestFile(stsh5));
        zsf.CopyScanData(file.ScanData());
        file.Reduce(zsf, config);
        file.Close();
        ZmwStatsSetVerbose();
    }

    // read
    {
        ZmwReducedStatsFile readback(filename);
        {
            ZmwReducedDataSet zrds = readback.GetDataSet("/ReducedZMWMetrics/All/Sum/NumBases");

            // check old attributes
            EXPECT_EQ("Number of called bases", zrds.Description());
            EXPECT_EQ(16384, zrds.BinSize());
            EXPECT_EQ("bases", zrds.UnitsOrEncoding());
            EXPECT_FALSE(zrds.HQRegion());

            // check new attriutes
            EXPECT_EQ(10,zrds.BinRows());
            EXPECT_EQ(8,zrds.BinCols());
            EXPECT_EQ(64,zrds.UnitCellOffsetX());
            EXPECT_EQ(64,zrds.UnitCellOffsetY());
            EXPECT_EQ(Reducer::Algorithm::Algorithm_t::Sum,zrds.Algorithm());
            EXPECT_EQ(Reducer::Filter::Filter_t::All      ,zrds.Filter());

            std::vector<double> data;
            zrds.DataSet() >> data;
            EXPECT_EQ(120*108,data.size());
            EXPECT_FLOAT_EQ(393*8*10, data[0]);
            EXPECT_FLOAT_EQ(393*8*10, data[1]);
        }
        if (0) {
            ZmwReducedDataSet zrds = readback.GetDataSet("/ReducedZMWMetrics/P1/Median/HQPkmid");

            EXPECT_EQ("Pkmid in HQ-region, 0 if no HQ-region", zrds.Description());
            EXPECT_EQ(16384, zrds.BinSize());
            EXPECT_EQ("photo e-", zrds.UnitsOrEncoding());
            EXPECT_TRUE(zrds.HQRegion());

            EXPECT_EQ(Reducer::Algorithm::Algorithm_t::Median,zrds.Algorithm());
            EXPECT_EQ(Reducer::Filter::Filter_t::P1         ,zrds.Filter());
        }
    }

    if (! HasFailure()) tfm.DontKeep();
}

#ifdef PB_MIC_COPROCESSOR
TEST(ZmwReducedStatsFile,DISABLED_TestWithRealData)
#else
TEST(ZmwReducedStatsFile,TestWithRealData)
#endif
{
    // PacBio::Logging::LogSeverityContext _(PacBio::Logging::LogLevel::DEBUG);

    TempFileManager tfm;
    tfm.Keep();

    std::string filename = tfm.GenerateTempFileName("_rsts.h5");
    ReducedStatsConfig config(ChipClass::Sequel);
    config.Load(json);

    // write
    {
        ZmwStatsSetQuiet();
        ZmwReducedStatsFile file(filename, config);
        ZmwStatsFile zsf(GetStsTestFile(realstsh5));
        zsf.CopyScanData(file.ScanData());
        file.Reduce(zsf, config);
        file.Close();
        ZmwStatsSetVerbose();
    }
}


#ifdef PB_MIC_COPROCESSOR
TEST(ZmwReducedStatsFile,DISABLED_Unity)
#else
TEST(ZmwReducedStatsFile,Unity)
#endif
{
    const std::string testJson =
            R"json(
    {
        "BinCols" : 8,
        "BinRows" : 8,
        "Outputs": [
           { "Input": "/ZMWMetrics/NumBases", "Algorithm":"Sum"},
           { "Input": "/ZMWMetrics/HQPkmid", "Algorithm":"Sum"}
        ]
    }
    )json";


    TempFileManager tfm;
 //   tfm.Keep();

    std::string filename = tfm.GenerateTempFileName("_rsts.h5");
    ReducedStatsConfig config(ChipClass::Sequel);
    config.Load(testJson);

    // write
    {
        ZmwStatsSetQuiet();
        ZmwReducedStatsFile output(filename, config);
        ZmwStatsFile input(GetStsTestFile(stsh5));
        output.Reduce(input, config);
        output.Close();
        ZmwStatsSetVerbose();
    }
}
