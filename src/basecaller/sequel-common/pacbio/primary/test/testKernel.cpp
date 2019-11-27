#include <memory>
#include <gtest/gtest.h>

#include <pacbio/text/String.h>
#include <pacbio/ipc/ThreadSafeQueue.h>
#include <pacbio/ipc/PoolMalloc.h>
#include <pacbio/ipc/JSON.h>
#include <pacbio/logging/Logger.h>
#include <pacbio/process/ProcessBase.h>
#include <pacbio/primary/SequelCalibrationFile.h>
#include <pacbio/primary/Tile.h>
#include <pacbio/primary/Acquisition_Setup.h>
#include <pacbio/primary/ipc_config.h>
#include <pacbio/primary/SequelMovieFileHDF5.h>
#include <pacbio/POSIX.h>


#include <Kernel.h>

using namespace PacBio::Primary;

TEST(Kernel,Constructor)
{
    std::string text = "[ [ 2 ] ]";
    Json::Value json = PacBio::IPC::ParseJSON(text);

    Kernel a(json);
    //std::cout << "a:\n" << a;
    EXPECT_FLOAT_EQ(2.0,a(0,0));
    EXPECT_EQ(1,a.NumRows());
    EXPECT_EQ(1,a.NumCols());

    Kernel a2 = a * a;
    //std::cout << "a2:\n" << a2;
    EXPECT_EQ(1,a2.NumRows());
    EXPECT_EQ(1,a2.NumCols());
    EXPECT_FLOAT_EQ(4.0,a2(0,0));

    Kernel e(3,3);
    e(1,1) = 1.0;
    Kernel e2 = e * e;
    //std::cout << "e:\n" << e << "e2:\n" << e2;
    EXPECT_EQ(0,e2(0,0));
    EXPECT_EQ(1.0,e2(2,2));

    Kernel b = a.Resize(5,5);
    //std::cout << "b:\n"<< b;
    EXPECT_EQ(5,b.NumRows());
    EXPECT_EQ(5,b.NumCols());
    EXPECT_FLOAT_EQ(0.0,b(0,0));
    EXPECT_FLOAT_EQ(2.0,b(2,2));
    EXPECT_FLOAT_EQ(0.0,b(4,4));

    Kernel c = b * b;
    //std::cout << "c:\n" << c;
    EXPECT_EQ(9,c.NumRows());
    EXPECT_EQ(9,c.NumCols());
    EXPECT_FLOAT_EQ(0.0,c(0,0));
    EXPECT_FLOAT_EQ(4.0,c(4,4));
    EXPECT_FLOAT_EQ(0.0,c(8,8));

    Kernel d = c.Normalize(1.0);
    //std::cout << "d:\n" << d;
    EXPECT_FLOAT_EQ(0.0,d(0,0));
    EXPECT_FLOAT_EQ(1.0,d(4,4));
    EXPECT_FLOAT_EQ(0.0,d(8,8));

}

TEST(Kernel,Austin)
{
    // source \\usmp-data\DATA\TraceSimulator\StudyCalibrations\OpticsSimulatorEC\Calibrations\PSFs\IMEC4p8Psf_Hybrid.mat

   auto Gjson = PacBio::IPC::ParseJSON("[[-0.001,0.0038,0.0067,0.0038,0.0019],"
                                 "[-0.001,0.0172,0.062,0.0153,0.0038],"
                                 "[0.001,0.0477,0.6692,0.0372,0.0076],"
                                 "[0.0029,0.0219,0.0629,0.0191,0.001],"
                                 "[0.0019,0.0048,0.0048,0.0038,0.0019]]");
   auto Rjson = PacBio::IPC::ParseJSON("[[0.003,0.004	,0.008	,0.005	,0.002],"
                            "[0.005	,0.023	,0.0549	,0.025	,0.006],"
                            "[0.01	,0.0579	,0.6004	,0.0579	,0.01],"
                            "[0.006	,0.022	,0.0509	,0.024	,0.006],"
                            "[0.002	,0.004	,0.007	,0.004	,0.002]]");

auto refJson = PacBio::IPC::ParseJSON("[[0.00008260,	0.00037200,	0.00081220,	0.00113290,	0.00090370,	0.00047140,	0.00012310],"
"[0.00032060,	-0.00086940,	-0.00337290,	-0.00704690,	-0.00407120,	-0.00186590,	0.00045530],"
"[0.00067390,	-0.00071310,	-0.02807010,	-0.09514780,	-0.02803120,	-0.00494040,	0.00087820],"
"[0.00106420,	-0.00636550,	-0.08347430,	 1.51236180,	-0.07823120,	-0.00970200,	0.00106420],"
"[0.00087820,	-0.00446380,	-0.02986920,	-0.09362650,	-0.02952290,	-0.00216590,	0.00067390],"
"[0.00045530,	-0.00186590,	-0.00404840,	-0.00559410,	-0.00337290,	-0.00179980,	0.00032060],"
"[0.00012310,	 0.00047140,	 0.00090370,	 0.00113290,	 0.00081220,	 0.00037200,	0.00008260]]");
   Kernel Gpsf(Gjson);
   Kernel Rpsf(Rjson);
   Kernel ref(refJson);

   Kernel sumPsf = Gpsf + Rpsf;
   Kernel meanPsf = sumPsf.Scale(0.5);
   //std::cout << "meanPsf:" << meanPsf;

   //std::cout << "meanPsf^-1:\n" << Invert(meanPsf);
    //std::cout << "Gpsf^-1:\n" << Invert(Gpsf);
    //std::cout << "Rpsf^-1:\n" << Invert(Rpsf);
    Kernel meanInv = (Gpsf.InvertTo7x7() + Rpsf.InvertTo7x7()).Scale(0.5);
    //std::cout << "mean(Gpsf^-1+Rpsf^-1):\n" << meanInv;

    for(int i=0;i<7;i++)
        for(int j=0;j<7;j++)
            EXPECT_NEAR(ref(i,j),meanInv(i,j),0.003);
}

TEST(Kernel,Austin2)
{
    auto refJson = PacBio::IPC::ParseJSON("[[0.00008260,	0.00037200,	0.00081220,	0.00113290,	0.00090370,	0.00047140,	0.00012310],"
                                                  "[0.00032060,	-0.00086940,	-0.00337290,	-0.00704690,	-0.00407120,	-0.00186590,	0.00045530],"
                                                  "[0.00067390,	-0.00071310,	-0.02807010,	-0.09514780,	-0.02803120,	-0.00494040,	0.00087820],"
                                                  "[0.00106420,	-0.00636550,	-0.08347430,	 1.51236180,	-0.07823120,	-0.00970200,	0.00106420],"
                                                  "[0.00087820,	-0.00446380,	-0.02986920,	-0.09362650,	-0.02952290,	-0.00216590,	0.00067390],"
                                                  "[0.00045530,	-0.00186590,	-0.00404840,	-0.00559410,	-0.00337290,	-0.00179980,	0.00032060],"
                                                  "[0.00012310,	 0.00047140,	 0.00090370,	 0.00113290,	 0.00081220,	 0.00037200,	0.00008260]]");
    Kernel ref(refJson);

#if 1
    // data from http://smrtwiki.nanofluidics.com/wiki/LVP1_wafer_dependence_page
        Spectrum  testVectors[13][3][2] = {
                {{{0.32, 0.68}, {0.05, 0.95}}, {{0.342, 0.658}, {0.105, 0.895}}, {{0.283, 0.717}, {0.018, 0.982}}},
                {{{0.39, 0.61}, {0.07, 0.93}}, {{0.404, 0.596}, {0.122, 0.878}}, {{0.354, 0.646}, {0.037, 0.963}}},
                {{{0.48, 0.52}, {0.08, 0.92}}, {{0.483, 0.517}, {0.131, 0.869}}, {{0.446, 0.554}, {0.047, 0.953}}},
                {{{0.48, 0.52}, {0.09, 0.91}}, {{0.483, 0.517}, {0.140, 0.860}}, {{0.446, 0.554}, {0.057, 0.943}}},
                {{{0.49, 0.51}, {0.09, 0.91}}, {{0.492, 0.508}, {0.140, 0.860}}, {{0.456, 0.544}, {0.057, 0.943}}},
                {{{0.50, 0.50}, {0.08, 0.92}}, {{0.501, 0.499}, {0.131, 0.869}}, {{0.467, 0.533}, {0.047, 0.953}}},
                {{{0.44, 0.56}, {0.06, 0.94}}, {{0.448, 0.552}, {0.114, 0.886}}, {{0.405, 0.595}, {0.028, 0.972}}},
                {{{0.47, 0.53}, {0.11, 0.89}}, {{0.474, 0.526}, {0.157, 0.843}}, {{0.436, 0.564}, {0.076, 0.924}}},
                {{{0.51, 0.49}, {0.09, 0.91}}, {{0.510, 0.490}, {0.140, 0.860}}, {{0.477, 0.523}, {0.057, 0.943}}},
                {{{0.48, 0.52}, {0.09, 0.91}}, {{0.483, 0.517}, {0.140, 0.860}}, {{0.446, 0.554}, {0.057, 0.943}}},
                {{{0.41, 0.59}, {0.07, 0.93}}, {{0.421, 0.579}, {0.122, 0.878}}, {{0.374, 0.626}, {0.037, 0.963}}},
                {{{0.50, 0.50}, {0.08, 0.92}}, {{0.501, 0.499}, {0.131, 0.869}}, {{0.467, 0.533}, {0.047, 0.953}}},
                {{{0.40, 0.60}, {0.07, 0.93}}, {{0.412, 0.588}, {0.122, 0.878}}, {{0.364, 0.636}, {0.037, 0.963}}}
        };
#else
    //updateSpectra=	raw matrix equivalents	Source data
    Spectrum  testVectors[12][2][2] = {
            {{{0.32, 0.68}, {0.05, 0.95}}, {{0.37, 0.63}, {0.13, 0.87}}},
            {{{0.39, 0.61}, {0.07, 0.93}}, {{0.44, 0.56}, {0.15, 0.85}}},
            {{{0.48, 0.52}, {0.08, 0.92}}, {{0.51, 0.49}, {0.16, 0.84}}},
            {{{0.48, 0.52}, {0.09, 0.91}}, {{0.51, 0.49}, {0.17, 0.83}}},
            {{{0.49, 0.51}, {0.09, 0.91}}, {{0.52, 0.48}, {0.17, 0.83}}},
            {{{0.50, 0.50}, {0.08, 0.92}}, {{0.53, 0.47}, {0.16, 0.84}}},
            {{{0.44, 0.56}, {0.06, 0.94}}, {{0.48, 0.52}, {0.14, 0.86}}},
            {{{0.47, 0.53}, {0.11, 0.89}}, {{0.50, 0.50}, {0.19, 0.81}}},
            {{{0.51, 0.49}, {0.09, 0.91}}, {{0.54, 0.46}, {0.17, 0.83}}},
            {{{0.48, 0.52}, {0.09, 0.91}}, {{0.51, 0.49}, {0.17, 0.83}}},
            {{{0.41, 0.59}, {0.07, 0.93}}, {{0.45, 0.55}, {0.15, 0.85}}},
            {{{0.50, 0.50}, {0.08, 0.92}}, {{0.53, 0.47}, {0.16, 0.84}}}
    };
#endif

    Kernel corrKernel = ref.Flip();

    for(int ivector=0;ivector<13;ivector++)
    {
        auto& expA = testVectors[ivector][0][0];
        auto& expB = testVectors[ivector][0][1];
        auto& rawA = testVectors[ivector][1][0];
        auto& rawB = testVectors[ivector][1][1];

        Spectrum corA;
        Spectrum corB;
        corA = CorrectSpectrum(ref,rawA);
        corB = CorrectSpectrum(ref,rawB);
        double accuracy = 0.009; // fixme, this should be on the order of 0.0005
        EXPECT_NEAR(expA[0],corA[0],accuracy);
        EXPECT_NEAR(expA[1],corA[1],accuracy);
        EXPECT_NEAR(expB[0],corB[0],accuracy);
        EXPECT_NEAR(expB[1],corB[1],accuracy);

    }
}



TEST(Kernel,UnityPSF)
{
    auto json = PacBio::IPC::ParseJSON("[[1]]");

    Kernel psf(json);

    //std::cout << "psf:\n" << psf;
    Kernel ipsf = psf.Resize(5,5).InvertTo7x7();
    //std::cout << "psf^-1:\n" << ipsf;
    EXPECT_FLOAT_EQ(1.0,ipsf.Sum());
    EXPECT_FLOAT_EQ(1.0,ipsf(3,3));
}

TEST(Kernel, IsUnity)
{
    auto kJson1 = PacBio::IPC::ParseJSON("[[1]]");
    Kernel u1(kJson1);
    EXPECT_TRUE(u1.IsUnity());

    auto kJson2 = PacBio::IPC::ParseJSON("[[0,0,0],[0,1,0],[0,0,0]]");
    Kernel u2(kJson2);
    EXPECT_TRUE(u2.IsUnity());

    auto kJson3 = PacBio::IPC::ParseJSON("[[0.99]]");
    Kernel u3(kJson3);
    EXPECT_FALSE(u3.IsUnity());

    auto kJson4 = PacBio::IPC::ParseJSON("[[0,0],[0,1]]");
    Kernel u4(kJson4);
    EXPECT_FALSE(u4.IsUnity());

    auto kJson5 = PacBio::IPC::ParseJSON("[[1,0],[0,0]]");
    Kernel u5(kJson5);
    EXPECT_FALSE(u5.IsUnity());
}
