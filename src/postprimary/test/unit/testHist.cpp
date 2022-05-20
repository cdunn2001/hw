#include <gtest/gtest.h>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <postprimary/stats/Histogram.h>

using namespace PacBio::Primary::Postprimary;

TEST(hist,infs)
{
    std::vector<float> values{std::numeric_limits<float>::infinity()};

    Histogram h(values, "OnlyInf");
    EXPECT_EQ(0, h.mean);
    EXPECT_TRUE(std::all_of(h.bins.begin(), h.bins.end(), [](int i ) { return i == 0; }));
}

TEST(hist,nans)
{
    // These angle estimates were directly pulled from the BAZ file from:
    // /pbi/collections/324/3241753/r54054_20200430_205334/1_A01/m54054_200430_210258.trc.h5
    std::vector<float> values{ 79.5625,79.625,79.875,79.6875,79.6875,79.6875,80.5625,81.0625,79.3125,std::numeric_limits<float>::quiet_NaN(),
                               79.3125,80.25,80.25,80.25,77.875,77.8125,77.75,77.75,77.75,78.25,78.1875,
                               78.375,78.25,77.75,77.75,78.125,78.0625,78.0625,78.125,78.125,77.0625,76.875,
                               77.0625,78.1875,78.1875,78.5,78.1875,78.125,78.1875,78.1875,79.1875,79.1875,
                               79.1875,79.1875,79.0625,79.1875,78.625,78.625,78.625,78.625,78.5,78.5,78.5,
                               78.5,79.75,77.9375 };

    Histogram h(values, "Angles");
    EXPECT_NE(0, h.mean);
    EXPECT_FALSE(isnan(h.mean));
    EXPECT_FALSE(std::all_of(h.bins.begin(), h.bins.end(), [](int i ) { return i == 0; }));
}

TEST(hist,varBins)
{
    std::vector<float> values(100, 0);
    for(int i = 0; i < 100; i++)
        values[i] = 500.0f*i;

    Histogram h(values, "TEST", 500, 10000);
    EXPECT_EQ(0, h.min);
    EXPECT_EQ(10000/500, h.bins.size());
    EXPECT_EQ(10000, h.max);
    EXPECT_EQ(500*99, h.last);
    EXPECT_EQ(35000, h.n50);

    Histogram h2(values, "TEST", 500, 500*100);
    EXPECT_EQ(500*99, h2.max);
    EXPECT_EQ(0, h2.min);
    EXPECT_EQ(99, h2.bins.size());
    EXPECT_EQ(35000, h2.n50);
}

TEST(hist,empty)
{
    Histogram h("empty", 500, 200);
    EXPECT_EQ(200, h.bins.size());
    EXPECT_EQ(500, h.binWidth);
    EXPECT_TRUE(std::all_of(h.bins.begin(), h.bins.end(), [](int b) { return b == 0; }));
    EXPECT_EQ(0, h.sampleSize);
    EXPECT_EQ(0, h.mean);
    EXPECT_EQ(0, h.median);
    EXPECT_EQ(0, h.stddev);
    EXPECT_EQ(0, h.min);
    EXPECT_EQ(0, h.max);
    EXPECT_EQ(0, h.n50);
}


TEST(hist,rng)
{
	std::vector<float> data = {368.1333,443.9700,390.0102,385.9676,383.3967,426.3473,427.4101,411.6332,348.4607,395.7673,363.3634,476.5937,443.2861,456.8727,395.2290,440.9303,377.9110,385.4221,346.7000,323.0699,300.3055,368.1207,422.1993,327.2441,438.4426,359.9148,410.9432,434.2984,331.6076,543.4594,390.3338,397.1609,401.8417,424.6409,433.3844,346.4417,487.7776,370.2568,363.8474,507.1267,381.3183,426.0616,464.1518,373.4477,435.3056,354.7677,404.1048,368.2112,361.1939,431.9972,491.9786,381.8985,401.8236,397.3317,360.3108,471.3449,394.2978,414.4627,428.0303,471.5451,406.4461,356.9346,429.9503,544.1463,341.0444,352.6345,339.3935,325.9908,354.2284,435.4581,488.9022,411.1414,403.2648,353.3392,482.6578,418.6866,397.7412,411.8806,401.9999,403.3795,405.5911,468.6039,344.9188,347.3419,398.4935,444.1310,372.4253,415.7935,442.9070,440.8013,418.8466,391.4614,309.0224,338.3220,311.0045,418.7782,429.3194,458.4690,417.5992,387.3044,367.3341,428.2388,375.3039,354.1467,443.2329,413.5831,401.4753,361.2673,370.7888,389.6137,406.5576,321.9117,380.2883,397.1915,464.1544,435.9470,334.7483,370.3574,337.4030,490.0523,497.9629,283.9536,408.1353,345.3818,383.7602,312.7410,429.7787,344.1282,375.7373,360.1646,443.1275,410.5663,300.5123,355.9540,371.9798,387.9361,361.5568,293.1070,469.0758,451.7410,417.6900,406.9915,391.5238,390.5293,455.0870,417.7582,411.5107,424.5332,278.8322,381.6561,454.2283,481.6095,346.9366,381.1269,467.2030,439.6872,263.2155,338.1501,488.6229,364.3120,432.9773,363.8359,372.0791,432.0005,471.2585,368.1025,420.0245,386.4181,401.5311,407.7526,404.0314,295.5453,463.9061,429.9546,452.3549,474.1119,442.9825,334.8777,343.5808,348.4049,473.5364,414.7245,427.7321,347.6047,367.7958,443.2857,470.9792,408.6496,436.9065,407.4936,405.1373,466.7901,369.0275,407.4088,390.7932,375.0053,333.5989,348.7245,428.1421,510.5641,383.5439,389.4611,309.7483,353.3551,341.1395,420.9901,421.6354,413.3389,474.3588,420.9807,329.4718,416.5646,393.9743,390.6077,416.1724,337.1077,352.1295,395.2116,380.6594,374.0422,434.4244,367.6015,448.6664,373.3239,398.0681,416.2253,487.1154,355.3695,396.7084,377.2876,317.0957,432.6884,430.2652,406.0518,386.4566,403.3990,411.9528,328.9062,398.5012,315.6444,477.6611,339.1863,401.5356,406.2598,363.8225,391.9400,357.7353,362.3626,408.3648,445.0323,399.5658,358.1004,431.8649,347.1786,370.9362,396.9963,369.8912,377.1329,412.7219,399.4887,464.6247,365.3151,401.2080,338.0397,424.4323,428.6749,411.5956,373.6250,586.7301,458.6035,331.7107,405.3534,436.1748,367.2232,448.2406,351.1688,355.6405,488.9458,347.0773,460.9318,371.0449,402.4589,419.7174,391.0023,423.1521,435.0383,434.7910,386.2533,426.7232,383.5452,359.9063,466.6870,356.1960,455.2929,411.2167,479.7051,355.6714,430.2620,413.4580,471.3388,359.7943,399.8411,398.8447,419.3192,344.4261,371.6644,359.9520,494.0957,453.5274,409.4036,349.6146,418.0003,260.8545,491.9454,446.0923,506.6624,400.2984,445.2957,344.3903,352.2650,398.9188,420.1981,414.8756,459.0652,350.6727,415.2874,411.5125,438.9141,412.5266,400.3204,375.9600,337.3506,475.9032,497.1488,384.4071,424.6288,460.1484,353.8916,494.9684,344.7548,447.5777,461.1047,396.1901,469.2153,290.2372,361.8690,402.2041,400.7724,397.5180,319.5913,455.9970,404.9987,372.9777,370.8111,411.9422,389.4395,465.3917,450.8162,494.8571,404.2271,416.3685,412.9463,370.8850,440.7750,401.4658,467.0051,384.4602,402.6525,375.1201,431.8307,271.8634,417.5778,439.9366,356.4805,480.1676,380.5783,272.7890,333.5369,424.9676,329.5300,433.5519,390.7238,462.3802,383.5764,428.2597,416.7048,446.0312,369.6089,354.6589,395.6615,416.5030,399.4464,430.1400,401.9766,395.1555,352.6189,517.7234,433.2277,418.3346,453.9093,326.1330,354.7281,477.7568,415.8825,337.6999,236.5266,431.4655,432.0124,473.1146,402.9294,341.4474,384.2941,363.4465,441.9508,369.7852,411.9267,302.0709,321.1226,488.2315,447.8627,334.6511,408.9575,350.6219,506.5393,453.1661,429.6368,401.2464,446.1960,477.2293,389.7539,412.5558,438.9938,387.4822,319.3629,424.0699,448.7436,439.4112,409.7486,466.6652,479.4357,354.7924,312.5215,361.3038,400.1991,345.9587,439.0051,400.4816,447.2065,367.4697,373.6106,381.2116,388.0681,464.5225,527.0954,411.9595,385.0844,493.7590,404.5887,442.1665,373.9578,457.5411,409.8787,431.6105,531.3489,388.4942,376.6659,398.0940,341.3758,407.1571,347.2176,393.1753,489.0551,318.3884,375.5647,355.0636,297.9566,390.1211,350.6405,349.0861,372.3703,433.5459,342.8251,419.0106,324.1545,364.6348,400.4604,362.9345,389.6236,373.5376,394.1508,379.5489,349.9216,293.2444,378.0810,394.2757,419.4743,423.2263,365.5155,417.6741,366.8390,345.8271,430.8566,339.2477,405.0929,362.1722,357.7260,411.8210,497.2593,467.5406,428.2094,448.8063,356.5792,435.8896,454.4227,418.2939,371.2422,354.4073,413.7209,409.9707,416.1583,405.1619,392.7825,345.8681,343.4371,317.8323,454.2874,429.8402,430.0728,445.4320,408.2785,422.2924,473.7395,384.2530,409.2962,372.6817,463.3757,411.9411,406.3855,365.0602,332.0730,427.9207,377.1182,384.7634,525.6879,442.3419,412.8876,421.9962,440.2947,426.1849,430.0190,387.1952,362.4090,358.5094,401.5963,333.4255,356.8959,366.5903,407.1838,442.9937,340.8363,376.5383,413.1597,348.5310,372.2534,489.5445,339.9320,450.7129,365.2887,346.3332,424.2406,378.8458,329.4252,344.9389,420.8065,377.1410,333.5130,448.6393,394.0195,359.4386,310.3026,397.4894,360.4597,434.7019,335.8335,415.4617,413.9457,369.9546,387.7918,442.6255,367.1448,437.0282,390.7621,438.6846,426.1553,410.7036,482.6857,489.8890,315.7601,420.6166,421.9993,414.9612,376.1130,357.3808,473.9352,476.2315,437.3972,457.9243,387.1037,371.0423,472.9625,387.9715,485.0444,383.8411,387.9019,377.4544,421.6755,366.9035,451.7349,480.3145,347.5584,388.7013,396.9472,462.2402,458.9165,404.5318,395.1366,383.1600,329.7095,448.4718,334.8548,371.0088,387.6313,397.4458,405.1297,336.7199,441.4022,458.3342,388.9259,382.6661,482.5866,384.2514,355.0716,341.5522,523.1943,293.9574,474.3596,429.0223,534.5129,329.7829,421.3962,424.9176,356.3229,465.5051,494.2435,402.0906,407.2707,460.4214,344.9147,339.7859,469.3860,471.4173,423.8347,391.6021,377.1077,474.4163,282.6074,375.9793,410.8221,348.1976,407.9275,468.5941,329.7411,382.4366,417.3076,492.7516,373.2669,407.1424,427.0275,420.7607,497.7187,403.5704,416.4243,301.5795,402.7224,426.9310,363.1849,452.0591,332.6192,413.3386,439.1922,433.3461,415.2306,394.8727,447.7048,328.8139,447.6351,388.2842,404.8204,345.4346,456.5345,406.8092,428.7606,317.6441,383.9096,451.0791,425.8536,481.7397,417.7787,426.9944,380.8098,394.6717,526.8007,435.4069,386.2944,410.5785,395.9800,422.1277,340.3184,396.4237,353.1071,419.1881,464.4855,341.2821,375.4311,414.9005,336.9386,361.3023,420.5627,438.5961,404.0839,411.4506,465.7185,405.9116,352.2302,341.5516,391.8387,370.0568,435.8169,378.8478,401.2155,462.4764,463.3920,430.2717,319.5160,387.7973,384.6849,362.0509,367.6112,430.9233,395.0430,287.8636,396.4352,418.6376,371.5701,417.3406,393.4246,433.1801,289.8263,338.9752,413.6200,349.5841,395.2794,385.6207,376.6789,453.1615,400.3320,402.7295,323.4188,446.5953,428.5757,419.8918,423.5743,426.5096,395.4510,495.4671,374.7465,433.3258,440.0980,295.2271,470.8320,434.3554,289.6200,371.3685,457.9533,399.5196,325.8360,334.9081,442.9641,361.0987,379.3560,435.8134,461.9946,455.0243,378.3314,401.5780,343.1044,498.3430,372.6012,443.2765,426.6753,282.9905,454.6912,326.1060,398.3492,469.1201,411.6412,397.3179,338.5780,435.0477,337.5445,363.2393,386.3742,434.4202,376.1872,471.9803,459.1094,337.4006,445.4266,416.1359,412.1496,407.7405,458.7636,340.2116,367.5121,340.9825,413.1728,374.4348,338.4485,402.9246,436.3571,377.1045,439.5022,418.9904,395.4682,408.4887,417.3785,481.0045,434.9509,396.7576,353.2808,353.2790,339.0892,422.8900,418.9968,432.5698,451.1991,337.8279,427.7036,438.6101,412.6654,299.4778,375.0357,422.4842,396.5475,447.5501,411.9793,391.0924,384.0535,382.4785,466.9565,351.0961,388.9585,482.3593,404.4099,332.6667,424.2471,299.8508,374.0081,462.6566,370.0450,431.5373,364.9927,394.4419,469.5842,391.5814,379.3535,382.8762,440.2586,371.6816,316.4302,308.9442,328.6165,361.5651,433.6129,393.6578,455.7720,435.2824,441.6778,408.4510,382.4795,470.2417,445.5828,384.8302,335.2455,398.5820,329.4700,434.3454,500.6001,449.1306,343.3702,399.0419,433.8894,471.7356,328.0215,413.4173,388.4669,473.8268,389.7153,325.8902,354.0551,381.2311,291.7697,433.7563,438.3280,418.0414,394.2507,401.0770,436.4331,354.1704,393.4160,417.6810,387.0850,400.9062,450.3186,387.4797,392.1312,401.2218,380.6874,411.8130,467.1323,328.8951,396.6805,562.1172,423.0253,325.4548,386.8586,342.8004,459.1318,368.5125,506.6051,303.8144,486.9857,375.8987,470.6895,462.3212,353.5960,446.9565,444.0056,474.5109,306.9637,393.6695,307.7516,459.2095,440.0206,358.0568,523.5173,472.8110,371.2796,294.7120,361.2482,412.1320,314.1172,418.7352,392.6552,379.3215,380.8270,358.2502,370.4946,377.5215,331.8193,408.1717,400.6808,454.3300,385.3706,388.0525,464.6627,486.4126,467.6948,416.3022,398.7660,436.5870,399.5014,450.3504,420.5228,443.1545,458.5318,410.7371,402.5718,417.2557,377.6643,478.5854,342.5862,424.9048,386.9343,432.1851,374.2888,377.8394,406.3752,441.4531,390.2171};
	Histogram h(data);
	std::cerr << h << std::endl;
}

TEST(hist,double2float)
{
    // Test that casting from double to float works as intended.

    double v1 = std::numeric_limits<double>::max();
    float f1 = static_cast<float>(v1);
    std::cerr << "f1: " << f1 << std::endl;
    // double max -> float inf
    EXPECT_EQ(std::numeric_limits<float>::infinity(), f1);

    double v2 = std::numeric_limits<double>::infinity();
    float f2 = static_cast<float>(v2);
    std::cerr << "f2: " << f2 << std::endl;
    // double inf -> float inf
    EXPECT_EQ(std::numeric_limits<float>::infinity(), f2);

    double v3 = std::numeric_limits<double>::min();
    float f3 = static_cast<float>(v3);
    std::cerr << "f3: " << f3 << std::endl;
    // double min is minimum positive normalized value -> float 0
    EXPECT_EQ(0, f3);

    double v4 = std::numeric_limits<double>::lowest();
    float f4 = static_cast<float>(v4);
    std::cerr << "f4: " << f4 << std::endl;
    // double lowest -> float -inf
    EXPECT_EQ(-1 * std::numeric_limits<float>::infinity(), f4);

    EXPECT_THROW({
        try
        {
            double v = std::numeric_limits<double>::max();
            (void) boost::numeric_cast<float>(v);
        }
        catch (boost::numeric::bad_numeric_cast& e)
        {
            EXPECT_STREQ("bad numeric conversion: positive overflow", e.what());
            throw;
        }
    }, boost::numeric::bad_numeric_cast);

    EXPECT_THROW({
        try
        {
            double v = std::numeric_limits<double>::infinity();
            (void) boost::numeric_cast<float>(v);
        }
        catch (boost::numeric::bad_numeric_cast& e)
        {
            EXPECT_STREQ("bad numeric conversion: positive overflow", e.what());
            throw;
        }
    }, boost::numeric::bad_numeric_cast);

    EXPECT_NO_THROW({
        try
        {
            // This is ok.
            double v = std::numeric_limits<double>::min();
            (void) boost::numeric_cast<float>(v);
        }
        catch (boost::numeric::bad_numeric_cast& e)
        {
            std::cerr << e.what() << std::endl;
            throw;
        }
    });

    EXPECT_THROW({
        try
        {
            double v = std::numeric_limits<double>::lowest();
            (void )boost::numeric_cast<float>(v);
        }
        catch (boost::numeric::bad_numeric_cast& e)
        {
            EXPECT_STREQ("bad numeric conversion: negative overflow", e.what());
            throw;
        }
    }, boost::numeric::bad_numeric_cast);
}



TEST(hist,sameval)
{
    // The case of the same data point should result in the bins with 0 counts. The
    // min and max will be set to the data point value, the sample size will reflect
    // the number of data points.

    std::vector<float> data;
    for (unsigned int i = 0; i < 100; i++) data.push_back(10);
    Histogram h(data);

    EXPECT_FLOAT_EQ(10, h.min);
    EXPECT_FLOAT_EQ(10, h.max);
    EXPECT_FLOAT_EQ(10, h.perc95th);
    EXPECT_FLOAT_EQ(10, h.mean);
    EXPECT_FLOAT_EQ(10, h.median);
    EXPECT_FLOAT_EQ(10, h.mode);
    EXPECT_FLOAT_EQ(0, h.stddev);
    EXPECT_FLOAT_EQ(10, h.first);
    EXPECT_FLOAT_EQ(10, h.last);
    EXPECT_FLOAT_EQ(0, h.binWidth);
    EXPECT_EQ(100, h.sampleSize);
    for (unsigned int b = 0; b < h.bins.size(); b++) EXPECT_EQ(0, h.bins[b]);

    std::vector<float> data2;
    for (unsigned int i = 0; i < 100; i++) data2.push_back(0);
    Histogram h2(data2);

    EXPECT_FLOAT_EQ(0, h2.min);
    EXPECT_FLOAT_EQ(0, h2.max);
    EXPECT_FLOAT_EQ(0, h2.perc95th);
    EXPECT_FLOAT_EQ(0, h2.mean);
    EXPECT_FLOAT_EQ(0, h2.median);
    EXPECT_FLOAT_EQ(0, h2.mode);
    EXPECT_FLOAT_EQ(0, h2.stddev);
    EXPECT_FLOAT_EQ(0, h2.first);
    EXPECT_FLOAT_EQ(0, h2.last);
    EXPECT_FLOAT_EQ(0, h2.binWidth);
    EXPECT_EQ(100, h2.sampleSize);
    for (unsigned int b = 0; b < h2.bins.size(); b++) EXPECT_EQ(0, h2.bins[b]);
}

TEST(hist,2vals)
{
    // For a 2 value distribution, compute summary stats but don't bin.

    std::vector<float> data = { 5, 6 };
    Histogram h(data);

    EXPECT_FLOAT_EQ(5, h.first);
    EXPECT_FLOAT_EQ(6, h.last);
    EXPECT_FLOAT_EQ(5.5, h.mean);
    EXPECT_FLOAT_EQ(6, h.median);
    EXPECT_FLOAT_EQ(0.5, h.stddev);
    EXPECT_FLOAT_EQ(5, h.mode);
    EXPECT_FLOAT_EQ(5, h.min);
    EXPECT_FLOAT_EQ(6, h.max);
    EXPECT_FLOAT_EQ(6, h.perc95th);
    EXPECT_FLOAT_EQ(0, h.binWidth);
    EXPECT_EQ(2, h.sampleSize);
    for (unsigned int b = 0; b < h.bins.size(); b++) EXPECT_EQ(0, h.bins[b]);
}

TEST(hist,mode)
{
    std::vector<float> data;
    for (unsigned int i = 0; i < 100; i++) data.push_back(10);
    for (unsigned int i = 0; i < 20; i++) data.push_back(i*2);
    Histogram h(data);

    EXPECT_FLOAT_EQ(0, h.first);
    EXPECT_FLOAT_EQ(38, h.last);
    EXPECT_FLOAT_EQ(11.5, h.mean);
    EXPECT_FLOAT_EQ(10, h.median);
    EXPECT_NEAR(5.78, h.stddev, 0.01);
    EXPECT_FLOAT_EQ(9.5, h.mode);
    EXPECT_FLOAT_EQ(0, h.min);
    EXPECT_FLOAT_EQ(38, h.max);
    EXPECT_FLOAT_EQ(28, h.perc95th);
    EXPECT_NEAR(1.26, h.binWidth, 0.01);
    EXPECT_EQ(120, h.sampleSize);

    Histogram h2(data, 1, 5);

    EXPECT_FLOAT_EQ(10.5, h2.mode);
}

TEST(hist,nanVal)
{
    std::vector<float> data = { std::numeric_limits<float>::quiet_NaN() };
    Histogram h(data);

    std::cerr << h << std::endl;
}

int Counts(const Histogram& h)
{
    int c = 0;
    for(auto x : h.bins) c+= x;
    return c;
}

TEST(hist,floatingPointRoundoff)
{
    typedef union { uint32_t u32; float d; } Float;
    Float min;
    Float max;
    Float binWidth;
    Float d;

    // This test case was discovered with SE-338. The data (d) was 1 ULP less than
    // the max value (numbers are the same except for the last bit). Due to round off,
    // the code used to throw an exception because the data appeared to be an outlier.
    // Now, the code will nudge the value into the top bin if the floating-point value is less than max.
    min.u32      = 0x4426a408;
    max.u32      = 0x454cc99a;
    binWidth.u32 = 0x42ae00a2;
    d.u32        = 0x454cc999;

    // first case where d is truly less than max, but previously, it threw an exception
    std::vector<float> input1;
    input1.push_back(d.d); // this one is not an outlier
    Histogram h1(input1, binWidth.d, min.d, 30, max.d);
    Float h1max;
    h1max.d = h1.max;

    EXPECT_EQ(min.d, h1.min); // don't use FLOAT_EQ, we want bit exact equal
    EXPECT_EQ(max.u32, h1max.u32);
    EXPECT_EQ(max.d, h1.max); // don't use FLOAT_EQ, we want bit exact equal
    EXPECT_EQ(binWidth.d, h1.binWidth); // don't use FLOAT_EQ, we want bit exact equal
    EXPECT_TRUE(std::isnan(h1.first));
    EXPECT_TRUE(std::isnan(h1.last)); // no outliers on the high side
    EXPECT_EQ(1, Counts(h1));

    // second case where d is max, so it is an outlier, because the interval is [min,max)
    std::vector<float> input2;
    input2.push_back(max.d); // this one is an outlier
    Histogram h2(input2, binWidth.d, min.d, 30, max.d);

    EXPECT_EQ(min.d, h2.min); // don't use FLOAT_EQ, we want bit exact equal
    EXPECT_EQ(max.d, h2.max); // don't use FLOAT_EQ, we want bit exact equal
    EXPECT_EQ(binWidth.d, h2.binWidth); // don't use FLOAT_EQ, we want bit exact equal
    EXPECT_TRUE(std::isnan(h2.first));
    EXPECT_FALSE(std::isnan(h2.last));
    EXPECT_EQ(0, Counts(h2));

//    printf("min %.20lg %lx\n",min.d, min.u32);
//    printf("max %.20lg %lx\n",max.d, max.u32);
//    printf("wid %.20lg %lx\n",binWidth.d, binWidth.u32);
//    printf("d   %.20lg %lx\n",d.d,d.u32);

//    double binHit0 = (d.d - min.d)/(binWidth.d);
//    unsigned int binHit = floor(binHit0);

//    printf("%lg %d\n",binHit0,binHit);

//    Float binWidth_;
//    binWidth_.d = (max.d - min.d)/30;
//    printf("binWidth_ %.20lg %lx\n",binWidth_.d, binWidth_.u32);
//    printf("binWidth  %.20lg %lx\n",binWidth.d , binWidth.u32);

//    Float diff1;
//    Float diff2;

//    diff1.d = max.d - min.d;
//    diff2.d = d.d   - min.d;
//    printf("diff1  %.20lg %lx\n",diff1.d , diff1.u32);
//    printf("dffi2  %.20lg %lx\n",diff2.d , diff2.u32);
}
