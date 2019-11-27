
#include <algorithm>


#include "gtest/gtest.h"

#include <pacbio/dev/gtest-extras.h>
#include <pacbio/primary/ChipLayout.h>
#include <pacbio/primary/ChipLayoutRTO2.h>
#include <pacbio/primary/ChipLayoutRTO3.h>
#include <pacbio/primary/ChipLayoutSpider1.h>
#include <pacbio/primary/ChipLayoutSpider1p0NTO.h>
#include <pacbio/primary/SequelROI.h>
#include <pacbio/primary/Sparse2DArray.h>
#include <pacbio/primary/Spider_1p0_NTO.h>

using namespace std;

#include <stdexcept>

using namespace PacBio::Primary;

TEST(ChipLayoutRTO2,Constructor)
{
    ChipLayoutRTO2 layout;
    EXPECT_EQ(layout.GetUnitCellType(UnitCell(0,0)), ChipLayoutRTO2::UnitCellType::powerlines);
    EXPECT_EQ(layout.GetUnitCellType(UnitCell(32,32)),ChipLayoutRTO2::UnitCellType::SideDummies);
    EXPECT_EQ(layout.GetUnitCellType(UnitCell(63,63)),ChipLayoutRTO2::UnitCellType::PowerTaps_nonpixel);
    EXPECT_EQ(layout.GetUnitCellType(UnitCell(63,64)),ChipLayoutRTO2::UnitCellType::LeadsIn_MetalOn);
    EXPECT_EQ(layout.GetUnitCellType(UnitCell(64,62)),ChipLayoutRTO2::UnitCellType::SideDummies);
    EXPECT_EQ(layout.GetUnitCellType(UnitCell(64,63)),ChipLayoutRTO2::UnitCellType::PowerTaps_nonpixel);
    EXPECT_EQ(layout.GetUnitCellType(UnitCell(64,64)),ChipLayoutRTO2::UnitCellType::Sequencing);
    EXPECT_EQ(layout.GetUnitCellType(UnitCell(1207,1023+64)),ChipLayoutRTO2::UnitCellType::powerlines);

    // Tests for virtual UnitCellTypeId.
    ChipLayout& cl = layout;
    auto expect = static_cast<uint8_t>(ChipLayoutRTO2::UnitCellType::powerlines);
    EXPECT_EQ(expect, cl.UnitCellTypeId(0, 0));
    expect = static_cast<uint8_t>(ChipLayoutRTO2::UnitCellType::SideDummies);
    EXPECT_EQ(expect, cl.UnitCellTypeId(32, 32));
    expect = static_cast<uint8_t>(ChipLayoutRTO2::UnitCellType::LeadsIn_MetalOn);
    EXPECT_EQ(expect, cl.UnitCellTypeId(63, 64));
    expect = static_cast<uint8_t>(ChipLayoutRTO2::UnitCellType::PowerTaps_nonpixel);
    EXPECT_EQ(expect, cl.UnitCellTypeId(64, 63));
    expect = static_cast<uint8_t>(ChipLayoutRTO2::UnitCellType::Sequencing);
    EXPECT_EQ(expect, cl.UnitCellTypeId(64, 64));
}

TEST(ExcelCoord,Test)
{
    EXPECT_EQ(1,Excel::GetCol("A"));
    EXPECT_EQ(1,Excel::GetCol("A1"));
    EXPECT_EQ(2,Excel::GetCol("B1"));
    EXPECT_EQ(27,Excel::GetCol("AA1"));
    EXPECT_EQ(28,Excel::GetCol("AB1"));
    EXPECT_EQ(26+26*26,Excel::GetCol("ZZ1"));
    EXPECT_EQ(1+26+26*26,Excel::GetCol("AAA1"));
    EXPECT_EQ(326,Excel::GetCol("LN113"));

    EXPECT_EQ(1,Excel::GetRow("A1"));
    EXPECT_EQ(1,Excel::GetRow("1"));
    EXPECT_EQ(666,Excel::GetRow("A666"));
    EXPECT_EQ(666,Excel::GetRow("AA666"));
    EXPECT_EQ(666,Excel::GetRow("ABC666"));
    EXPECT_EQ(113,Excel::GetRow("LN113"));

    EXPECT_THROW(Excel::GetCol(""), std::runtime_error);
    EXPECT_THROW(Excel::GetCol("1"), std::runtime_error);
    EXPECT_THROW(Excel::GetRow(""), std::runtime_error);
    EXPECT_THROW(Excel::GetRow("A"), std::runtime_error);
}

TEST(ChipLayoutRTO2,ExcelCoordinates)
{
    ChipLayoutRTO2 layout;

    UnitCell x;
    x.x = 0;
    x.y = 0;
    EXPECT_EQ("A1",x.ExcelCell());
    x.x = 1;
    x.y = 0;
    EXPECT_EQ("A2",x.ExcelCell());
    x.x = 0;
    x.y = 1;
    EXPECT_EQ("B1",x.ExcelCell());
    x.x = 0;
    x.y = 26;
    EXPECT_EQ("AA1",x.ExcelCell());
    x.x = 0;
    x.y = ChipLayoutRTO2::MaxUnitCellY-1;
    EXPECT_EQ("AOV1",x.ExcelCell());
    x.x = ChipLayoutRTO2::MaxUnitCellX-1;
    x.y = ChipLayoutRTO2::MaxUnitCellY-1;
    EXPECT_EQ("AOV1208",x.ExcelCell());
    x.x = ChipLayoutRTO2::MaxUnitCellX-1;
    x.y = 0;
    EXPECT_EQ("A1208",x.ExcelCell());

    UnitCell x1(layout.ConvertAbsoluteRowPixelToUnitCellX(0),
                layout.ConvertAbsoluteColPixelToUnitCellY(0));
    EXPECT_EQ("AG33",x1.ExcelCell());
    EXPECT_EQ(0,layout.ConvertUnitCellToColPixels(x1).Value());
    EXPECT_EQ(0,layout.ConvertUnitCellToRowPixels(x1).Value());

    UnitCell x2(layout.ConvertAbsoluteRowPixelToUnitCellX(1),
                layout.ConvertAbsoluteColPixelToUnitCellY(0));
    EXPECT_EQ("AG34",x2.ExcelCell());
    EXPECT_EQ(0,layout.ConvertUnitCellToColPixels(x2).Value());
    EXPECT_EQ(1,layout.ConvertUnitCellToRowPixels(x2).Value());

    UnitCell x3(layout.ConvertAbsoluteRowPixelToUnitCellX(0),
                layout.ConvertAbsoluteColPixelToUnitCellY(2));
    EXPECT_EQ("AH33",x3.ExcelCell());
    EXPECT_EQ(2,layout.ConvertUnitCellToColPixels(x3).Value());
    EXPECT_EQ(0,layout.ConvertUnitCellToRowPixels(x3).Value());

    UnitCell x4(layout.ConvertAbsoluteRowPixelToUnitCellX(1),
                layout.ConvertAbsoluteColPixelToUnitCellY(2));
    EXPECT_EQ("AH34",x4.ExcelCell());
    EXPECT_EQ(2,layout.ConvertUnitCellToColPixels(x4).Value());
    EXPECT_EQ(1,layout.ConvertUnitCellToRowPixels(x4).Value());

    EXPECT_EQ(32+13, layout.ConvertAbsoluteRowPixelToUnitCellX(13).Value());
    EXPECT_EQ(32+6, layout.ConvertAbsoluteColPixelToUnitCellY(13).Value());
    EXPECT_EQ(32+7, layout.ConvertAbsoluteColPixelToUnitCellY(14).Value());

    UnitCell a1("A1");
    EXPECT_EQ(0,a1.y);
    EXPECT_EQ(0,a1.x);

    UnitCell a2("A2");
    EXPECT_EQ(0,a2.y);
    EXPECT_EQ(1,a2.x);

    UnitCell aa1("AA1");
    EXPECT_EQ(26,aa1.y);
    EXPECT_EQ(0,aa1.x);

    UnitCell b2("B2");
    EXPECT_EQ(1,b2.y);
    EXPECT_EQ(1,b2.x);

    EXPECT_EQ(Sequel::maxPixelRows * Sequel::maxPixelCols/Sequel::numPixelColsPerZmw, layout.GetNumUnitCells());
}


TEST(ChipLayoutRTO3,Constructor)
{
    ChipLayoutRTO3 layout;
    EXPECT_EQ(1171456,layout.GetSensorROI().NumUnitCells());

    EXPECT_EQ(ChipLayoutRTO3::UnitCellType::PowerLines,         layout.GetUnitCellType(UnitCell(0, 0)));
    EXPECT_EQ(ChipLayoutRTO3::UnitCellType::SideDummies,        layout.GetUnitCellType(UnitCell(32, 32)));
    EXPECT_EQ(ChipLayoutRTO3::UnitCellType::PowerTaps_nonpixel, layout.GetUnitCellType(UnitCell(63, 63)));
    EXPECT_EQ(ChipLayoutRTO3::UnitCellType::LeadsIn_MetalOn,    layout.GetUnitCellType(UnitCell(63, 64)));
    EXPECT_EQ(ChipLayoutRTO3::UnitCellType::SideDummies,        layout.GetUnitCellType(UnitCell(64, 62)));
    EXPECT_EQ(ChipLayoutRTO3::UnitCellType::PowerTaps_nonpixel, layout.GetUnitCellType(UnitCell(64, 63)));
    EXPECT_EQ(ChipLayoutRTO3::UnitCellType::Sequencing,         layout.GetUnitCellType(UnitCell(64, 64)));
    EXPECT_EQ(ChipLayoutRTO3::UnitCellType::PowerLines,         layout.GetUnitCellType(UnitCell(1207, 1023 + 64)));

    EXPECT_EQ(ChipLayoutRTO3::UnitCellType::AntiZMW             ,layout.GetUnitCellType(UnitCell("GJ128")));
    EXPECT_EQ(ChipLayoutRTO3::UnitCellType::AntiAperture        ,layout.GetUnitCellType(UnitCell("GZ144")));

    EXPECT_EQ(ChipLayoutRTO3::UnitCellType::CalibPad_O_field    ,layout.GetUnitCellType(UnitCell("LN113")));
    EXPECT_EQ(ChipLayoutRTO3::UnitCellType::CalibPad_O_edge     ,layout.GetUnitCellType(UnitCell("LN124")));
    EXPECT_EQ(ChipLayoutRTO3::UnitCellType::CalibPad_RedFilter  ,layout.GetUnitCellType(UnitCell("LQ120")));
    EXPECT_EQ(ChipLayoutRTO3::UnitCellType::CalibPad_GreenFilter,layout.GetUnitCellType(UnitCell("LU120")));

    // Tests for virtual UnitCellTypeId.
    ChipLayout& cl = layout;
    auto expect = static_cast<uint8_t>(ChipLayoutRTO3::UnitCellType::PowerLines);
    EXPECT_EQ(expect, cl.UnitCellTypeId(0, 0));
    expect = static_cast<uint8_t>(ChipLayoutRTO3::UnitCellType::SideDummies);
    EXPECT_EQ(expect, cl.UnitCellTypeId(32, 32));
    expect = static_cast<uint8_t>(ChipLayoutRTO3::UnitCellType::LeadsIn_MetalOn);
    EXPECT_EQ(expect, cl.UnitCellTypeId(63, 64));
    expect = static_cast<uint8_t>(ChipLayoutRTO3::UnitCellType::PowerTaps_nonpixel);
    EXPECT_EQ(expect, cl.UnitCellTypeId(64, 63));
    expect = static_cast<uint8_t>(ChipLayoutRTO3::UnitCellType::Sequencing);
    EXPECT_EQ(expect, cl.UnitCellTypeId(64, 64));
}


TEST(ChipLayoutRTO3,Features)
{
    ChipLayoutRTO3 layout;

    EXPECT_TRUE(ChipLayout::HasFeature(layout.GetUnitCellFeatures(UnitCell(0, 0)), ChipLayoutSequel::Power)); // PowerLines
    EXPECT_TRUE(ChipLayout::HasFeature(layout.GetUnitCellFeatures(UnitCell(32, 32)), ChipLayoutSequel::InactiveArea)); // SideDummies
    EXPECT_TRUE(ChipLayout::HasFeature(layout.GetUnitCellFeatures(UnitCell(63, 63)), ChipLayoutSequel::Power)); // PowerTaps_nonpixel
    EXPECT_TRUE(ChipLayout::HasFeature(layout.GetUnitCellFeatures(UnitCell(63, 64)), ChipLayoutSequel::Power)); // LeadsIn_MetalOn
    EXPECT_TRUE(ChipLayout::HasFeature(layout.GetUnitCellFeatures(UnitCell(64, 62)), ChipLayoutSequel::InactiveArea)); // SideDummies
    EXPECT_TRUE(ChipLayout::HasFeature(layout.GetUnitCellFeatures(UnitCell(64, 63)), ChipLayoutSequel::Power)); // PowerTaps_nonpixel
    EXPECT_TRUE(ChipLayout::IsSequencing(layout.GetUnitCellFeatures(UnitCell(64, 64)))); // PowerLines
    EXPECT_TRUE(ChipLayout::HasFeature(layout.GetUnitCellFeatures(UnitCell(1207, 1023 + 64)), ChipLayoutSequel::Power)); // PowerLines

    EXPECT_TRUE(ChipLayout::HasFeature(layout.GetUnitCellFeatures(UnitCell("GJ128")), ChipLayoutSequel::NoZMW)); // AntiZMW
    EXPECT_TRUE(ChipLayout::HasFeature(layout.GetUnitCellFeatures(UnitCell("GZ144")), ChipLayoutSequel::Aperture2Closed)); // AntiAperture

    EXPECT_TRUE(ChipLayout::HasFeature(layout.GetUnitCellFeatures(UnitCell("LN113")), ChipLayoutSequel::Fiducial)); // CalibPad_O_field
    EXPECT_TRUE(ChipLayout::HasFeature(layout.GetUnitCellFeatures(UnitCell("LN124")), ChipLayoutSequel::Fiducial)); // CalibPad_O_edge
    EXPECT_TRUE(ChipLayout::HasFeature(layout.GetUnitCellFeatures(UnitCell("LQ120")), ChipLayoutSequel::Fiducial)); // CalibPad_RedFilter
    EXPECT_TRUE(ChipLayout::HasFeature(layout.GetUnitCellFeatures(UnitCell("LU120")), ChipLayoutSequel::Fiducial)); // CalibPad_GreenFilter
}

TEST(ChipLayoutSpider1,Features)
{
    ChipLayoutSpider1 layout;

    EXPECT_TRUE(ChipLayout::IsSequencing(layout.GetUnitCellFeatures(UnitCell(0, 0))));
    EXPECT_EQ(8025472,layout.GetSensorROI().NumUnitCells());
}


TEST(ChipLayout,PlatformSensorROI)
{
    auto roi1 = ChipLayout::GetSensorROI(Platform::Sequel1PAC1);

    EXPECT_EQ(Sequel::maxPixelRows, roi1.PhysicalRows());

    auto roi2 = ChipLayout::GetSensorROI(Platform::Spider);
    EXPECT_EQ(Spider::maxPixelRows, roi2.PhysicalRows());

    EXPECT_THROW(ChipLayout::GetSensorROI(Platform::UNKNOWN),std::exception);
}


TEST(ChipLayoutSpider1_1p0_NTO,Lookup)
{
    Sparse2DArray<uint8_t> sa;
    sa.CompressSparseArray(PacBio::Primary::Spider_1p0_NTO::numNonZmws,
                           PacBio::Primary::Spider_1p0_NTO::xes,
                           PacBio::Primary::Spider_1p0_NTO::yes,
                           PacBio::Primary::Spider_1p0_NTO::zes,1);

    auto translate = [&sa](int16_t x, int16_t y)
    {
        return sa.Value(static_cast<int16_t>(x - 78), static_cast<int16_t>(y - 78));
    };

    EXPECT_EQ(6,translate(1,1));
    EXPECT_EQ(6,translate(1,2));
    EXPECT_EQ(3,translate(2,1));
    EXPECT_EQ(3,translate(2,2));

    EXPECT_EQ(5,translate(77,79));
    EXPECT_EQ(5,translate(78,79));
    EXPECT_EQ(1,translate(79,79));
    EXPECT_EQ(3,translate(78,78));
    EXPECT_EQ(3,translate(79,78));

    EXPECT_EQ(3,translate(100,78));
    EXPECT_EQ(1,translate(100,79));

    //space invader bullets
    EXPECT_EQ(1,translate(2613,2589));
    EXPECT_EQ(9,translate(2614,2589));
    EXPECT_EQ(9,translate(2615,2589));
    EXPECT_EQ(1,translate(2616,2589));
    EXPECT_EQ(1,translate(2617,2589));
    EXPECT_EQ(9,translate(2618,2589));
    EXPECT_EQ(9,translate(2619,2589));
    EXPECT_EQ(1,translate(2620,2589));
}

TEST(ChipLayoutSpider1_1p0_NTO,Features)
{
    ChipLayoutSpider1p0NTO layout;

    EXPECT_EQ(8025472,layout.GetSensorROI().NumUnitCells());

    EXPECT_EQ(ChipLayoutSpider1p0NTO::UnitCellType::LeadsInMetalNotPreset,   layout.GetUnitCellType(UnitCell(-78, -78)));
    EXPECT_EQ(ChipLayoutSpider1p0NTO::UnitCellType::Sequencing,              layout.GetUnitCellType(UnitCell(0,0)));

    EXPECT_FALSE(ChipLayout::IsSequencing(layout.GetUnitCellFeatures(UnitCell(-78,-78))));
    EXPECT_TRUE( ChipLayout::IsSequencing(layout.GetUnitCellFeatures(UnitCell(0, 0))));

}

TEST(ChipLayoutSpider_1p0_NTO, Mapping)
{
    ChipLayoutSpider1p0NTO layout;

    const uint16_t samples[] = {0, 1, 42, 1000};
    for (auto x : samples)
    {
        for (auto y : samples)
        {
            UnitCell uc (x, y);
            PixelCoord pc = layout.ConvertUnitCellToPixelCoord(uc);
            EXPECT_EQ(ColPixels(y), pc.col);
            EXPECT_EQ(RowPixels(x), pc.row);

            // Round trip
            UnitCell rt = layout.ConvertPixelCoordToUnitCell(pc);
            EXPECT_EQ(x, rt.x);
            EXPECT_EQ(y, rt.y);
        }
    }

    // Transform an ROI.
    SequelRectangularROI roi = layout.GetROIFromUnitCellRectangle(0,0,2 /*rows*/,32 /*cols*/);
    EXPECT_EQ(RowPixels(0), roi.AbsoluteRowPixelMin()) << roi;
    EXPECT_EQ(RowPixels(2), roi.AbsoluteRowPixelMax()) << roi;
    EXPECT_EQ(ColPixels(0), roi.AbsoluteColPixelMin()) << roi << " " << layout.GetSensorROI();
    EXPECT_EQ(ColPixels(32), roi.AbsoluteColPixelMax()) << roi << " " << layout.GetSensorROI();

    std::vector<UnitCell> andBackAgain = layout.GetUnitCellList(roi);
    std::sort(andBackAgain.begin(),andBackAgain.end(),[](const UnitCell& a, const UnitCell& b){return a.x < b.x || (a.x==b.x && a.y < b.y);});

    ASSERT_EQ(2*32, andBackAgain.size())  << " " << layout.GetSensorROI();
    EXPECT_EQ(0,andBackAgain[0].x)  << " " << layout.GetSensorROI();
    EXPECT_EQ(0,andBackAgain[0].y)  << " " << layout.GetSensorROI();
    EXPECT_EQ(1,andBackAgain[2*32-1].x)  << " " << layout.GetSensorROI();
    EXPECT_EQ(31,andBackAgain[2*32-1].y)  << " " << layout.GetSensorROI();

#if 0
    // this tests a Unit Cell ROI that has a non-conforming column width (not modulo 32).
    // It is not supported and the test will fail.
    SequelRectangularROI roi = layout.GetROIFromUnitCellRectangle(1,1,2 /*rows*/,3 /*cols*/);
    EXPECT_EQ(RowPixels(PacBio::Primary::Spider::maxPixelRows-2),roi.AbsoluteRowPixelMin()) << roi;
    EXPECT_EQ(RowPixels(PacBio::Primary::Spider::maxPixelRows  ),roi.AbsoluteRowPixelMax()) << roi;
    EXPECT_EQ(ColPixels(0),roi.AbsoluteColPixelMin()) << roi << " " << layout.GetSensorROI();
    EXPECT_EQ(ColPixels(3),roi.AbsoluteColPixelMax()) << roi << " " << layout.GetSensorROI();

    std::vector<UnitCell> andBackAgain = layout.GetUnitCellList(roi);

    TEST_COUT << "before sort:\n";
    for(const auto& uc: andBackAgain)
    {
        TEST_COUT << std::hex << uc.Number() << std::dec << std::endl;
    }

    std::sort(andBackAgain.begin(),andBackAgain.end(),[](const UnitCell& a, const UnitCell& b){return a.x < b.x || (a.x==b.x && a.y < b.y);});

    TEST_COUT << "after sort:\n";
    for(const auto& uc: andBackAgain)
    {
        TEST_COUT << std::hex << uc.Number() << std::dec <<std::endl;
    }

    EXPECT_EQ(1,andBackAgain[0].x)  << " " << layout.GetSensorROI();
    EXPECT_EQ(1,andBackAgain[0].y)  << " " << layout.GetSensorROI();
    EXPECT_EQ(2*3, andBackAgain.size())  << " " << layout.GetSensorROI();
#endif
}

TEST(ChipLayoutSpider1_1p0_NTO,DISABLED_Dump)
{
    ChipLayoutSpider1p0NTO layout;
    SequelRectangularROI roi(RowPixels(0),ColPixels(0),RowPixels(128),ColPixels(128), SequelSensorROI::Spider());
    UnitCell uclast;
    std::cout << "Type:\n";
    for(const auto& uc : layout.GetUnitCellList(roi))
    {
        if (uc.x != uclast.x) std::cout << "\n";
        int itype = layout.GetUnitCellIntType(uc);
        std::cout << std::setw(3) << itype;
        uclast = uc;
    }
    std::cout << "\n\n";

    std::cout << "X:\n";
    for(const auto& uc : layout.GetUnitCellList(roi))
    {
        if (uc.x != uclast.x) std::cout << "\n";
        std::cout << std::setw(5) << uc.x;
        uclast = uc;
    }
    std::cout << "\n\n";

    std::cout << "Y:\n";
    for(const auto& uc : layout.GetUnitCellList(roi))
    {
        if (uc.x != uclast.x) std::cout << "\n";
        std::cout << std::setw(4) << uc.y;
        uclast = uc;
    }
    std::cout << "\n\n";

}

TEST(ChipLayoutSpider1_1p0_NTO,IsAPI)
{
    std::unique_ptr<ChipLayout> layout(ChipLayout::Factory("Spider_1p0_NTO"));

    bool (PacBio::Primary::ChipLayout::*Filter)(uint16_t, uint16_t) const = &ChipLayout::IsSequencing;
    EXPECT_TRUE((layout.get()->*Filter)(0,0));

    bool (PacBio::Primary::ChipLayout::*Filter2)(uint16_t, uint16_t) const = &ChipLayout::IsApertureClosed;
    EXPECT_FALSE((layout.get()->*Filter2)(0,0));

    bool (PacBio::Primary::ChipLayout::*Filter3)(uint16_t, uint16_t) const = &ChipLayout::IsApertureOpen;
    EXPECT_FALSE((layout.get()->*Filter3)(0,0));

    // Random spot checks for the various types of unit cells.
    uint16_t x = 2372;
    uint16_t y = 2761;
    EXPECT_EQ(1, layout->UnitCellTypeId(x, y));
    EXPECT_TRUE(layout->IsPorSequencing(x, y));
    EXPECT_TRUE(layout->IsSequencing(x, y));
    EXPECT_FALSE(layout->IsLaserScatter(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p0x(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower1p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower2p0x(x, y));
    EXPECT_FALSE(layout->IsApertureClosed(x, y));
    EXPECT_FALSE(layout->IsApertureOpen(x, y));

    x = 1339; y = 2976;
    EXPECT_EQ(3, layout->UnitCellTypeId(x, y));
    EXPECT_FALSE(layout->IsPorSequencing(x, y));
    EXPECT_FALSE(layout->IsSequencing(x, y));
    EXPECT_FALSE(layout->IsLaserScatter(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p0x(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower1p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower2p0x(x, y));
    EXPECT_FALSE(layout->IsApertureClosed(x, y));
    EXPECT_FALSE(layout->IsApertureOpen(x, y));

    x = 918; y = 1940;
    EXPECT_EQ(4, layout->UnitCellTypeId(x, y));
    EXPECT_FALSE(layout->IsPorSequencing(x, y));
    EXPECT_FALSE(layout->IsSequencing(x, y));
    EXPECT_FALSE(layout->IsLaserScatter(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p0x(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower1p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower2p0x(x, y));
    EXPECT_FALSE(layout->IsApertureClosed(x, y));
    EXPECT_FALSE(layout->IsApertureOpen(x, y));

    x = 2830; y = 810;
    EXPECT_EQ(5, layout->UnitCellTypeId(x, y));
    EXPECT_FALSE(layout->IsPorSequencing(x, y));
    EXPECT_FALSE(layout->IsSequencing(x, y));
    EXPECT_FALSE(layout->IsLaserScatter(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p0x(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower1p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower2p0x(x, y));
    EXPECT_FALSE(layout->IsApertureClosed(x, y));
    EXPECT_FALSE(layout->IsApertureOpen(x, y));

    x = 2833; y = 1599;
    EXPECT_EQ(6, layout->UnitCellTypeId(x, y));
    EXPECT_FALSE(layout->IsPorSequencing(x, y));
    EXPECT_FALSE(layout->IsSequencing(x, y));
    EXPECT_FALSE(layout->IsLaserScatter(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p0x(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower1p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower2p0x(x, y));
    EXPECT_FALSE(layout->IsApertureClosed(x, y));
    EXPECT_FALSE(layout->IsApertureOpen(x, y));

    x = 2043; y = 2290;
    EXPECT_EQ(7, layout->UnitCellTypeId(x, y));
    EXPECT_FALSE(layout->IsPorSequencing(x, y));
    EXPECT_FALSE(layout->IsSequencing(x, y));
    EXPECT_FALSE(layout->IsLaserScatter(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p0x(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower1p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower2p0x(x, y));
    EXPECT_FALSE(layout->IsApertureClosed(x, y));
    EXPECT_FALSE(layout->IsApertureOpen(x, y));

    x = 641; y = 695;
    EXPECT_EQ(8, layout->UnitCellTypeId(x, y));
    EXPECT_FALSE(layout->IsPorSequencing(x, y));
    EXPECT_TRUE(layout->IsSequencing(x, y));
    EXPECT_FALSE(layout->IsLaserScatter(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p0x(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower1p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower2p0x(x, y));
    EXPECT_FALSE(layout->IsApertureClosed(x, y));
    EXPECT_FALSE(layout->IsApertureOpen(x, y));

    x = 2517; y = 2506;
    EXPECT_EQ(9, layout->UnitCellTypeId(x, y));
    EXPECT_FALSE(layout->IsPorSequencing(x, y));
    EXPECT_FALSE(layout->IsSequencing(x, y));
    EXPECT_FALSE(layout->IsLaserScatter(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p0x(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower1p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower2p0x(x, y));
    EXPECT_FALSE(layout->IsApertureClosed(x, y));
    EXPECT_FALSE(layout->IsApertureOpen(x, y));

    x = 2; y = 2731;
    EXPECT_EQ(10, layout->UnitCellTypeId(x, y));
    EXPECT_FALSE(layout->IsPorSequencing(x, y));
    EXPECT_TRUE(layout->IsSequencing(x, y));
    EXPECT_TRUE(layout->IsLaserScatter(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p0x(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower1p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower2p0x(x, y));
    EXPECT_FALSE(layout->IsApertureClosed(x, y));
    EXPECT_FALSE(layout->IsApertureOpen(x, y));

    x = 553; y = 2117;
    EXPECT_EQ(11, layout->UnitCellTypeId(x, y));
    EXPECT_FALSE(layout->IsPorSequencing(x, y));
    EXPECT_TRUE(layout->IsSequencing(x, y));
    EXPECT_FALSE(layout->IsLaserScatter(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p0x(x, y));
    EXPECT_TRUE(layout->IsLaserPower0p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower1p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower2p0x(x, y));
    EXPECT_FALSE(layout->IsApertureClosed(x, y));
    EXPECT_FALSE(layout->IsApertureOpen(x, y));

    x = 1305; y = 807;
    EXPECT_EQ(12, layout->UnitCellTypeId(x, y));
    EXPECT_FALSE(layout->IsPorSequencing(x, y));
    EXPECT_TRUE(layout->IsSequencing(x, y));
    EXPECT_FALSE(layout->IsLaserScatter(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p0x(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower1p5x(x, y));
    EXPECT_TRUE(layout->IsLaserPower2p0x(x, y));
    EXPECT_FALSE(layout->IsApertureClosed(x, y));
    EXPECT_FALSE(layout->IsApertureOpen(x, y));

    x = 2578; y = 43;
    EXPECT_EQ(13, layout->UnitCellTypeId(x, y));
    EXPECT_FALSE(layout->IsPorSequencing(x, y));
    EXPECT_TRUE(layout->IsSequencing(x, y));
    EXPECT_FALSE(layout->IsLaserScatter(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p0x(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower1p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower2p0x(x, y));
    EXPECT_FALSE(layout->IsApertureClosed(x, y));
    EXPECT_FALSE(layout->IsApertureOpen(x, y));

    x = 2591; y = 45;
    EXPECT_EQ(14, layout->UnitCellTypeId(x, y));
    EXPECT_FALSE(layout->IsPorSequencing(x, y));
    EXPECT_FALSE(layout->IsSequencing(x, y));
    EXPECT_FALSE(layout->IsLaserScatter(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p0x(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower1p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower2p0x(x, y));
    EXPECT_FALSE(layout->IsApertureClosed(x, y));
    EXPECT_FALSE(layout->IsApertureOpen(x, y));

    x = 2050; y = 2290;
    EXPECT_EQ(15, layout->UnitCellTypeId(x, y));
    EXPECT_FALSE(layout->IsPorSequencing(x, y));
    EXPECT_TRUE(layout->IsSequencing(x, y));
    EXPECT_FALSE(layout->IsLaserScatter(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p0x(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower1p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower2p0x(x, y));
    EXPECT_FALSE(layout->IsApertureClosed(x, y));
    EXPECT_FALSE(layout->IsApertureOpen(x, y));

    x = 691; y = 678;
    EXPECT_EQ(16, layout->UnitCellTypeId(x, y));
    EXPECT_FALSE(layout->IsPorSequencing(x, y));
    EXPECT_TRUE(layout->IsSequencing(x, y));
    EXPECT_FALSE(layout->IsLaserScatter(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p0x(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower1p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower2p0x(x, y));
    EXPECT_FALSE(layout->IsApertureClosed(x, y));
    EXPECT_FALSE(layout->IsApertureOpen(x, y));

    x = 2098; y = 2294;
    EXPECT_EQ(17, layout->UnitCellTypeId(x, y));
    EXPECT_FALSE(layout->IsPorSequencing(x, y));
    EXPECT_TRUE(layout->IsSequencing(x, y));
    EXPECT_FALSE(layout->IsLaserScatter(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p0x(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower1p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower2p0x(x, y));
    EXPECT_FALSE(layout->IsApertureClosed(x, y));
    EXPECT_FALSE(layout->IsApertureOpen(x, y));

    x = 1716; y = 2120;
    EXPECT_EQ(18, layout->UnitCellTypeId(x, y));
    EXPECT_FALSE(layout->IsPorSequencing(x, y));
    EXPECT_TRUE(layout->IsSequencing(x, y));
    EXPECT_FALSE(layout->IsLaserScatter(x, y));
    EXPECT_TRUE(layout->IsLaserPower0p0x(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower1p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower2p0x(x, y));
    EXPECT_FALSE(layout->IsApertureClosed(x, y));
    EXPECT_FALSE(layout->IsApertureOpen(x, y));

    x = 196; y = 829;
    EXPECT_EQ(19, layout->UnitCellTypeId(x, y));
    EXPECT_FALSE(layout->IsPorSequencing(x, y));
    EXPECT_TRUE(layout->IsSequencing(x, y));
    EXPECT_FALSE(layout->IsLaserScatter(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p0x(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p5x(x, y));
    EXPECT_TRUE(layout->IsLaserPower1p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower2p0x(x, y));
    EXPECT_FALSE(layout->IsApertureClosed(x, y));
    EXPECT_FALSE(layout->IsApertureOpen(x, y));

    x = 2637; y = 123;
    EXPECT_EQ(100, layout->UnitCellTypeId(x, y));
    EXPECT_FALSE(layout->IsPorSequencing(x, y));
    EXPECT_TRUE(layout->IsSequencing(x, y));
    EXPECT_FALSE(layout->IsLaserScatter(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p0x(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower1p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower2p0x(x, y));
    EXPECT_FALSE(layout->IsApertureClosed(x, y));
    EXPECT_FALSE(layout->IsApertureOpen(x, y));

    x = 2658; y = 189;
    EXPECT_EQ(101, layout->UnitCellTypeId(x, y));
    EXPECT_FALSE(layout->IsPorSequencing(x, y));
    EXPECT_TRUE(layout->IsSequencing(x, y));
    EXPECT_FALSE(layout->IsLaserScatter(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p0x(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower1p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower2p0x(x, y));
    EXPECT_FALSE(layout->IsApertureClosed(x, y));
    EXPECT_FALSE(layout->IsApertureOpen(x, y));

    x = 2659; y = 245;
    EXPECT_EQ(102, layout->UnitCellTypeId(x, y));
    EXPECT_FALSE(layout->IsPorSequencing(x, y));
    EXPECT_TRUE(layout->IsSequencing(x, y));
    EXPECT_FALSE(layout->IsLaserScatter(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p0x(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower1p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower2p0x(x, y));
    EXPECT_FALSE(layout->IsApertureClosed(x, y));
    EXPECT_FALSE(layout->IsApertureOpen(x, y));

    x = 2641; y = 312;
    EXPECT_EQ(103, layout->UnitCellTypeId(x, y));
    EXPECT_FALSE(layout->IsPorSequencing(x, y));
    EXPECT_TRUE(layout->IsSequencing(x, y));
    EXPECT_FALSE(layout->IsLaserScatter(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p0x(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower1p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower2p0x(x, y));
    EXPECT_FALSE(layout->IsApertureClosed(x, y));
    EXPECT_FALSE(layout->IsApertureOpen(x, y));

    x = 2595; y = 119;
    EXPECT_EQ(104, layout->UnitCellTypeId(x, y));
    EXPECT_FALSE(layout->IsPorSequencing(x, y));
    EXPECT_TRUE(layout->IsSequencing(x, y));
    EXPECT_FALSE(layout->IsLaserScatter(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p0x(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower1p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower2p0x(x, y));
    EXPECT_FALSE(layout->IsApertureClosed(x, y));
    EXPECT_FALSE(layout->IsApertureOpen(x, y));

    x = 2594; y = 171;
    EXPECT_EQ(105, layout->UnitCellTypeId(x, y));
    EXPECT_FALSE(layout->IsPorSequencing(x, y));
    EXPECT_TRUE(layout->IsSequencing(x, y));
    EXPECT_FALSE(layout->IsLaserScatter(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p0x(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower1p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower2p0x(x, y));
    EXPECT_FALSE(layout->IsApertureClosed(x, y));
    EXPECT_FALSE(layout->IsApertureOpen(x, y));

    x = 2580; y = 245;
    EXPECT_EQ(106, layout->UnitCellTypeId(x, y));
    EXPECT_FALSE(layout->IsPorSequencing(x, y));
    EXPECT_TRUE(layout->IsSequencing(x, y));
    EXPECT_FALSE(layout->IsLaserScatter(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p0x(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower1p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower2p0x(x, y));
    EXPECT_FALSE(layout->IsApertureClosed(x, y));
    EXPECT_FALSE(layout->IsApertureOpen(x, y));

    x = 2579; y = 293;
    EXPECT_EQ(107, layout->UnitCellTypeId(x, y));
    EXPECT_FALSE(layout->IsPorSequencing(x, y));
    EXPECT_FALSE(layout->IsSequencing(x, y));
    EXPECT_FALSE(layout->IsLaserScatter(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p0x(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower1p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower2p0x(x, y));
    EXPECT_FALSE(layout->IsApertureClosed(x, y));
    EXPECT_FALSE(layout->IsApertureOpen(x, y));

    x = 2644; y = 112;
    EXPECT_EQ(110, layout->UnitCellTypeId(x, y));
    EXPECT_FALSE(layout->IsPorSequencing(x, y));
    EXPECT_FALSE(layout->IsSequencing(x, y));
    EXPECT_FALSE(layout->IsLaserScatter(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p0x(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower1p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower2p0x(x, y));
    EXPECT_FALSE(layout->IsApertureClosed(x, y));
    EXPECT_FALSE(layout->IsApertureOpen(x, y));

    x = 2642; y = 173;
    EXPECT_EQ(111, layout->UnitCellTypeId(x, y));
    EXPECT_FALSE(layout->IsPorSequencing(x, y));
    EXPECT_FALSE(layout->IsSequencing(x, y));
    EXPECT_FALSE(layout->IsLaserScatter(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p0x(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower1p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower2p0x(x, y));
    EXPECT_FALSE(layout->IsApertureClosed(x, y));
    EXPECT_FALSE(layout->IsApertureOpen(x, y));

    x = 2643; y = 238;
    EXPECT_EQ(112, layout->UnitCellTypeId(x, y));
    EXPECT_FALSE(layout->IsPorSequencing(x, y));
    EXPECT_FALSE(layout->IsSequencing(x, y));
    EXPECT_FALSE(layout->IsLaserScatter(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p0x(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower1p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower2p0x(x, y));
    EXPECT_FALSE(layout->IsApertureClosed(x, y));
    EXPECT_FALSE(layout->IsApertureOpen(x, y));

    x = 2643; y = 301;
    EXPECT_EQ(113, layout->UnitCellTypeId(x, y));
    EXPECT_FALSE(layout->IsPorSequencing(x, y));
    EXPECT_FALSE(layout->IsSequencing(x, y));
    EXPECT_FALSE(layout->IsLaserScatter(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p0x(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower1p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower2p0x(x, y));
    EXPECT_FALSE(layout->IsApertureClosed(x, y));
    EXPECT_FALSE(layout->IsApertureOpen(x, y));

    x = 2580; y = 109;
    EXPECT_EQ(114, layout->UnitCellTypeId(x, y));
    EXPECT_FALSE(layout->IsPorSequencing(x, y));
    EXPECT_FALSE(layout->IsSequencing(x, y));
    EXPECT_FALSE(layout->IsLaserScatter(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p0x(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower1p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower2p0x(x, y));
    EXPECT_FALSE(layout->IsApertureClosed(x, y));
    EXPECT_FALSE(layout->IsApertureOpen(x, y));

    x = 2582; y = 177;
    EXPECT_EQ(115, layout->UnitCellTypeId(x, y));
    EXPECT_FALSE(layout->IsPorSequencing(x, y));
    EXPECT_FALSE(layout->IsSequencing(x, y));
    EXPECT_FALSE(layout->IsLaserScatter(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p0x(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower1p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower2p0x(x, y));
    EXPECT_FALSE(layout->IsApertureClosed(x, y));
    EXPECT_FALSE(layout->IsApertureOpen(x, y));

    x = 2583; y = 240;
    EXPECT_EQ(116, layout->UnitCellTypeId(x, y));
    EXPECT_FALSE(layout->IsPorSequencing(x, y));
    EXPECT_FALSE(layout->IsSequencing(x, y));
    EXPECT_FALSE(layout->IsLaserScatter(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p0x(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower1p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower2p0x(x, y));
    EXPECT_FALSE(layout->IsApertureClosed(x, y));
    EXPECT_FALSE(layout->IsApertureOpen(x, y));

    x = 2567; y = 300;
    EXPECT_EQ(117, layout->UnitCellTypeId(x, y));
    EXPECT_FALSE(layout->IsPorSequencing(x, y));
    EXPECT_TRUE(layout->IsSequencing(x, y));
    EXPECT_FALSE(layout->IsLaserScatter(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p0x(x, y));
    EXPECT_FALSE(layout->IsLaserPower0p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower1p5x(x, y));
    EXPECT_FALSE(layout->IsLaserPower2p0x(x, y));
    EXPECT_FALSE(layout->IsApertureClosed(x, y));
    EXPECT_FALSE(layout->IsApertureOpen(x, y));
}

TEST(ChipLayoutBenchyDemo,IsAPI)
{
    std::unique_ptr <ChipLayout> layout(ChipLayout::Factory("BenchyDemo1"));

    bool (PacBio::Primary::ChipLayout::*Filter)(uint16_t, uint16_t) const = &ChipLayout::IsSequencing;

    EXPECT_TRUE((layout.get()->*Filter)(0,0));

    bool (PacBio::Primary::ChipLayout::*Filter2)(uint16_t, uint16_t) const = &ChipLayout::IsApertureClosed;

    EXPECT_FALSE((layout.get()->*Filter2)(0,0));

    bool (PacBio::Primary::ChipLayout::*Filter3)(uint16_t, uint16_t) const = &ChipLayout::IsApertureOpen;

    EXPECT_FALSE((layout.get()->*Filter3)(0,0));
}

