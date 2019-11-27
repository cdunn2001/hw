// Copyright (c) 2017-2019, Pacific Biosciences of California, Inc.
//
// All rights reserved.
//
// THIS SOFTWARE CONSTITUTES AND EMBODIES PACIFIC BIOSCIENCES' CONFIDENTIAL
// AND PROPRIETARY INFORMATION.
//
// Disclosure, redistribution and use of this software is subject to the
// terms and conditions of the applicable written agreement(s) between you
// and Pacific Biosciences, where "you" refers to you or your company or
// organization, as applicable.  Any other disclosure, redistribution or
// use is prohibited.
//
// THIS SOFTWARE IS PROVIDED BY PACIFIC BIOSCIENCES AND ITS CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL PACIFIC BIOSCIENCES OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
// OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
// ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//  Description:
/// \brief   Chip Layout definition for first Benchy sensor chip (name of
/// layout may change, this is a placeholder)

#ifndef SEQUELACQUISITION_CHIPLAYOUTBenchyDemo_H
#define SEQUELACQUISITION_CHIPLAYOUTBenchyDemo_H

#include <pacbio/primary/ChipLayoutBenchy.h>
#include <pacbio/primary/Sparse2DArray.h>

namespace PacBio {
namespace Primary {

class ChipLayoutBenchyDemo : public ChipLayoutBenchy
{
public:
    static constexpr int32_t NumRows = 1360;
    static constexpr int32_t NumCols = 1440;
    static const int32_t  MinUnitCellX = -78; // row
    static const int32_t  MinUnitCellY = -78; // column
    static const int32_t  MaxUnitCellX = (NumRows+77); // row
    static const int32_t  MaxUnitCellY = (NumCols+77); // column
    static const int32_t  UnitCellOffsetToSensorX = 0; // rows
    static const int32_t  UnitCellOffsetToSensorY = 0; // cols
    static const int32_t  BorderSize = 78; // row or col

    static_assert(MinUnitCellX == -BorderSize, "consistency between ChipLayout and pixel dimensions");
    static_assert(MinUnitCellY == -BorderSize, "consistency between ChipLayout and pixel dimensions");
#if 0
    static_assert(MaxUnitCellX == PacBio::Primary::BenchyLayout::maxPixelRows + BorderSize - 1, "consistency between ChipLayout and pixel dimensions");
    static_assert(MaxUnitCellY == PacBio::Primary::BenchyLayout::maxPixelCols + BorderSize - 1, "consistency between ChipLayout and pixel dimensions");
#endif

public:
    enum class UnitCellType
    {
        Unknown                = 0,
        Sequencing             = 1,
        SideDummies            = 3,
        AntiZMW                = 4,
        LeadsInMetalPresent    = 5,
        LeadsInMetalNotPreset  = 6,
        HighSNRDark            = 7,
        HighSNRBright          = 8,
        AntiZMWAsymmetricMark  = 9,
        LaserScattering        = 10,
        IrradianceSplit0p5xDimmer = 11,
        IrradianceSplit2XBrighter = 12,
        VernierActive          = 13,
        VernierDark            = 14,
        HighSNRBrightAP1_2200_AP2_2400_AP3_3000      = 15,
        HighSNRBrightAP1_3000_AP2_3000_AP3_3000Dense = 16,
        HighSNRBrightAP1_2200_AP2_2400_AP3_3000Dense = 17,
        IrradianceSplit0XDark  = 18,
        IrradianceSplit1p5XBrighter = 19,
        CPW_NEG_BRIGHT         = 100,
        CPW_POS_BRIGHT         = 101,
        CELL_H_NEG_BRIGHT      = 102,
        CELL_H_POS_BRIGHT      = 103,
        NO_DEEP_NPS_RIGHT      = 104,
        REDUCED_PD_SIZE_BRIGHT = 105,
        NODTI_BRIGHT           = 106,
        CROSSTALKWITHDTI_DARK  = 107,
        CPW_NEG_DARK           = 110,
        CPW_POS_DARK           = 111,
        CELL_H_NEG_DARK        = 112,
        CELL_H_POS_DARK        = 113,
        NO_DEEP_NPS_DARK       = 114,
        REDUCED_PD_SIZE_DARK   = 115,
        NODTI_DARK             = 116,
        CROSSTALKWITHDTI_BRIGHT= 117
    };

public:     // Static functions
    static std::string ClassName()
    { return "BenchyDemo1"; }

    /// Is the unit cell type a functionally sequencing but non-standard one?
    static bool IsNonStdSequencing(UnitCellType typeId)
    {
        // The type ids considered "sequencing". Must be sorted!
        static const int seqTypes[]
                = {8, 10, 11, 12, 13, 15, 16, 17, 18, 19,
                   100, 101, 102, 103, 104, 105, 106, 117};
        assert(std::is_sorted(std::begin(seqTypes), std::end(seqTypes)));
        const auto t = static_cast<int>(typeId);
        return std::binary_search(std::begin(seqTypes), std::end(seqTypes), t);
        // TODO: Might be able to accelerate this by exploiting contiguous
        // ranges in seqTypes.
        // Sure would be nice if all the sequencing type identifiers were in
        // one contiguous range!
    }

public:     // Structors
    ChipLayoutBenchyDemo();

    virtual ~ChipLayoutBenchyDemo()
    { }

public:     // Functions
    std::string Name() const override
    { return ClassName(); }

    UnitCellType GetUnitCellType(const UnitCell& cell) const
    { return GetUnitCellType(cell.x, cell.y); }

    int GetUnitCellIntType(const UnitCell& cell) const override
    {
        uint8_t itype = unitCellTypes_.Value(cell.x, cell.y);
        return static_cast<int>(itype);
    }

    UnitFeature GetUnitCellFeatures(const UnitCell& uc) const override;

    virtual uint8_t UnitCellTypeId(uint16_t x, uint16_t y) const
    { return unitCellTypes_.Value(x, y); }

    SequelSensorROI GetSensorROI() const override
    {
        auto roi = SequelSensorROI(0, 0,
                                   NumRows, NumCols,
                                   Spider::numPixelRowsPerZmw, Spider::numPixelColsPerZmw);
        roi.SetPixelLaneWidth(Tile::NumPixels);
        return roi;
    }

public:     // Unit cell functional predicates.
    virtual bool IsSequencing(uint16_t x, uint16_t y) const override
    {
        const auto uct = GetUnitCellType(x, y);
        return uct == UnitCellType::Sequencing || IsNonStdSequencing(uct);
    }

    virtual bool IsPorSequencing(uint16_t x, uint16_t y) const override
    { return GetUnitCellType(x, y) == UnitCellType::Sequencing; }

    virtual bool IsLaserScatter(uint16_t x, uint16_t y) const override
    { return GetUnitCellType(x, y) == UnitCellType::LaserScattering; }

    virtual bool IsLaserPower0p0x(uint16_t x, uint16_t y) const override
    { return GetUnitCellType(x, y) == UnitCellType::IrradianceSplit0XDark; }

    virtual bool IsLaserPower0p5x(uint16_t x, uint16_t y) const override
    { return GetUnitCellType(x, y) == UnitCellType::IrradianceSplit0p5xDimmer; }

    virtual bool IsLaserPower1p5x(uint16_t x, uint16_t y) const override
    { return GetUnitCellType(x, y) == UnitCellType::IrradianceSplit1p5XBrighter; }

    virtual bool IsLaserPower2p0x(uint16_t x, uint16_t y) const override
    { return GetUnitCellType(x, y) == UnitCellType::IrradianceSplit2XBrighter; }

private:    // Non-virtual inline functions
    UnitCellType GetUnitCellType(uint16_t holeX, uint16_t holeY) const
    {
        const auto uct = unitCellTypes_.Value(holeX , holeY);
        return static_cast<UnitCellType>(uct);
    }

private:
    Sparse2DArray<uint8_t> unitCellTypes_;
};

}}  // PacBio::Primary

#endif //SEQUELACQUISITION_CHIPLAYOUTRTO3_H
