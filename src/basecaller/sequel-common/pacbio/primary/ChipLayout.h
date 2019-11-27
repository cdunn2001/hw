
// Copyright (c) 2014-2018, Pacific Biosciences of California, Inc.
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
// File Description:
/// \brief  a class used to describe the Sequel chip layout
//
// Programmer: Mark Lakata
//
// Overview:
//  The ChipLayout defines the unit cells. Most unit cells are sequencing ZMWs, but other unit cells are non
//  sequencing and they need to be filtered from the basecalling results.  Unit cell coordinates
//  use the Sequel instrument convention and use the terms "X" and "Y".  The instrument convention defines
//  the direction of increasing X and Y, where X increases going down the chip and Y increases going toward the right.
//  The origin is typically near the upper-left corner of the chip.  For Sequel, the origin was arbitrarily assigned to
//  the absolute upper-left of the Excel spreadsheet, and thus all values are non negative. For Spider, the origin was
//  assigned to the first sequencing ZMW near the upper-left, so there some negative values allowed for unit cells that
//  lie off of the active area.
//
//      ----> Y
//      |
//      |
//      |
//      V X
//
//  The ChipLayout class also defines the mapping between pixel space and unit cell space.  Pixel coordinates are
//  specified using the terms "row" and "column" (in that canonical order), and are
//  defined to be always non-negative, with (0,0) being the first pixel read out by the sensor. For Sequel, the
//  (0,0) pixel is in the upper-left, and for Spider, the (0,0) pixel is in the lower-left.
//  The values rowPixelsPerZmw_ and colPixelsPerZmw_ can be negative, which means that the pixel have the opposite
//  orientation from the unit cells.
#pragma once

#include <cassert>
#include <functional>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <pacbio/primary/SequelROI.h>
#include <pacbio/primary/UnitCell.h>
#include <pacbio/primary/Platform.h>
#include <pacbio/primary/ChipClass.h>
#include <pacbio/PBException.h>

namespace PacBio {
namespace Primary {

/// class that returns row and column offsets for an Excel cell label. For example,
/// "A1" results in row=0 and col=0, "AA1" results in row=0, col=26, "B1" results in row=1, col=0.
class Excel
{
public:
    static int GetRow(const char* excelLabel);
    static int GetCol(const char* excelLabel);
};
class ChipLayout
{
public:     // Types
    using UnitFeature = uint32_t;

public:     // Unit cell functional predicates.
    virtual bool IsSequencing(uint16_t holeX, uint16_t holeY) const
    { (void) holeX; (void) holeY; return true; }

    virtual bool IsPorSequencing(uint16_t holeX, uint16_t holeY) const
    { (void) holeX; (void) holeY; return true; }

    virtual bool IsLaserScatter(uint16_t holeX, uint16_t holeY) const
    { (void) holeX; (void) holeY; return false; }

    virtual bool IsLaserPower0p0x(uint16_t holeX, uint16_t holeY) const
    { (void) holeX; (void) holeY; return false; }

    virtual bool IsLaserPower0p5x(uint16_t holeX, uint16_t holeY) const
    { (void) holeX; (void) holeY; return false; }

    virtual bool IsLaserPower1p5x(uint16_t holeX, uint16_t holeY) const
    { (void) holeX; (void) holeY; return false; }

    virtual bool IsLaserPower2p0x(uint16_t holeX, uint16_t holeY) const
    { (void) holeX; (void) holeY; return false; }

    virtual bool IsApertureClosed(uint16_t holeX, uint16_t holeY) const
    { (void) holeX; (void) holeY; return false; }

    virtual bool IsApertureOpen(uint16_t holeX, uint16_t holeY) const
    { (void) holeX; (void) holeY; return false; }

    // TODO: Can this one be eliminated?
    virtual bool IsScatteringMetrology(uint16_t holeX, uint16_t holeY) const
    { (void) holeX; (void) holeY; return false; }

public:   // Unit cell type identifier.
    // The integer returned will typically be the value underlying the
    // appropriate enum value defined by the implementation.
    // Implementations will typically store the type identifiers in a sparse
    // 2d array where the default element value is the id for the standard
    // sequencing unit cell.
    virtual uint8_t UnitCellTypeId(uint16_t holeX, uint16_t holeY) const = 0;

protected:
    struct Params
    {
        int32_t minUnitCellX_;    ///! minimum legal unit cell coordinate, x (row) coordinate, inclusive
        int32_t minUnitCellY_;    ///! minimum legal unit cell coordinate, y (col) coordinate, inclusive
        int32_t maxUnitCellX_;    ///! maximum legal unit cell coordinate, x (row) coordinate, inclusive
        int32_t maxUnitCellY_;    ///! maximum legal unit cell coordinate, y (col) coordinate, inclusive
        int32_t rowPixelsPerZmw_; ///! this value can be negative, which implies that the pixel rows increase from the
        ///! "bottom" to the "top" of the chip.
        int32_t colPixelsPerZmw_; ///! this value can be negative, which implies that the pixel columns increase from
        ///! the "right" to the "left" of the chip
        int32_t unitCellOffsetToSensorX_; ///! the unit cell coordinate to the origin of the sensor (X aka row)
        int32_t unitCellOffsetToSensorY_; ///! the unit cell coordinate to the origin of the sensor (Y aka col)
    };
    Params params;

protected:
    ChipLayout(const Params& initializer);

public:
    virtual ~ChipLayout();
    virtual std::string Name() const = 0;

    /// Unit features are deprecated. Use predicates (e.g., IsSequencing) or
    /// unit type identifiers instead.
    virtual UnitFeature GetUnitCellFeatures(const UnitCell& uc) const = 0;
    virtual std::vector<std::pair<UnitFeature,UnitCell>> GetUnitCellFeatureList(const SequelROI& roi) const;

    // the filter map maps conventional frequency (0 is lowest frequency,i.e. red, 1 is higher frequency, i.e. green)
    // to pixel offset in the chip.
    // For Sequel alpha, the FilterMap is {1,0}, that is the red and green pixels must be swapped.
    virtual std::vector<uint16_t> FilterMap() const = 0;

    virtual SequelSensorROI GetSensorROI() const = 0;
    virtual ChipClass GetChipClass() const = 0;
    uint64_t GetNumUnitCells() const { return GetSensorROI().NumUnitCells(); }

    static bool IsSequencing(UnitFeature features)
    { return features == 0; }

    static bool HasFeature(UnitFeature features, UnitFeature feature)
    { return (features & feature) != 0; }

    std::vector<UnitCell> GetUnitCellList(const SequelROI& roi) const;

    SequelRectangularROI GetFullChipROI() const
    {
        return SequelRectangularROI(GetSensorROI());
    }

    SequelRectangularROI GetROIFromUnitCellRectangle(UnitX xMin, UnitY yMin, UnitX xHeight, UnitY yWidth);
    UnitX ConvertAbsoluteRowPixelToUnitCellX(RowPixels x) const;
    UnitY ConvertAbsoluteColPixelToUnitCellY(ColPixels y) const;
    UnitX ConvertRelativeRowPixelToUnitCellX(RowPixels x) const;
    UnitY ConvertRelativeColPixelToUnitCellY(ColPixels y) const;
    UnitCell ConvertPixelCoordToUnitCell(PixelCoord pc) const
    {
        UnitCell uc(ConvertAbsoluteRowPixelToUnitCellX(pc.row),
                    ConvertAbsoluteColPixelToUnitCellY(pc.col));
        return uc;
    }

    PixelCoord ConvertUnitCellToPixelCoord(const UnitCell& uc) const
    {
        return PixelCoord(ConvertUnitCellToRowPixels(uc),ConvertUnitCellToColPixels(uc));
    }

    virtual int GetUnitCellIntType(const UnitCell& cell) const = 0;

public:
    std::vector<UnitCell> GetUnitCellListByPredicate(std::function<bool(const UnitCell&)> predicate, const SequelROI& roi) const;

    RowPixels ConvertUnitCellToRowPixels(const UnitCell& uc) const
    {
        int32_t rowPixels = (uc.x - params.unitCellOffsetToSensorX_) * params.rowPixelsPerZmw_;
        if (rowPixels < 0)
        {
            std::stringstream ss;
            ss << "Can't covert UnitCellX=" << uc.x
               << " to pixel coordinates.  Conversion resulted in negative pixel row:" << rowPixels;
            throw PBException(ss.str());
        }
        return RowPixels(static_cast<uint32_t>(rowPixels));
    }

    ColPixels ConvertUnitCellToColPixels(const UnitCell& uc) const
    {
        int32_t colPixels = (uc.y - params.unitCellOffsetToSensorY_) * params.colPixelsPerZmw_;
        if (colPixels < 0)
        {
            std::stringstream ss;
            ss << "Can't covert UnitCellY= " << uc.y
               << " to pixel coordinates. Conversion resulted in negative pixel col:" << colPixels;
            throw PBException(ss.str());
        }
        return ColPixels(static_cast<uint32_t>(colPixels));
    }

public:     // Static functions
    static const std::vector<std::string>& GetLayoutNames();

    // TODO: Should this point to `const ChipLayout`?
    static std::unique_ptr<ChipLayout> Factory(const std::string& layoutName);

    static SequelSensorROI GetSensorROI(Platform platform);

public:     // Parameters of the transform between unit cell and pixel coordinates.
    int32_t RowPixelsPerZmw() const { return params.rowPixelsPerZmw_; }

    int32_t ColPixelsPerZmw() const { return params.colPixelsPerZmw_; }

    int32_t PixelsPerZmw() const { return RowPixelsPerZmw() * ColPixelsPerZmw(); }

protected:
    std::vector<UnitCell> GetUnitCellListByIntType(int unitType, const SequelROI& roi);
    std::vector<std::pair<int,UnitCell>> GetUnitCellIntTypeList(const SequelROI& roi) const;
};

class ChipLayoutFile : public ChipLayout
{
public:
    ChipLayoutFile(const std::string& gzfilename);
};

}}  // PacBio::Primary
