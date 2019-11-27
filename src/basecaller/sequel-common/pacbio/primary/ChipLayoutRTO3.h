// Copyright (c) 2017-2018, Pacific Biosciences of California, Inc.
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
/// \brief   Chip Layout definition for RTO3 of the Sequel sensor chip
///

#ifndef SEQUELACQUISITION_CHIPLAYOUTRTO3_H
#define SEQUELACQUISITION_CHIPLAYOUTRTO3_H

#include "ChipLayoutSequel.h"

namespace PacBio {
namespace Primary {

class ChipLayoutRTO3 : public ChipLayoutSequel
{
public:
    enum class UnitCellType
    {
        Sequencing = 1,

        SideDummies = 3,
        AntiZMW = 4,
        AntiAperture = 5,

        PowerTaps = 12,
        PowerTaps_nonpixel = 13,
        PowerLines = 14,
        LeadsInMetalFree = 15,
        PowerTaps_nonpixel_nometal = 16,
        CalibPad_RedFilter = 17,
        CalibPad_GreenFilter = 18,

        LeadsIn_MetalOn = 22,
        FLTI = 23,
        FLTI_nopixel = 24,
        ScatteringMetrology = 25,

        CalibPad_O_field = 27,
        CalibPad_O_edge = 28,
        CalibPad_X_field = 29,
        CalibPad_X_edge = 30,
        CalibPad_X_RedFilter = 31,
        CalibPad_X_GreenFilter = 32
    };

    ChipLayoutRTO3();

    static std::string ClassName()
    { return "SequEL_4.0_RTO3"; }

    std::string Name() const override
    { return ClassName(); }

    UnitCellType GetUnitCellType(const UnitCell& cell) const
    { return static_cast<UnitCellType>(GetUnitCellIntType_Common(cell)); }

    static const SequelUnitFeature featureLookup[33];

    UnitFeature GetUnitCellFeatures(const UnitCell& uc) const override
    { return GetUnitCellFeaturesInline(uc.x, uc.y); }

public:   // Unit cell functional predicates.
    virtual bool IsPorSequencing(uint16_t holeX, uint16_t holeY) const override
    { return GetUnitCellTypeInline(holeX, holeY) == UnitCellType::Sequencing; }

    virtual bool IsSequencing(uint16_t holeX, uint16_t holeY) const override
    { return GetUnitCellTypeInline(holeX, holeY) == UnitCellType::Sequencing; }

    virtual bool IsLaserScatter(uint16_t holeX, uint16_t holeY) const override
    {
        // TODO: This definition is a guess based on names.
        return GetUnitCellTypeInline(holeX, holeY) == UnitCellType::ScatteringMetrology;
    }

    virtual bool IsApertureClosed(uint16_t holeX, uint16_t holeY) const override
    {
        const auto f = GetUnitCellFeaturesInline(holeX, holeY);
        return (f & Fiducial) && (f & NoZMW) && (f & Aperture3Closed);
    }

    virtual bool IsApertureOpen(uint16_t holeX, uint16_t holeY) const override
    {
        const auto f = GetUnitCellFeaturesInline(holeX, holeY);
        return (f & Fiducial) && (f & NoZMW) && !(f & Aperture3Closed);
    }

    virtual bool IsScatteringMetrology(uint16_t holeX, uint16_t holeY) const override
    { return IsLaserScatter(holeX, holeY); }

private:    // Non-virtual inline functions
    UnitCellType GetUnitCellTypeInline(uint16_t x, uint16_t y) const
    { return static_cast<UnitCellType>(GetUnitCellIntType_Common(x, y)); }

    UnitFeature GetUnitCellFeaturesInline(uint16_t holeX, uint16_t holeY) const
    {
        UnitCellType type = GetUnitCellTypeInline(holeX, holeY);
        uint32_t itype = static_cast<uint32_t>(type);
        if (itype < 33) return featureLookup[itype];
        return NotUsed;
    }
};

}}  // PacBio::Primary

#endif //SEQUELACQUISITION_CHIPLAYOUTRTO3_H
