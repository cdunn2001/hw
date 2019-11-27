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
/// \brief   Chip Layout definition for RTO2 of the Sequel sensor chip
///

#ifndef SEQUELACQUISITION_CHIPLAYOUTRTO2_H
#define SEQUELACQUISITION_CHIPLAYOUTRTO2_H

#include "ChipLayoutSequel.h"

namespace PacBio {
namespace Primary {

class ChipLayoutRTO2 : public ChipLayoutSequel
{
public:
    enum class UnitCellType
    {
        Sequencing = 1,
        Sequencing_with_color_filter_flipped_LR = 2,
        SideDummies = 3,
        AntiZMW = 4,
        AntiAperture = 5,
        FDZMW = 6, // not used in Sequel
        IsoCell_on = 7, // not used in Sequel
        IsoCell_off = 8, // not used in Sequel
        AntiLens_on = 9, // not used in Sequel
        AntiLens_off = 10, // not used in Sequel
        BlackHoles = 11, // not used in Sequel
        PowerTaps = 12,
        PowerTaps_nonpixel = 13,
        powerlines = 14,
        leads_in_metal_free = 15,
        PowerTaps_nonpixel_nometal = 16,
        CalibPad_RedFilter = 17,
        CalibPad_GreenFilter = 18,
        CalibPad_Dark = 19,
        CalibPad_NoAperture = 20,
        CalibPad_off = 21,
        LeadsIn_MetalOn = 22,
        FLTI = 23,
        FLTI_nopixel = 24,
        ScatteringMetrology = 25
    };

    static const std::string UnitTypeName(unsigned int index)
    {
        static const std::string UnitTypeNames[] = {
            "SEQUENCING",
            "SEQUENCING_COLOR_FILTER_FLIPPED_LR",
            "SIDEDUMMIES",
            "ANTIZMW",
            "ANTIAPERTURE",
            "FDZMW",
            "ISOCELL_ON",
            "ISOCELL_OFF",
            "ANTILENS_ON",
            "ANTILENS_OFF",
            "BLACKHOLES",
            "POWERTAPS",
            "POWERTAPS_NONPIXELS",
            "POWERLINES",
            "LEADS_METAL_FREE",
            "POWERTAPS_NONPIXEL_NOMETAL",
            "CALIBPAD_REDFILTER",
            "CALIBPAD_GREENFILTER",
            "CALIBPAD_DARK",
            "CALIBPAD_NOAPERTURE",
            "CALIBPAD_OFF",
            "LEADS_METALON",
            "FLTI",
            "FLTI_NOPIXEL",
            "SCATTERING_METROLOGY"
        };
        return UnitTypeNames[index];
    }

    ChipLayoutRTO2();

    static std::string ClassName()
    { return "SequEL_4.0_RTO2"; }

    std::string Name() const override
    { return ClassName(); }

    UnitCellType GetUnitCellType(const UnitCell& cell) const
    { return static_cast<UnitCellType>(GetUnitCellIntType_Common(cell)); }

    UnitCellType GetUnitCellType(uint16_t x, uint16_t y) const
    { return static_cast<UnitCellType>(GetUnitCellIntType_Common(x, y)); }

    UnitFeature GetUnitCellFeatures(const UnitCell& uc) const override;

public:   // Unit cell functional predicates.
    virtual bool IsPorSequencing(uint16_t holeX, uint16_t holeY) const override
    { return GetUnitCellType(holeX, holeY) == UnitCellType::Sequencing; }

    virtual bool IsSequencing(uint16_t holeX, uint16_t holeY) const override
    { return GetUnitCellType(holeX, holeY) == UnitCellType::Sequencing; }

    virtual bool IsLaserScatter(uint16_t holeX, uint16_t holeY) const override
    { return GetUnitCellType(holeX, holeY) == UnitCellType::ScatteringMetrology; }

    virtual bool IsApertureClosed(uint16_t holeX, uint16_t holeY) const override
    { return GetUnitCellType(holeX, holeY) == UnitCellType::CalibPad_NoAperture; }

    virtual bool IsApertureOpen(uint16_t holeX, uint16_t holeY) const override
    {
        const auto uct = GetUnitCellType(holeX, holeY);
        return uct == UnitCellType::CalibPad_Dark
                || uct == UnitCellType::CalibPad_off;
    }

    virtual bool IsScatteringMetrology(uint16_t holeX, uint16_t holeY) const override
    { return IsLaserScatter(holeX, holeY); }
};

inline ChipLayoutSequel::SequelUnitFeature&
operator|=(ChipLayoutSequel::SequelUnitFeature& a, ChipLayoutSequel::SequelUnitFeature b)
{ return a = static_cast<ChipLayoutSequel::SequelUnitFeature>(a | b); }

}}  // PacBio::Primary

#endif //SEQUELACQUISITION_CHIPLAYOUTRTO2_H
