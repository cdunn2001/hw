#ifndef Sequel_Common_PacBio_Primary_ChipLayoutSequel_H_
#define Sequel_Common_PacBio_Primary_ChipLayoutSequel_H_

// Copyright (c) 2018, Pacific Biosciences of California, Inc.
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
//  Defines class ChipLayoutSequel.

#include "ChipLayout.h"

#include <type_traits>

namespace PacBio {
namespace Primary {

/// An intermediate class that defines things that are common over all Sequel
/// chip layouts.
class ChipLayoutSequel : public ChipLayout
{
public:     // Types
    /// Unit features are deprecated. Use predicates (e.g., IsSequencing) or
    /// unit type identifiers instead.
    /// Note that there is no flag bit for standard ZMW.
    enum SequelUnitFeature : UnitFeature
    {
        NoFlags                 = 0,
        StandardZMW             = 0,
        NoZMW                   = 1UL<<0,
        WaveguideClosed         = 1UL<<3,
        ColorFilterAllRed       = 1UL<<4,
        Aperture1GreenClosed    = 1UL<<5,
        Aperture1RedClosed      = 1UL<<6,
        Aperture2Closed         = 1UL<<7,
        Aperture3Closed         = 1UL<<8,
        DontUseForLaserPowerFeedback = 1UL<<9,
        ColorJustGreen          = 1UL<<11,
        Power                   = 1UL<<12,
        Fiducial                = 1UL<<13,
        InactiveArea            = 1UL<<14,
        NotUsed                 = 1UL<<31,

        // Combinations
        Aperture1Closed         = Aperture1GreenClosed | Aperture1RedClosed
    };

public:     // Static constants
    static const int32_t MinUnitCellX = 0;
    static const int32_t MinUnitCellY = 0;
    static const int32_t MaxUnitCellY = 1088; // columns
    static const int32_t MaxUnitCellX = 1208; // rows
    static const int32_t UnitCellOffsetToSensorX = 32; // rows
    static const int32_t UnitCellOffsetToSensorY = 32; // cols

public:     // Structors
    ChipLayoutSequel();
    virtual ~ChipLayoutSequel();

public:     // Functions
    int GetUnitCellIntType(const UnitCell& cell) const override
    { return GetUnitCellIntType_Common(cell); }

    std::vector<uint16_t> FilterMap() const override
    { return std::vector<uint16_t>{ 1 ,0 }; }

    SequelSensorROI GetSensorROI() const override
    { return SequelSensorROI::SequelAlpha(); }

    ChipClass GetChipClass() const override
    { return ChipClass::Sequel; }

    /// Type identifier of the unit cell at (holeX, holeY).
    /// \returns a plain integer that can be cast to SequelUnitType.
    virtual uint8_t UnitCellTypeId(uint16_t holeX, uint16_t holeY) const override
    { return GetUnitCellIntType_Common(holeX, holeY); }

    /// Load the layout data from \a file.
    void Load(std::istream& file);

protected:  // Non-virtual inline functions
    uint8_t GetUnitCellIntType_Common(uint16_t x, uint16_t y) const
    {
        if (x < params.maxUnitCellX_ && y < params.maxUnitCellY_) return FullMatrix(x, y);
        throw PBException("out of range ");
    }

    int GetUnitCellIntType_Common(const UnitCell& cell) const
    { return GetUnitCellIntType_Common(cell.x, cell.y); }

protected:  // Functions for CRU-ing the layout data.
    void InitFullMatrix();

    /// \returns the unit cell "type" at (x,y). Note that not all derived classes use this method.
    uint8_t& FullMatrix(int32_t x, int32_t y)
    {
        assert(fullMatrix_ != nullptr);
        return fullMatrix_[x * params.maxUnitCellY_ + y];
    }

    uint8_t FullMatrix(int32_t x, int32_t y) const
    {
        assert(fullMatrix_ != nullptr);
        return fullMatrix_[x * params.maxUnitCellY_ + y];
    }

private:    // Data
    uint8_t* fullMatrix_ = nullptr;
};


// Bitwise operators for ChipLayoutSequel::SequelUnitFeature.
inline ChipLayoutSequel::SequelUnitFeature
operator|(ChipLayoutSequel::SequelUnitFeature lhs, ChipLayoutSequel::SequelUnitFeature rhs)
{
    using IntType = std::underlying_type<ChipLayoutSequel::SequelUnitFeature>::type;
    const auto a = static_cast<IntType>(lhs);
    const auto b = static_cast<IntType>(rhs);
    return static_cast<ChipLayoutSequel::SequelUnitFeature>(a | b);
}

//inline ChipLayoutSequel::SequelUnitFeature&
//operator|=(ChipLayoutSequel::SequelUnitFeature& a, ChipLayoutSequel::SequelUnitFeature b)
//{ return a = static_cast<ChipLayoutSequel::SequelUnitFeature>(a | b); }

}}  // PacBio::Primary

#endif // Sequel_Common_PacBio_Primary_ChipLayoutSequel_H_
