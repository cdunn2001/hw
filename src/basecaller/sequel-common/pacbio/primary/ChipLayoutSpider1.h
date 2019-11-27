// Copyright (c) 2017=2018, Pacific Biosciences of California, Inc.
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
/// \brief   Chip Layout definition for first Spider sensor chip (name of
/// layout may change, this is a placeholder)

#ifndef SEQUELACQUISITION_CHIPLAYOUTSpider1_H
#define SEQUELACQUISITION_CHIPLAYOUTSpider1_H

#include <pacbio/primary/ChipLayoutSpider.h>

namespace PacBio {
namespace Primary {

class ChipLayoutSpider1 : public ChipLayoutSpider
{
public:
    static const int32_t MinUnitCellX = 0;
    static const int32_t MinUnitCellY = 0;
    static const int32_t MaxUnitCellY = Spider::maxPixelCols; // fixme
    static const int32_t MaxUnitCellX = Spider::maxPixelRows; // fixme
    static const int32_t UnitCellOffsetToSensorX = 0; // rows fixme
    static const int32_t UnitCellOffsetToSensorY = 0; // cols fixme

public:

    enum class UnitCellType
    {
        Sequencing = 1
    };

public:     // Structors
    ChipLayoutSpider1();

    virtual ~ChipLayoutSpider1()
    { }

public:     // Functions
    static std::string ClassName()
    { return "ChipLayoutSpider1"; }

    std::string Name() const override
    { return ClassName(); }

    UnitCellType GetUnitCellType(const UnitCell& /*cell*/) const
    { return UnitCellType::Sequencing; }

    int GetUnitCellIntType(const UnitCell& /*cell*/) const override
    { return static_cast<int>(UnitCellType::Sequencing); }

    UnitFeature GetUnitCellFeatures(const UnitCell& /*cell*/) const override
    { return StandardZMW; }

    SequelSensorROI GetSensorROI() const override
    {
        auto roi = SequelSensorROI::Spider();
        roi.SetPixelLaneWidth(1); // disable check for alignment
        return roi;
    }

    virtual uint8_t UnitCellTypeId(uint16_t holeX, uint16_t holeY) const
    { (void) holeX; (void) holeY; return static_cast<uint8_t>(UnitCellType::Sequencing); }

    // No need to override any of the default definitions of the unit-cell
    // functional predicates in the base class ChipLayout.
};

}}  // PacBio::Primary

#endif //SEQUELACQUISITION_CHIPLAYOUTRTO3_H
