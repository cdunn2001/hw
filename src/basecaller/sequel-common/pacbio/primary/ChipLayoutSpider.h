#ifndef Sequel_Common_PacBio_Primary_ChipLayoutSpider_H_
#define Sequel_Common_PacBio_Primary_ChipLayoutSpider_H_

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
//  Defines class ChipLayoutSpider.

#include "ChipLayout.h"

namespace PacBio {
namespace Primary {

/// An intermediate class that defines things that are common to all Spider
/// chip layouts.
class ChipLayoutSpider : public ChipLayout
{
public:     // Types
    /// Unit features are deprecated. Use predicates (e.g., IsSequencing) or
    /// unit type identifiers instead.
    enum SpiderUnitFeature : UnitFeature
    {
        StandardZMW         = 0,
        NonStandardZMW      = 1UL << 0,
        NonSequencing       = 1UL << 1,

        // Combinations
        Sequencing          = StandardZMW | NonStandardZMW
    };

public:     // Structors
    ChipLayoutSpider(const Params& p);
    virtual ~ChipLayoutSpider();

public:     // Functions
    ChipClass GetChipClass() const override
    { return ChipClass::Spider; }

    std::vector<uint16_t> FilterMap() const override
    { return std::vector<uint16_t>{0}; }

    SequelSensorROI GetSensorROI() const override
    { return SequelSensorROI::Spider(); }
};


}}  // PacBio::Primary

#endif // Sequel_Common_PacBio_Primary_ChipLayoutSpider_H_
