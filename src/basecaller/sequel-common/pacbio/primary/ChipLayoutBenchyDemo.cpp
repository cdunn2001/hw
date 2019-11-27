
// Copyright (c) 2019, Pacific Biosciences of California, Inc.
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
//  Defines members of class ChipLayoutBenchyDemo.

#include <boost/numeric/conversion/cast.hpp>
#include "ChipLayoutBenchyDemo.h"

#include "Spider_1p0_NTO.h"

namespace PacBio {
namespace Primary {

ChipLayoutBenchyDemo::ChipLayoutBenchyDemo()
    : ChipLayoutBenchy([]()
    {
        Params r;
        r.minUnitCellX_ = MinUnitCellX;
        r.minUnitCellY_ = MinUnitCellY;
        r.maxUnitCellX_ = MaxUnitCellX;
        r.maxUnitCellY_ = MaxUnitCellY;
        r.rowPixelsPerZmw_ = 1;
        r.colPixelsPerZmw_ = 1;
        r.unitCellOffsetToSensorX_  = UnitCellOffsetToSensorX;
        r.unitCellOffsetToSensorY_  = UnitCellOffsetToSensorY;
        return r;
    }())
{
    // Account for moving origin to (0,0).
    auto xes_converted = std::unique_ptr<int16_t[]>(new int16_t[Spider_1p0_NTO::numNonZmws]);
    auto yes_converted = std::unique_ptr<int16_t[]>(new int16_t[Spider_1p0_NTO::numNonZmws]);

    for (size_t i = 0; i < Spider_1p0_NTO::numNonZmws; i++)
    {
        using boost::numeric_cast;
        xes_converted[i] = numeric_cast<int16_t>(Spider_1p0_NTO::xes[i] - 1);
        yes_converted[i] = numeric_cast<int16_t>(Spider_1p0_NTO::yes[i] - 1);
    }

    unitCellTypes_.CompressSparseArray(Spider_1p0_NTO::numNonZmws,
                                       xes_converted.get(),
                                       yes_converted.get(),
                                       Spider_1p0_NTO::zes,1);
}


ChipLayout::UnitFeature
ChipLayoutBenchyDemo::GetUnitCellFeatures(const UnitCell& uc) const
{
    UnitCellType type = GetUnitCellType(uc);
    BenchyUnitFeature features;
    if (type == UnitCellType::Sequencing) features = StandardZMW;
    else if (IsNonStdSequencing(type)) features = NonStandardZMW;
    else features = NonSequencing;
    return features;
}


}}  // PacBio::Primary
