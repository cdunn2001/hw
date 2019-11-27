
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
//  Defines members of class ChipLayoutRTO2.


#include "ChipLayoutRTO2.h"

#include <pacbio/text/imemstream.h>

extern unsigned int  _home_UNIXHOME_mlakata_Matrices_Full_csv_gz_len;
extern unsigned char _home_UNIXHOME_mlakata_Matrices_Full_csv_gz[];

namespace PacBio {
namespace Primary {

ChipLayoutRTO2::ChipLayoutRTO2()
{
    InitFullMatrix();
    // reads stream from internal .TEXT segment (which is gzip compressed)
    PacBio::Text::imemstream buf((const char*) _home_UNIXHOME_mlakata_Matrices_Full_csv_gz,
                                 (size_t) _home_UNIXHOME_mlakata_Matrices_Full_csv_gz_len);
    Load(buf);
}


ChipLayout::UnitFeature ChipLayoutRTO2::GetUnitCellFeatures(const UnitCell& uc) const
{
    UnitCellType type = GetUnitCellType(uc);
    SequelUnitFeature features = NoFlags;
    if (type == UnitCellType::Sequencing)
    {
        features |= StandardZMW;
    }
    else
    {
        features |= DontUseForLaserPowerFeedback;
        if (type == UnitCellType::powerlines ||
                type == UnitCellType::SideDummies ||
                type == UnitCellType::PowerTaps_nonpixel ||
                type == UnitCellType::leads_in_metal_free ||
                type == UnitCellType::LeadsIn_MetalOn ||
                type == UnitCellType::FLTI_nopixel)
            features |= Power;

        if (type == UnitCellType::CalibPad_NoAperture)
            features |= NoZMW | Aperture1Closed | Aperture2Closed | Aperture3Closed | Fiducial;

        if (type == UnitCellType::CalibPad_Dark ||
                type == UnitCellType::CalibPad_off)
            features |= NoZMW | Fiducial;
    }

    return features;
}

}}  // PacBio::Primary

