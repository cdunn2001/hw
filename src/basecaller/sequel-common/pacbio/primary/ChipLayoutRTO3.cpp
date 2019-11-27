
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
//  Defines members of ChipLayoutRTO3.

#include "ChipLayoutRTO3.h"

#include <pacbio/text/imemstream.h>

extern unsigned int  SequEl_4_0_RTO3_csv_gz_len;
extern unsigned char SequEl_4_0_RTO3_csv_gz[];

namespace PacBio {
namespace Primary {


const ChipLayoutSequel::SequelUnitFeature ChipLayoutRTO3::featureLookup[33] =
{
    NotUsed,     //0
    StandardZMW, //1
    NotUsed,     //2
    NoZMW | Aperture1Closed | Aperture2Closed | Aperture3Closed | InactiveArea,     //3
    NoZMW,       //4
    Aperture1Closed | Aperture2Closed | Aperture3Closed,     //5
    NotUsed,     //6
    NotUsed,     //7
    NotUsed,     //8
    NotUsed,     //9
    NotUsed,     //10
    NotUsed,     //11
    NoZMW | Aperture1Closed | Aperture2Closed | Aperture3Closed | Power,     //12
    NoZMW | Aperture1Closed | Aperture2Closed | Aperture3Closed | Power | InactiveArea,     //13
    NoZMW | Aperture1Closed | Aperture2Closed | Aperture3Closed | Power | InactiveArea,     //14
    NoZMW | Aperture1Closed | Aperture2Closed | Aperture3Closed | Power | InactiveArea,     //15
    NoZMW | Aperture1Closed | Aperture2Closed | Aperture3Closed | Power | InactiveArea,     //16
    Aperture1GreenClosed   | Fiducial,    //17 red
    Aperture1RedClosed     | Fiducial,    //18 green
    NotUsed,     //19
    NotUsed,     //20
    NotUsed,     //21
    NoZMW | Aperture1Closed | Aperture2Closed | Aperture3Closed | Power  | InactiveArea,     //22
    NoZMW | WaveguideClosed,     //23
    NoZMW | WaveguideClosed | Aperture1Closed | Aperture2Closed | Aperture3Closed | InactiveArea,     //24
    NoZMW |                                     Aperture3Closed,     //25
    NotUsed,     //26
    NoZMW |                                     Aperture3Closed | Fiducial,     //27
    Aperture1Closed | Aperture2Closed | Aperture3Closed | Fiducial,     //28
    NoZMW | Aperture1Closed | Aperture2Closed | Aperture3Closed | Fiducial,     //29
    Aperture1Closed | Aperture2Closed | Aperture3Closed | Fiducial,     //30
    Aperture1GreenClosed   | Fiducial,     //31 red
    Aperture1RedClosed     | Fiducial,     //32 green
};


ChipLayoutRTO3::ChipLayoutRTO3()
{
    InitFullMatrix();

    // reads stream from internal .TEXT segment (which is gzip compressed)
    PacBio::Text::imemstream buf((const char*) SequEl_4_0_RTO3_csv_gz,
                                 (size_t)      SequEl_4_0_RTO3_csv_gz_len);
    Load(buf);
}

}}  // PacBio::Primary
