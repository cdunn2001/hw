// Copyright (c) 2014-2016, Pacific Biosciences of California, Inc.
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted (subject to the limitations in the
// disclaimer below) provided that the following conditions are met:
//
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
//  * Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
//  * Neither the name of Pacific Biosciences nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE
// GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY PACIFIC
// BIOSCIENCES AND ITS CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL PACIFIC BIOSCIENCES OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
// USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
// OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGE.

// Programmer: Armin Töpfer

#pragma once

#include "SmartBazEnum.h"

namespace PacBio {
namespace Primary {

/// Enum with all available packet field names
SMART_BAZ_ENUM(PacketFieldName,
    READOUT       = 0, // Base or pulse readout {A, C, G, T}

    DEL_TAG       = 1, // Deletion Tag       {A, C, G, T, N}
    SUB_TAG          , // Substitution Tag   {A, C, G, T, N}
                       
    LABEL            , // Pulse Label Tag
    ALT_LABEL        , // Pulse Alternative Label Tag

    DEL_QV           , // Deletion QV
    SUB_QV           , // Substitution QV
    INS_QV           , // Insertion QV
    MRG_QV           , // Merge QV

    LAB_QV           , // Label QV
    ALT_QV           , // Alternative Label QV

    IPD_LL           , // Inter Pulse Duration of a base, lossless (8 bit part)
    IPD16_LL         , // Inter Pulse Duration of a base, lossless (16 bit extension)
    IPD32_LL         , // Inter Pulse Duration of a base, lossless (32 bit extension)
    IPD_V1           , // Inter Pulse Duration of a base, lossy

    PW_LL            , // Pulse width of a base, lossless (8 bit part)
    PW16_LL          , // Pulse width of a base, lossless (16 bit extension)
    PW32_LL          , // Pulse width of a base, lossless (32 bit extension)
    PW_V1            , // Pulse width of a base, lossy
    
    PKMEAN_LL        , // PK mean, lossless (8 bit part)
    PKMEAN16_LL      , // PK mean, lossless (16 bit extension)
    
    PKMID_LL         , // PK mid, lossless (8 bit part)
    PKMID16_LL       , // PK mid, lossless (16 bit extension)
    
    IS_BASE          , // Is this event a base?
    IS_PULSE         , // Is this a pulse? Otherwise it was added by P2B

    PX_LL            , // Pulse width of the underlying pulse, lossless (8 bit part)
    PX16_LL          , // Pulse width of the underlying pulse, lossless (16 bit extension)
    PX32_LL          , // Pulse width of the underlying pulse, lossless (32 bit extension)

    PD_LL            , // Pre pulse frames of the underlying pulse, lossless (8 bit part)
    PD16_LL          , // Pre pulse frames of the underlying pulse, lossless (16 bit extension)
    PD32_LL          , // Pre pulse frames of the underlying pulse, lossless (32 bit extension)

    OVERALL_QV       ,

    PULSE_MRG_QV     ,

    START_FRAME      ,

    PKMEAN2_LL       , // PK mean 2, two 16bit channels combined via bitshifting: green << 16 | red

    PKMID2_LL        , // PK mid2, two 16bit channels combined via bitshifting: green << 16 | red

    GAP          = -1
);

}}
