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

/// Enum with all available metric field names
SMART_BAZ_ENUM(MetricFieldName ,

    // Metrics to be obsoleted

               NUM_FRAMES               = 0,
               NUM_BASES                   ,
               ANGLE_A                     ,
               ANGLE_C                     ,
               ANGLE_G                     ,
               ANGLE_T                     ,
               NUM_PULSES_A                ,
               NUM_PULSES_C                ,
               NUM_PULSES_G                ,
               NUM_PULSES_T                ,

    // Metrics common to both Sequel/Spider

               NUM_PULSES                  ,
               PULSE_WIDTH                 ,
               BASE_WIDTH                  ,
               PKMID_A                     ,
               PKMID_C                     ,
               PKMID_G                     ,
               PKMID_T                     ,
               PKMID_FRAMES_A              ,
               PKMID_FRAMES_C              ,
               PKMID_FRAMES_G              ,
               PKMID_FRAMES_T              ,
               NUM_SANDWICHES              ,
               NUM_HALF_SANDWICHES         ,
               NUM_PULSE_LABEL_STUTTERS    ,
               PKMAX_A                     ,
               PKMAX_C                     ,
               PKMAX_G                     ,
               PKMAX_T                     ,
               PULSE_DETECTION_SCORE       ,
               TRACE_AUTOCORR              ,
               PIXEL_CHECKSUM              ,


               BPZVAR_A                    ,
               BPZVAR_C                    ,
               BPZVAR_G                    ,
               BPZVAR_T                    ,
               PKZVAR_A                    ,
               PKZVAR_C                    ,
               PKZVAR_G                    ,
               PKZVAR_T                    ,

    // New metrics for Kiwi/Spider

               NUM_BASES_A                 ,
               NUM_BASES_C                 ,
               NUM_BASES_G                 ,
               NUM_BASES_T                 ,
               DME_STATUS                  ,
               ACTIVITY_LABEL              ,

    // Sequel-only metrics
               BASELINE_RED_SD             ,
               BASELINE_GREEN_SD           ,
               BASELINE_RED_MEAN           ,
               BASELINE_GREEN_MEAN         ,
               ANGLE_RED                   ,
               ANGLE_GREEN                 ,

    // Spider-only metrics
               BASELINE_SD                 ,
               BASELINE_MEAN               ,

               GAP                      = -1
);


}}
