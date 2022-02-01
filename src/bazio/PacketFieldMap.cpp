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

#include "PacketFieldMap.h"

namespace PacBio {
namespace Primary {

std::map<PacketFieldName, std::pair<std::string, FieldType>> PacketFieldMap::packetBaseFieldToBamID =
    {
        {PacketFieldName::DEL_TAG, std::make_pair("dt", FieldType::CHAR)},
        {PacketFieldName::SUB_TAG, std::make_pair("st", FieldType::CHAR)},
        {PacketFieldName::DEL_QV,  std::make_pair("dq", FieldType::CHAR)},
        {PacketFieldName::SUB_QV,  std::make_pair("sq", FieldType::CHAR)},
        {PacketFieldName::INS_QV,  std::make_pair("iq", FieldType::CHAR)},
        {PacketFieldName::MRG_QV,  std::make_pair("mq", FieldType::CHAR)},
        {PacketFieldName::IPD_LL,  std::make_pair("ip", FieldType::UINT16)},
        {PacketFieldName::IPD_V1,  std::make_pair("ip", FieldType::UINT8)},
        {PacketFieldName::PW_LL,   std::make_pair("pw", FieldType::UINT16)},
        {PacketFieldName::PW_V1,   std::make_pair("pw", FieldType::UINT8)}
    };

std::map<PacketFieldName, std::pair<std::string, FieldType>> PacketFieldMap::packetPulseFieldToBamID =
    {
        {PacketFieldName::LAB_QV,       std::make_pair("pq", FieldType::CHAR)},
        {PacketFieldName::ALT_QV,       std::make_pair("pv", FieldType::CHAR)},
        {PacketFieldName::ALT_LABEL,    std::make_pair("pt", FieldType::CHAR)},
        {PacketFieldName::PKMEAN_LL,    std::make_pair("pa", FieldType::UINT16)},
        {PacketFieldName::PKMID_LL,     std::make_pair("pm", FieldType::UINT16)},
        {PacketFieldName::PKMEAN2_LL,   std::make_pair("ps", FieldType::UINT32)},
        {PacketFieldName::PKMID2_LL,    std::make_pair("pi", FieldType::UINT32)},
        {PacketFieldName::PX_LL,        std::make_pair("px", FieldType::UINT16)},
        {PacketFieldName::PD_LL,        std::make_pair("pd", FieldType::UINT16)},
        {PacketFieldName::PULSE_MRG_QV, std::make_pair("pg", FieldType::CHAR)},
        {PacketFieldName::START_FRAME,  std::make_pair("sf", FieldType::UINT32)}
    };

}}
