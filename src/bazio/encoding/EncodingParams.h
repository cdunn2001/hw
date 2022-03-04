// Copyright (c) 2021, Pacific Biosciences of California, Inc.
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


#ifndef PACBIO_BAZIO_ENCODING_ENCODING_PARAMS_H
#define PACBIO_BAZIO_ENCODING_ENCODING_PARAMS_H

#include <vector>

#include <pacbio/configuration/types/Variant.h>
#include <pacbio/configuration/PBConfig.h>

#include <bazio/encoding/FieldNames.h>
#include <bazio/encoding/Types.h>

namespace PacBio {
namespace BazIO {

struct NoOpTransformParams : Configuration::PBConfig<NoOpTransformParams>
{
    PB_CONFIG(NoOpTransformParams);
};
struct DeltaCompressionParams : Configuration::PBConfig<DeltaCompressionParams>
{
    PB_CONFIG(DeltaCompressionParams);
};
struct CodecParams : Configuration::PBConfig<CodecParams>
{
    PB_CONFIG(CodecParams);

    PB_CONFIG_PARAM(uint16_t, numBits, 0);
};
struct FixedPointParams : Configuration::PBConfig<FixedPointParams>
{
    PB_CONFIG(FixedPointParams);

    PB_CONFIG_PARAM(uint32_t, scale, 0);
};
struct FloatFixedCodecParams : Configuration::PBConfig<FloatFixedCodecParams>
{
    PB_CONFIG(FloatFixedCodecParams);

    PB_CONFIG_PARAM(uint32_t, scale, 0);
    PB_CONFIG_PARAM(uint32_t, numBytes, 4);
};
struct TransformsParams : Configuration::PBConfig<TransformsParams>
{
    PB_CONFIG(TransformsParams);

    PB_CONFIG_VARIANT(params, NoOpTransformParams, CodecParams, FixedPointParams, DeltaCompressionParams, FloatFixedCodecParams);
};

struct TruncateParams : Configuration::PBConfig<TruncateParams>
{
    PB_CONFIG(TruncateParams);

    PB_CONFIG_PARAM(uint16_t, numBits, 0);
};
struct SimpleOverflowParams : Configuration::PBConfig<SimpleOverflowParams>
{
    PB_CONFIG(SimpleOverflowParams);

    PB_CONFIG_PARAM(uint16_t, numBits, 0);
    PB_CONFIG_PARAM(uint16_t, overflowBytes, 0);
};
struct CompactOverflowParams : Configuration::PBConfig<CompactOverflowParams>
{
    PB_CONFIG(CompactOverflowParams);

    PB_CONFIG_PARAM(uint16_t, numBits, 0);
};
struct SerializeParams : Configuration::PBConfig<SerializeParams>
{
    PB_CONFIG(SerializeParams);

    PB_CONFIG_VARIANT(params, TruncateParams, SimpleOverflowParams, CompactOverflowParams);
};

template <typename FieldNames>
struct FieldParams : Configuration::PBConfig<FieldParams<FieldNames>>
{
    PB_CONFIG(FieldParams);

    PB_CONFIG_PARAM(FieldNames, name, PacketFieldName::Label);
    PB_CONFIG_PARAM(StoreSigned::UnderlyingType, storeSigned, false);
    PB_CONFIG_OBJECT(std::vector<TransformsParams>, transform);
    PB_CONFIG_OBJECT(SerializeParams, serialize);
};

template <typename FieldNames>
struct GroupParams : Configuration::PBConfig<GroupParams<FieldNames>>
{
    PB_CONFIG(GroupParams);

    PB_CONFIG_OBJECT(std::vector<FieldParams<FieldNames>>, members);
    PB_CONFIG_PARAM(std::vector<size_t>, numBits, std::vector<size_t>{});
    PB_CONFIG_PARAM(size_t, totalBits, 0);
};

}}

#endif //PACBIO_BAZIO_ENCODING_ENCODING_PARAMS_H
