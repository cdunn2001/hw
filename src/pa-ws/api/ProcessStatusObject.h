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
// File Description:
///  \brief The JSON object model for process_status used in several endpoints
//
// Programmer: Mark Lakata
#ifndef PA_WS_API_PROCESSSTATUSOBJECT_H
#define PA_WS_API_PROCESSSTATUSOBJECT_H

#include <pacbio/configuration/PBConfig.h>

#include "ObjectTypes.h"

namespace PacBio {
namespace API {

struct ProcessStatusObject : PacBio::Configuration::PBConfig<ProcessStatusObject>
{
    PB_CONFIG(ProcessStatusObject);

    SMART_ENUM(ExecutionStatus_t, UNKNOWN, READY, RUNNING, COMPLETE);
    SMART_ENUM(CompletionStatus_t, UNKNOWN, SUCCESS, FAILED, ABORTED);
    PB_CONFIG_PARAM(ExecutionStatus_t, execution_status, ExecutionStatus_t::UNKNOWN);
    PB_CONFIG_PARAM(CompletionStatus_t, completion_status, CompletionStatus_t::UNKNOWN);
    PB_CONFIG_PARAM(ISO8601_Timestamp_t, timestamp, "00000101T00:00:00.000Z"); /// < ISO8601 with milliseconds
    PB_CONFIG_PARAM(int, exit_code, 0);
};

}}

#endif
