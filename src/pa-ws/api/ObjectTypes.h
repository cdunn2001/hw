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
///  \brief Some declarations of types for the JSON REST API.
//
// Programmer: Mark Lakata

#ifndef PA_WS_API_OBJECT_TYPES_H
#define PA_WS_API_OBJECT_TYPES_H

#include <string>

#include <pacbio/utilities/SmartEnum.h>

namespace PacBio {
namespace API {

typedef std::string url;  ///< A URL to the local /storages endpoint, or to an external file (if supported) EXAMPLE("http://pa-12345:23632/storages/m123456_0000001/mylog.txt")
typedef std::string ControlledString_t; ///< A string that has a limited set of allowed strings EXAMPLE("SequEL_4.0_RTO3")
typedef std::string ISO8601_Timestamp_t; ///< A timestamp string in the ISO8601 format, with milliseconds allowed. EXAMPLE("20210101T01:23:45.678Z")

SMART_ENUM(LogLevel_t,DEBUG,INFO,WARN,ERROR); ///< The verbosity of the log file expressed as a severity level

}}

#endif
