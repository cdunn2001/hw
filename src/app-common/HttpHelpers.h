// Copyright (c) 2017, Pacific Biosciences of California, Inc.
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
/// \brief   miscellaneous enum and small classes to aid with HTTP management
//
// Programmer: Mark Lakata

#ifndef APP_COMMON_HTTPHELPERS_H
#define APP_COMMON_HTTPHELPERS_H

#include <string>
#include <stdexcept>

#include <pacbio/process/ConfigurationBase.h>
#include <pacbio/utilities/SmartEnum.h>

namespace PacBio {
namespace IPC {

SMART_ENUM(HttpStatus,
           OK = 200,
           CREATED = 201,
           TEMPORARY_REDIRECT = 307,
           PERMANENT_REDIRECT = 308,
           BAD_REQUEST = 400,
           FORBIDDEN = 403,
           NOT_FOUND = 404,
           INTERNAL_SERVER_ERROR = 500,
           NOT_IMPLEMENTED = 501,
           SERVICE_UNAVAILABLE = 503
);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct HttpResponse
{
    HttpStatus httpStatus;
    std::string contentType = "application/json";
    Json::Value json;
    std::string extraHeaders = "";
};

class HttpResponseException :
        public std::runtime_error
{
public:
    HttpResponseException(HttpStatus httpStatus, const std::string& message)
            :
            std::runtime_error(
                    message + "(" + std::to_string(static_cast<int>(httpStatus)) + "/" + httpStatus.toString() + ")"),
            httpStatus_(httpStatus) {}

    HttpStatus GetHttpStatus() const { return httpStatus_; }

private:
    HttpStatus httpStatus_;
};

}}

#endif //#define APP_COMMON_HTTPHELPERS_H

