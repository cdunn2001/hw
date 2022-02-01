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
///  \brief The JSON object models for /transfers endpoints
//
// Programmer: Mark Lakata


#ifndef PA_WS_API_TRANSFER_OBJECT_H
#define PA_WS_API_TRANSFER_OBJECT_H

#include <pacbio/configuration/PBConfig.h>

namespace PacBio {
namespace API {

struct TransferStatusObject : PacBio::Configuration::PBConfig<TransferStatusObject>
{
    PB_CONFIG(TransferStatusObject);

    PB_CONFIG_PARAM(double, progress, 0.0); ///< The transfer progress measured as a factor of bytes transferred, normalized to [0.0,1.0] EXAMPLE(0.63)
    PB_CONFIG_PARAM(uint64_t, bytesTransferred, 0); ///< The number of bytes transferred so far EXAMPLE(98746374817)
    PB_CONFIG_PARAM(std::string, currentFile, ""); ///< The file that is currently being transferred EXAMPLE("foo.bam")
    PB_CONFIG_PARAM(double, estimatedTimeRemaining, 0.0); ///< Time in seconds to complete the transfer. EXAMPLE(1355.0)
    PB_CONFIG_PARAM(std::vector<std::string>, completedFiles, 0); ///< A list of files that have been successfully transferred so far EXAMPLE(["http://file1","http://file2"])
};

struct TransferObject : PacBio::Configuration::PBConfig<TransferObject>
{
    PB_CONFIG(TransferObject);

    SMART_ENUM(Protocol, UNKNOWN, RSYNC, SSH_RSYNC);

    PB_CONFIG_PARAM(std::string, mid, ""); ///< The movie context ID used to create this TransferObject EXAMPLE("m123456_987654")
    PB_CONFIG_PARAM(std::vector<std::string>, urlsToTransfer, 0); ///< A list of all source URLs to transfer to the destinationUrl EXAMPLE(["http://file1","http://file2",...])
    PB_CONFIG_PARAM(std::string, destinationUrl, "discard:"); ///< URL of the destination directory EXAMPLE("rsync://lims.customer.com:54321/foo")
    PB_CONFIG_PARAM(Protocol, protocol, Protocol::UNKNOWN); ///< The protocol used for the file transfer EXAMPLE("RSYNC")
    PB_CONFIG_PARAM(url, outputLogUrl, "discard:"); ///< Destination URL of the log file EXAMPLE("http://localhost:23632/storages/m123456_987654/transfer.log
    PB_CONFIG_PARAM(LogLevel_t, logLevel, LogLevel_t::INFO); ///< log severity threshold EXAMPLE("INFO")

    PB_CONFIG_OBJECT(TransferStatusObject, status);
    PB_CONFIG_OBJECT(ProcessStatusObject, processStatus);
};

}}

#endif // include guard
