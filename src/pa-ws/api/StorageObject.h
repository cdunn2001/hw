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
///  \brief The JSON object models for /storages endpoints
//
// Programmer: Mark Lakata

#ifndef PA_WS_API_STORAGEOBJECT_H
#define PA_WS_API_STORAGEOBJECT_H

#include <pacbio/configuration/PBConfig.h>
#include <pacbio/utilities/SmartEnum.h>

namespace PacBio {
namespace API {

struct StorageItemObject : PacBio::Configuration::PBConfig<StorageItemObject>
{
    PB_CONFIG(StorageItemObject);

    SMART_ENUM(Category_t, UNKNOWN, BAM);

    PB_CONFIG_PARAM(std::string, url, ""); ///< URL to this object
    PB_CONFIG_PARAM(std::string, timestamp, ""); ///< ISO8601 timestamp of file write time
    PB_CONFIG_PARAM(uint64_t, size, ""); ///< ISO8601 timestamp of file write time
    PB_CONFIG_PARAM(Category_t, category, Category_t::UNKNOWN); ///< 
    PB_CONFIG_PARAM(std::string, source_info, ""); ///< information about the source of this file
};

struct StorageDiskReport : PacBio::Configuration::PBConfig<StorageDiskReport>
{
    PB_CONFIG(StorageDiskReport);

    PB_CONFIG_PARAM(uint64_t, total_space,0);
    PB_CONFIG_PARAM(uint64_t, free_space,0);
};

struct StorageObject : PacBio::Configuration::PBConfig<StorageObject>
{
    PB_CONFIG(StorageObject);

    PB_CONFIG_PARAM(std::string, mid, ""); ///< Movie context ID
    PB_CONFIG_PARAM(std::string, root_url, ""); ///< symbolic link to storage directory
    PB_CONFIG_PARAM(std::string, linux_path, ""); ///< physical path to storage directory (should only be used for debugging and logging)
    PB_CONFIG_PARAM(url, output_log_url, "discard:");
    PB_CONFIG_PARAM(LogLevel_t, log_level, LogLevel_t::INFO); ///< log severity threshold

    PB_CONFIG_PARAM(std::vector<StorageItemObject>, files, 0);

    PB_CONFIG_OBJECT(StorageDiskReport, space);
    PB_CONFIG_OBJECT(ProcessStatusObject, process_status);       ///< This is the process of "file deletion".
};

}}

#endif // include guard
