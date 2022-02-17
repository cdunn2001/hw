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

    SMART_ENUM(Category_t, UNKNOWN, BAM, BAZ, CAL);

    PB_CONFIG_PARAM(std::string, url, ""); ///< URL of this object EXAMPLE("http://localhost:23632/storages/m123456_987654/foobar1.bam")
    PB_CONFIG_PARAM(std::string, timestamp, ""); ///< ISO8601 timestamp of file write time EXAMPLE("2017-01-31T01:59:49.103998Z")
    PB_CONFIG_PARAM(uint64_t, size, ""); ///< size of the file  EXAMPLE(6593845929837)
    PB_CONFIG_PARAM(Category_t, category, Category_t::UNKNOWN); ///< The category for this particular item in the StorageObject EXAMPLE("BAM")
    PB_CONFIG_PARAM(std::string, sourceInfo, ""); ///< information about the source of this file
};

struct StorageDiskReportObject : PacBio::Configuration::PBConfig<StorageDiskReportObject>
{
    PB_CONFIG(StorageDiskReportObject);

    PB_CONFIG_PARAM(uint64_t, totalSpace,0);  ///< Total space allocated in bytes for this StorageObject, include used and unused space EXAMPLE(6593845929837)
    PB_CONFIG_PARAM(uint64_t, freeSpace,0);   ///< Total unused space in bytes of this StorageObject EXAMPLE(6134262344238)
};

struct StorageObject : PacBio::Configuration::PBConfig<StorageObject>
{
    PB_CONFIG(StorageObject);

    PB_CONFIG_PARAM(std::string, mid, ""); ///< Movie context ID EXAMPLE("m123456_987654")
    PB_CONFIG_PARAM(std::string, rootUrl, ""); ///< symbolic link to storage directory which points back to this StorageObject EXAMPLE("http://localhost:23632/storages/m123456_987654")
    PB_CONFIG_PARAM(std::string, linuxPath, ""); ///< physical path to storage directory (should only be used for debugging and logging) EXAMPLE("file:/data/pa/m123456_987654")
    PB_CONFIG_PARAM(url, logUrl, "discard:"); ///< Destination URL for the log file. Logging happens during construction and freeing. EXAMPLE("http://localhost:23632/storages/m123456_987654/storage.log")
    PB_CONFIG_PARAM(LogLevel_t, logLevel, LogLevel_t::INFO); ///< log severity threshold EXAMPLE("INFO")

    // TBD. The /storages/$mid/files endpoint refers to the actual files themselves. It is possible GET
    // the contents of the files. Under the hood, pa-ws will translate any references to these files into
    // local filenames when passed to the command lines of the managed processes. For example, if postprimary is
    // posted with a setting such as "bazTempFile":"http://nrt1:23602/storages/m1234/files/foo_temp.txt", pa-ws
    // will translate this to a command line that will look something like
    // `baz2bam --tempfile /data/pa/tmp.Egf3vDs1Fc/foo_temp.txt`, where /data/pa/tmp.Egf3vDs1Fc has been allocated
    // for the "m1234" storages object.
    // A GET to this top level directory will return a list of files in the directory.
    // TBD not sure if the GET should recursively walk the subdirectories. Maybe this is an option.
    PB_CONFIG_PARAM(std::vector<StorageItemObject>, files, 0); ///< A list of all the files in this StorageObject EXAMPLE(["http://localhost:23632/storages/m123456_987654/storage.log","http://localhost:23632/storages/m123456_987654/my.baz",...])

    PB_CONFIG_OBJECT(StorageDiskReportObject, space);
    PB_CONFIG_OBJECT(ProcessStatusObject, processStatus);       ///< This is the process of "file deletion".
};

}}

#endif // include guard
