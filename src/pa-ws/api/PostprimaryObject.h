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
///  \brief The JSON object models for /postprimaries endpoints
//
// Programmer: Mark Lakata

#ifndef PA_WS_API_POSTPRIMARYOBJECT_H
#define PA_WS_API_POSTPRIMARYOBJECT_H

#include <pacbio/configuration/PBConfig.h>

#include "ProcessStatusObject.h"
#include "ObjectTypes.h"

namespace PacBio {
namespace API {

struct PostprimaryStatusObject : PacBio::Configuration::PBConfig<PostprimaryStatusObject>
{
    PB_CONFIG(PostprimaryStatusObject);

    PB_CONFIG_PARAM(std::vector<url>, output_urls, 0);
    PB_CONFIG_PARAM(double, progress, 0.0); ///< progress of job completion. Range is [0.0, 1.0]
    PB_CONFIG_PARAM(double, baz2bam_zmws_per_min, 0.0);
    PB_CONFIG_PARAM(double, ccs_zmws_per_min, 0.0);
    PB_CONFIG_PARAM(uint64_t, num_zmws, 0);
    PB_CONFIG_PARAM(double, baz2bam_peak_rss_gb, 0.0);
    PB_CONFIG_PARAM(double, ccs_peak_rss_gb, 0.0);
};

struct PostprimaryObject : PacBio::Configuration::PBConfig<PostprimaryObject>
{
    PB_CONFIG(PostprimaryObject);

    PB_CONFIG_PARAM(std::string, mid, "");
    PB_CONFIG_PARAM(url, baz_file_url, "discard:"); // RFC 863
    PB_CONFIG_PARAM(std::string, uuid, "");


    PB_CONFIG_PARAM(url, output_log_url, "discard:");
    PB_CONFIG_PARAM(LogLevel_t, log_level, LogLevel_t::INFO); ///< log severity threshold
    /// output_dataset_url can be either .subreadset.xml or .consensusreadset.xml

    /// output files will use the output_dataset_url minus the .dataset.xml dataset extension
    /// For example,
    ///     output_dataset_url_prefix":http://localhost:23632/storages/0/m12346
    /// will result in files such as
    ///     http://localhost:23632/storages/0/m12346.subreadset.xml
    ///     http://localhost:23632/storages/0/m12346.subreadset.bam
    ///     http://localhost:23632/storages/0/m12346.zmws.bam
    ///     http://localhost:23632/storages/0/m12346.scraps.bam
    ///     http://localhost:23632/storages/0/m12346.consensusreadset.xml
    ///     http://localhost:23632/storages/0/m12346.reads.bam
    ///     http://localhost:23632/storages/0/m12346.ccs.bam
    ///     http://localhost:23632/storages/0/m12346.ccs_reports.json
    ///     http://localhost:23632/storages/0/m12346.zmw_metrics.json.gz
    ///     http://localhost:23632/storages/0/m12346.hifi_summary.json
    PB_CONFIG_PARAM(url, output_prefix_url, "discard:");

    PB_CONFIG_PARAM(url, output_stats_xml_url, "discard:"); ///< "stats.xml"
    PB_CONFIG_PARAM(url, output_stats_h5_url, "discard:"); ///< ".sts.h5"
    PB_CONFIG_PARAM(url, output_reduce_stats_h5_url, "discard:"); ///< if set, run reduce stats
    PB_CONFIG_PARAM(std::string, chiplayout, ""); ///< controlled name of the sensor chip unit cell layout
    PB_CONFIG_PARAM(std::string, subreadset_metadata_xml, "");
    PB_CONFIG_PARAM(bool, include_kinetics, false);
    PB_CONFIG_PARAM(bool, ccs_on_instrument, true);

    PB_CONFIG_OBJECT(PostprimaryStatusObject,status);
    PB_CONFIG_OBJECT(ProcessStatusObject,process_status);
};

}}

#endif
