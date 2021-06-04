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
///  \brief Temporary mockups of the JSON API objects.
//
// Programmer: Mark Lakata

#include <pacbio/utilities/ISO8601.h>
#include <pacbio/text/String.h>

#include "mockup.h"

namespace PacBio {
namespace API {

std::string CompressedISO8601()
{
    std::string s1 = PacBio::Utilities::ISO8601::TimeString();
    std::string s2;
    for(char c : s1)
    {
        if (c != ':' && c != '-') s2 += c;
    }
    return s2;
}

SocketObject CreateMockupOfSocketObject(int index)
{
    std::string mid = "m12345" + std::to_string(index);
    SocketObject so;
    so.index = index;
    so.darkcal.process_status.execution_status = ProcessStatusObject::ExecutionStatus_t::COMPLETE;
    so.darkcal.process_status.completion_status = ProcessStatusObject::CompletionStatus_t::FAILED;
    so.darkcal.process_status.exit_code = 137; // sig_segv
    so.darkcal.process_status.timestamp = "20210601T 01:23:45.000Z";
    so.darkcal.movie_max_frames = 512;
    so.darkcal.movie_max_seconds = 6;
    so.darkcal.movie_number = 111;
    so.darkcal.calib_file_url = "http://pac1:23632/storages/" + mid + "/darkcal.h5";
    so.darkcal.log_url = "http://pac1:23632/storages/" + mid + "/darkcal.log";

    so.loadingcal.process_status.execution_status = ProcessStatusObject::ExecutionStatus_t::READY;
    so.loadingcal.process_status.completion_status = ProcessStatusObject::CompletionStatus_t::UNKNOWN;
    so.loadingcal.process_status.timestamp = "20210601T 01:32:15.000Z";
    so.loadingcal.movie_max_frames = 0;
    so.loadingcal.movie_max_time = 0;
    so.loadingcal.movie_number = 0;
    so.loadingcal.calib_file_url = "discard:";
    so.loadingcal.log_url = "discard:";

    so.basecaller.process_status.execution_status = ProcessStatusObject::ExecutionStatus_t::RUNNING;
    so.basecaller.process_status.timestamp = PacBio::Utilities::ISO8601::TimeString();
    so.basecaller.mid = mid;
    so.basecaller.uuid = "00104afe-c341-11eb-8529-0242ac13000" + std::to_string(index);
    so.basecaller.movie_max_frames = 1000000;
    so.basecaller.movie_max_seconds = 10000;
    so.basecaller.movie_number = 113;
    so.basecaller.baz_url = "http://pac1:23632/storages/" + mid + "/" + mid + ".baz";
    so.basecaller.log_url = "http://pac1:23632/storages/" + mid + "/loadingcal.log";
    so.basecaller.chiplayout = "Minesweeper";
    so.basecaller.darkcal_url = "http://pac1:23632/storages/" + mid + "/darkcal.h5";
    so.basecaller.pixel_spread_function.resize(5);
    for (auto& r : so.basecaller.pixel_spread_function) r.resize(5);
    so.basecaller.pixel_spread_function[2][2] = 1.0;

    so.basecaller.crosstalk_filter.resize(7);
    for (auto& r : so.basecaller.crosstalk_filter) r.resize(7);
    so.basecaller.crosstalk_filter[3][3] = 1.0;
    so.basecaller.analogs.resize(4);
    so.basecaller.analogs[0].base_label = AnalogObject::BaseLabel_t::A;
    so.basecaller.analogs[0].relative_amp = 0.2;
    so.basecaller.analogs[1].base_label = AnalogObject::BaseLabel_t::C;
    so.basecaller.analogs[1].relative_amp = 0.4;
    so.basecaller.analogs[2].base_label = AnalogObject::BaseLabel_t::G;
    so.basecaller.analogs[2].relative_amp = 0.7;
    so.basecaller.analogs[3].base_label = AnalogObject::BaseLabel_t::T;
    so.basecaller.analogs[3].relative_amp = 1.0;
    so.basecaller.expected_frame_rate = 100.0;
    so.basecaller.photoelectron_sensitivity = 1.4;
    so.basecaller.ref_snr = 15.0;
    so.basecaller.rt_metrics.url = "http://pac1:23632/storages/" + mid + "/rt_metrics_" +
       CompressedISO8601() + ".xml";

    return so;
}

StorageObject CreateMockupOfStorageObject(int socket_number, const std::string& mid)
{
    StorageObject so;

    std::string rootUrl = "http://pac1:23632/storages/" + mid;

    so.mid = mid;
    so.root_url = rootUrl;
    so.linux_path = "file:/data/pa/storages/"+ mid;

    so.space.total_space = 100'000'000'000ULL;
    so.space.free_space = 90'000'000'000ULL;

    so.files.emplace_back();
    so.files.back().url = rootUrl + "/" + mid + ".baz";
    so.files.back().timestamp = PacBio::Utilities::ISO8601::TimeString();
    so.files.back().size = 1'230'000'000ULL;
    so.files.back().category = StorageItemObject::Category_t::BAM;
    so.files.back().source_info = "basecaller";

    so.files.emplace_back();
    so.files.back().url = rootUrl + "/" + mid + ".bam";
    so.files.back().timestamp = PacBio::Utilities::ISO8601::TimeString();
    so.files.back().size = 4'560'000'000ULL;
    so.files.back().category = StorageItemObject::Category_t::UNKNOWN;
    so.files.back().source_info = "baz2bam";

    so.process_status.execution_status = ProcessStatusObject::ExecutionStatus_t::READY;
    return so;
}

PostprimaryObject CreateMockupOfPostprimaryObject(const std::string& mid)
{
    PostprimaryObject po;
    po.mid = mid;
    std::string root = "http://localhost:23632/storages/" + mid + "/";
    po.baz_file_url =  root + mid + ".baz";
    po.uuid = "00104afe-c341-11eb-8529-0242ac130003";
    po.output_log_url = root + mid + ".ppa.log";
    po.log_level = LogLevel_t::DEBUG;
    po.output_prefix_url = root + mid;
    po.output_stats_xml_url = root + mid + ".stats.xml";
    po.output_stats_h5_url = root + mid + ".sts.h5";
    po.output_reduce_stats_h5_url = root + mid + ".rsts.h5";
    po.chiplayout = "Minesweeper";
    po.subreadset_metadata_xml = "<pbds:SubreadSet> ... lots of XML goes here ... </pbds:SubreadSet>";
    po.include_kinetics = true;
    po.ccs_on_instrument = true;

    po.process_status.execution_status = ProcessStatusObject::ExecutionStatus_t::RUNNING;
    po.process_status.timestamp = PacBio::Utilities::ISO8601::TimeString();
    po.status.progress = 0.34;
    po.status.output_urls.push_back(root + mid + ".dataset.xml");
    po.status.output_urls.push_back(root + mid + ".ccs.bam");
    po.status.output_urls.push_back(root + mid + ".baz2bam.log");
    po.status.baz2bam_zmws_per_min = 1.0e5;
    po.status.ccs_zmws_per_min = 1.0e5;
    po.status.num_zmws = 18579372;
    po.status.baz2bam_peak_rss_gb = 401.1;
    po.status.ccs_peak_rss_gb = 56.9;
    return po;
}

TransferObject CreateMockupOfTransferObject(int index, const std::string& mid)
{
    TransferObject to;
    to.mid = mid;
    to.protocol = TransferObject::Protocol::RSYNC;
    to.destination_url = "rsync://my.smrt.server.org:54321/this/experiment/"
        +mid+"_00000" + std::to_string(index);
    std::string root = "http://localhost:23632/storages/" + mid + "/";

    to.urls_to_transfer.push_back(root + mid + ".dataset.xml");
    to.urls_to_transfer.push_back(root + mid + ".ccs.bam");
    to.urls_to_transfer.push_back(root + mid + ".baz2bam.log");
    to.status.current_file = root + mid + ".ccs.bam";
    to.status.estimated_time_remaining = 3600.1;
    to.status.progress = 0.95;
    to.process_status.execution_status = ProcessStatusObject::ExecutionStatus_t::RUNNING;
    to.process_status.timestamp = PacBio::Utilities::ISO8601::TimeString();

    return to;
}

}} // namespace

