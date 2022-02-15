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

SocketObject CreateMockupOfSocketObject(const std::string& socketId)
{
    std::string mid = "m12345" + socketId;
    // "The actual movie identifier looks more like "m" + machine serial number + "_" + date code."
    SocketObject so;
    so.socketId = socketId;
    so.darkcal.processStatus.executionStatus = ProcessStatusObject::ExecutionStatus_t::COMPLETE;
    so.darkcal.processStatus.completionStatus = ProcessStatusObject::CompletionStatus_t::FAILED;
    so.darkcal.processStatus.exitCode = 137; // sig_segv
    so.darkcal.processStatus.timestamp = "20210601T 01:23:45.000Z";
    so.darkcal.movieMaxFrames = 512;
    so.darkcal.movieMaxSeconds = 6;
    so.darkcal.movieNumber = 111;
    so.darkcal.calibFileUrl = "http://pac1:23632/storages/" + mid + "/darkcal.h5";
    so.darkcal.logUrl = "http://pac1:23632/storages/" + mid + "/darkcal.log";

    so.loadingcal.processStatus.executionStatus = ProcessStatusObject::ExecutionStatus_t::READY;
    so.loadingcal.processStatus.completionStatus = ProcessStatusObject::CompletionStatus_t::UNKNOWN;
    so.loadingcal.processStatus.timestamp = "20210601T 01:32:15.000Z";
    so.loadingcal.movieMaxFrames = 0;
    so.loadingcal.movieMaxSeconds = 0;
    so.loadingcal.movieNumber = 0;
    so.loadingcal.calibFileUrl = "discard:";
    so.loadingcal.logUrl = "discard:";

    so.basecaller.processStatus.executionStatus = ProcessStatusObject::ExecutionStatus_t::RUNNING;
    so.basecaller.processStatus.timestamp = PacBio::Utilities::ISO8601::TimeString();
    so.basecaller.mid = mid;
    so.basecaller.uuid = "00104afe-c341-11eb-8529-0242ac13000" + socketId;
    // "I took a valid UUID, removed the last digit and added the socketNumber."
    so.basecaller.movieMaxFrames = 1000000;
    so.basecaller.movieMaxSeconds = 10000;
    so.basecaller.movieNumber = 113;
    so.basecaller.bazUrl = "http://pac1:23632/storages/" + mid + "/" + mid + ".baz";
    so.basecaller.logUrl = "http://pac1:23632/storages/" + mid + "/loadingcal.log";
    so.basecaller.chiplayout = "Minesweeper";
    so.basecaller.darkcalFileUrl = "http://pac1:23632/storages/" + mid + "/darkcal.h5";
    so.basecaller.pixelSpreadFunction.resize(5);
    for (auto& r : so.basecaller.pixelSpreadFunction) r.resize(5);
    so.basecaller.pixelSpreadFunction[2][2] = 1.0;

    so.basecaller.crosstalkFilter.resize(7);
    for (auto& r : so.basecaller.crosstalkFilter) r.resize(7);
    so.basecaller.crosstalkFilter[3][3] = 1.0;
    so.basecaller.analogs.resize(4);
    so.basecaller.analogs[0].baseLabel = AnalogObject::BaseLabel_t::A;
    so.basecaller.analogs[0].relativeAmp = 0.2;
    so.basecaller.analogs[1].baseLabel = AnalogObject::BaseLabel_t::C;
    so.basecaller.analogs[1].relativeAmp = 0.4;
    so.basecaller.analogs[2].baseLabel = AnalogObject::BaseLabel_t::G;
    so.basecaller.analogs[2].relativeAmp = 0.7;
    so.basecaller.analogs[3].baseLabel = AnalogObject::BaseLabel_t::T;
    so.basecaller.analogs[3].relativeAmp = 1.0;
    so.basecaller.expectedFrameRate = 100.0;
    so.basecaller.photoelectronSensitivity = 1.4;
    so.basecaller.refSnr = 15.0;
    so.basecaller.rtMetrics.url = "http://pac1:23632/storages/" + mid + "/rt_metrics_" +
       CompressedISO8601() + ".xml";

    return so;
}

StorageObject CreateMockupOfStorageObject(const std::string& /*socketId*/, const std::string& mid)
{
    StorageObject so;

    std::string rootUrl = "http://pac1:23632/storages/" + mid;

    so.mid = mid;
    so.rootUrl = rootUrl;
    so.linuxPath = "file:/data/pa/storages/"+ mid;

    so.space.totalSpace = 100'000'000'000ULL;
    so.space.freeSpace = 90'000'000'000ULL;

    so.files.emplace_back();
    so.files.back().url = rootUrl + "/" + mid + ".baz";
    so.files.back().timestamp = PacBio::Utilities::ISO8601::TimeString();
    so.files.back().size = 1'230'000'000ULL;
    so.files.back().category = StorageItemObject::Category_t::BAM;
    so.files.back().sourceInfo = "basecaller";

    so.files.emplace_back();
    so.files.back().url = rootUrl + "/" + mid + ".bam";
    so.files.back().timestamp = PacBio::Utilities::ISO8601::TimeString();
    so.files.back().size = 4'560'000'000ULL;
    so.files.back().category = StorageItemObject::Category_t::UNKNOWN;
    so.files.back().sourceInfo = "baz2bam";

    so.processStatus.executionStatus = ProcessStatusObject::ExecutionStatus_t::READY;
    return so;
}

PostprimaryObject CreateMockupOfPostprimaryObject(const std::string& mid)
{
    PostprimaryObject po;
    po.mid = mid;
    std::string root = "http://localhost:23632/storages/" + mid + "/";
    po.bazFileUrl =  root + mid + ".baz";
    po.uuid = "00104afe-c341-11eb-8529-0242ac130003";
    po.outputLogUrl = root + mid + ".ppa.log";
    po.logLevel = LogLevel_t::DEBUG;
    po.outputPrefixUrl = root + mid;
    po.outputStatsXmlUrl = root + mid + ".stats.xml";
    po.outputStatsH5Url = root + mid + ".sts.h5";
    po.outputReduceStatsH5Url = root + mid + ".rsts.h5";
    po.chiplayout = "Minesweeper";
    po.subreadsetMetadataXml = "<pbds:SubreadSet> ... lots of XML goes here ... </pbds:SubreadSet>";
    po.includeKinetics = true;
    po.ccsOnInstrument = true;

    po.processStatus.executionStatus = ProcessStatusObject::ExecutionStatus_t::RUNNING;
    po.processStatus.timestamp = PacBio::Utilities::ISO8601::TimeString();
    po.status.progress = 0.34;
    po.status.outputUrls.push_back(root + mid + ".dataset.xml");
    po.status.outputUrls.push_back(root + mid + ".ccs.bam");
    po.status.outputUrls.push_back(root + mid + ".baz2bam.log");
    po.status.baz2bamZmwsPerMin = 1.0e5;
    po.status.ccsZmwsPerMin = 1.0e5;
    po.status.numZmws = 18579372;
    po.status.baz2bamPeakRssGb = 401.1;
    po.status.ccsPeakRssGb = 56.9;
    return po;
}

}} // namespace
