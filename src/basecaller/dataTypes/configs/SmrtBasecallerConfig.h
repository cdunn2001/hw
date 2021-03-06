// Copyright (c) 2019-2020, Pacific Biosciences of California, Inc.
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

#ifndef mongo_dataTypes_configs_SmrtBasecallerConfig_H_
#define mongo_dataTypes_configs_SmrtBasecallerConfig_H_

#include <pacbio/configuration/PBConfig.h>
#include <pacbio/configuration/types/Variant.h>

#include <bazio/BazIOConfig.h>

#include <dataTypes/configs/BasecallerAlgorithmConfig.h>
#include <dataTypes/configs/BatchLayoutConfig.h>
#include <dataTypes/configs/DataSourceConfig.h>
#include <dataTypes/configs/PrelimHQConfig.h>
#include <dataTypes/configs/RealTimeMetricsConfig.h>
#include <dataTypes/configs/SourceConfig.h>
#include <dataTypes/configs/SystemsConfig.h>
#include <dataTypes/configs/TraceSaverConfig.h>

namespace PacBio {
namespace Mongo {
namespace Data {

class SmrtBasecallerConfig : public Configuration::PBConfig<SmrtBasecallerConfig>
{
    PB_CONFIG(SmrtBasecallerConfig);

    PB_CONFIG_OBJECT(SystemsConfig, system);
    PB_CONFIG_OBJECT(BasecallerAlgorithmConfig, algorithm);
    PB_CONFIG_OBJECT(BatchLayoutConfig, layout);
    PB_CONFIG_OBJECT(TraceSaverConfig, traceSaver);
    PB_CONFIG_OBJECT(PrelimHQConfig, prelimHQ);
    PB_CONFIG_OBJECT(RealTimeMetricsConfig, realTimeMetrics);
    PB_CONFIG_OBJECT(Primary::BazIOConfig, bazIO);
    PB_CONFIG_OBJECT(DataSourceConfig, dataSource);

    PB_CONFIG_VARIANT(source,
                      TraceReplication,
                      TraceReanalysis,
                      WXIPCDataSourceConfig);

    PB_CONFIG_PARAM(bool, internalMode, false);
    PB_CONFIG_PARAM(bool, multipleBazFiles, true);

    /// Minimum duration (in frames) between intermediate reports
    /// for both memory and compute statistics.  Reports will happen
    /// only on chunk boundaries
    PB_CONFIG_PARAM(size_t, monitoringReportInterval, 8192);
};

}}}     // namespace PacBio::Mongo::Data


#endif //mongo_dataTypes_configs_SmrtBasecallerConfig_H_
