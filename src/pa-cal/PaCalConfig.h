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
///  \brief Defines the calibration parameters for the pa-cal application


#ifndef KES_CAL_CONFIG_H
#define KES_CAL_CONFIG_H

#include <pacbio/configuration/PBConfig.h>
#include <pacbio/configuration/types/Variant.h>

#include <acquisition/wxipcdatasource/WXIPCDataSourceConfig.h>

namespace PacBio::Calibration {

// Config primarily for test, that fabricates it's own data
struct SimInputConfig : PacBio::Configuration::PBConfig<SimInputConfig>
{
    PB_CONFIG(SimInputConfig);

    PB_CONFIG_PARAM(uint32_t, nRows, 1000);
    PB_CONFIG_PARAM(uint32_t, nCols, 1000);

    PB_CONFIG_PARAM(uint32_t, Pedestal, 0);

    // TODO add params to control characteristics of simulated data?

    // The minimum time it will take to produce the chunk of data for
    // processing.  Intended to be used for testing to make sure
    // data generation doesn't take trivial time and that things
    // that are intended to be concurrent can be tested as such.
    PB_CONFIG_PARAM(double, minInputDelaySeconds, 1.0);
};

using WXIPCDataSourceConfig = PacBio::Acquisition::DataSource::WXIPCDataSourceConfig;

struct PaCalConfig : PacBio::Configuration::PBConfig<PaCalConfig>
{
    PB_CONFIG(PaCalConfig);

    PB_CONFIG_VARIANT(source, WXIPCDataSourceConfig, SimInputConfig);

    // TODO configs to enable/configure some form of BIST?
    // TODO config for injecting a signal after some time?
};

} // namespace PacBio::Calibration


#endif // WXDAEMON_CONFIG_H
