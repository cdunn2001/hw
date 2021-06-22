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
///  \brief The JSON object models for /sockets endpoints
//
// Programmer: Mark Lakata

#ifndef PA_WS_API_SOCKETOBJECT_H
#define PA_WS_API_SOCKETOBJECT_H

#include <stdint.h>

#include <pacbio/configuration/PBConfig.h>

#include "ProcessStatusObject.h"
#include "ObjectTypes.h"

namespace PacBio {
namespace API {


struct SocketDarkcalObject : PacBio::Configuration::PBConfig<SocketDarkcalObject>
{
    PB_CONFIG(SocketDarkcalObject);

    PB_CONFIG_OBJECT(ProcessStatusObject, process_status);       
    PB_CONFIG_PARAM(uint64_t, movie_max_frames, 0); ///< Movie length in frames
    PB_CONFIG_PARAM(double, movie_max_seconds, 0); ///< Movie length in seconds
    PB_CONFIG_PARAM(uint32_t, movie_number, 0); ///< arbitrary movie number to delimite the start and end
    PB_CONFIG_PARAM(url, calib_file_url, "discard:"); ///< calibration file destination
    PB_CONFIG_PARAM(url, log_url, "discard:"); ///< log file destination
    PB_CONFIG_PARAM(LogLevel_t, log_level, LogLevel_t::INFO); ///< log severity threshold
};

struct SocketLoadingcalObject : PacBio::Configuration::PBConfig<SocketLoadingcalObject>
{
    PB_CONFIG(SocketLoadingcalObject);
    
    PB_CONFIG_OBJECT(ProcessStatusObject, process_status);       
    PB_CONFIG_PARAM(uint64_t, movie_max_frames, 0); ///< Movie length in frames
    PB_CONFIG_PARAM(double, movie_max_time, 0); ///< Movie length in seconds
    PB_CONFIG_PARAM(uint32_t, movie_number, 0); ///< arbitrary movie number to delimite the start and end
    PB_CONFIG_PARAM(url, dark_frame_file_url, "discard:"); ///< dark_frame file URL source
    PB_CONFIG_PARAM(url, calib_file_url, "discard:"); ///< calibration file destination
    PB_CONFIG_PARAM(url, log_url, "discard:"); ///< log file destination
    PB_CONFIG_PARAM(LogLevel_t, log_level, LogLevel_t::INFO); ///< log severity threshold
};

struct SocketBasecallerRTMetricsObject : PacBio::Configuration::PBConfig<SocketBasecallerRTMetricsObject>
{
    PB_CONFIG(SocketBasecallerRTMetricsObject);

    PB_CONFIG_PARAM(std::string, url, "http://localhost:23632/m123456/rtmetrics_20210602T060545Z.xml" );
};

struct AnalogObject : PacBio::Configuration::PBConfig<AnalogObject>
{
    PB_CONFIG(AnalogObject);

    SMART_ENUM(BaseLabel_t, N,A,T,G,C);
    PB_CONFIG_PARAM(BaseLabel_t, base_label, BaseLabel_t::N);
    PB_CONFIG_PARAM(double, relative_amp, 1.0);
    PB_CONFIG_PARAM(double, inter_pulse_distance_sec, 0.0);
    PB_CONFIG_PARAM(double, excess_noise_cv, 0.0);
    PB_CONFIG_PARAM(double, pulse_width_sec, 0.0);
    PB_CONFIG_PARAM(double, pw_to_slow_step_ratio, 0.15);
    PB_CONFIG_PARAM(double, ipd_to_slow_step_ratio, 0.15);
};


struct SocketBasecallerObject : PacBio::Configuration::PBConfig<SocketBasecallerObject>
{
    PB_CONFIG(SocketBasecallerObject);

    PB_CONFIG_PARAM(std::string, mid, ""); // EXAMPLE("m123456_987654")
    PB_CONFIG_PARAM(std::string, uuid, ""); ///< subreadset UUID EXAMPLE("123e4567-e89b-12d3-a456-426614174000")

    PB_CONFIG_PARAM(uint64_t, movie_max_frames, 0); ///< Movie length in frames EXAMPLE(3600000)
    PB_CONFIG_PARAM(double, movie_max_seconds, 0); ///< Movie length in seconds EXAMPLE(36000)
    PB_CONFIG_PARAM(uint32_t, movie_number, 0); ///< arbitrary movie number to delimite the start and end EXAMPLE(567)
    PB_CONFIG_PARAM(url, baz_url, "discard:"); ///< baz destination EXAMPLE("http://localhost:23632/storages/m123456_987654/thefile.baz")
    PB_CONFIG_PARAM(url, trace_file_url, "discard:"); ///< trace file destination EXAMPLE("discard:")
    PB_CONFIG_PARAM(url, log_url, "discard:"); ///< log file destination EXAMPLE("http://localhost:23632/storages/m123456_987654/basecaller.log")
    PB_CONFIG_PARAM(LogLevel_t, log_level, LogLevel_t::INFO); ///< log severity threshold
    PB_CONFIG_PARAM(ControlledString_t, chiplayout, ""); ///< controlled name of the sensor chip unit cell layout
    PB_CONFIG_PARAM(url, darkcal_url, ""); // EXAMPLE("http://localhost:23632/storages/m123456_987654/darkcal.h5")
    PB_CONFIG_PARAM(std::vector<std::vector<double>>, pixel_spread_function, 0); ///< This is required and a function of the sensor NFC tag EXAMPLE([[0.0,0.1,0.0],[0.1,0.6,0.1],[0.0,0.1,0.0]])
    PB_CONFIG_PARAM(std::vector<std::vector<double>>, crosstalk_filter, 0); ///< Optional kernel definition of the crosstalk deconvolution.  THe pixel_spread_function is used to automatically calculate one if this is not specified. EXAMPLE([[0.0,0.1,0.0],[0.1,0.6,0.1],[0.0,0.1,0.0]])
    PB_CONFIG_PARAM(std::vector<AnalogObject>, analogs, 0);
    PB_CONFIG_PARAM(std::vector<std::vector<int>>, sequencing_roi, 0);
    PB_CONFIG_PARAM(std::vector<std::vector<int>>, trace_file_roi, 0);

    PB_CONFIG_PARAM(double, expected_frame_rate, 100.0); // EXAMPLE(100.0)
    PB_CONFIG_PARAM(double, photoelectron_sensitivity, 0.0); // EXAMPLE(1.4)
    PB_CONFIG_PARAM(double, ref_snr, 5.0); // EXAMPLE(10.0)

    PB_CONFIG_PARAM(url, simulation_file_url, "discard:"); ///< loopback file EXAMPLE("file://localhost/data/pa/sample_file.trc.h5")
    PB_CONFIG_PARAM(std::string, smrt_basecaller_config, "{}"); ///< SmrtBasecallerConfig. Passed to smrt_basecaller --config

    PB_CONFIG_OBJECT(ProcessStatusObject, process_status);       
    PB_CONFIG_OBJECT(SocketBasecallerRTMetricsObject, rt_metrics);
};

struct SocketObject : PacBio::Configuration::PBConfig<SocketObject>
{
    PB_CONFIG(SocketObject);

    PB_CONFIG_PARAM(int, index, -1); // EXAMPLE(1)

    PB_CONFIG_OBJECT(SocketDarkcalObject, darkcal);
    PB_CONFIG_OBJECT(SocketLoadingcalObject, loadingcal);
    PB_CONFIG_OBJECT(SocketBasecallerObject, basecaller);
};

}}

#endif //include guard
