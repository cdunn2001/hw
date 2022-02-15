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

    PB_CONFIG_OBJECT(ProcessStatusObject, processStatus);       
    PB_CONFIG_PARAM(uint64_t, movieMaxFrames, 0); ///< Movie length in frames.  The values movieMaxFrames and movieMaxTime should be similar, but not exactly the same, depending on whether true elapsed time or accurate frame count is desired. One value should be the desired amount and the other value should be an emergency stop amount. EXAMPLE(500)
    PB_CONFIG_PARAM(double, movieMaxSeconds, 0); ///< Movie length in seconds.  The values movieMaxFrames and movieMaxTime should be similar, but not exactly the same, depending on whether true elapsed time or accurate frame count is desired. One value should be the desired amount and the other value should be an emergency stop amount. EXAMPLE(6.0)
    PB_CONFIG_PARAM(uint32_t, movieNumber, 0); ///< Arbitrary movie number to delimite the start and end EXAMPLE(1)
    PB_CONFIG_PARAM(url, calibFileUrl, "discard:"); ///< Destination URL of the calibration file EXAMPLE("http://localhost:23632/storages/m123456_987654/darkcal.h5")
    PB_CONFIG_PARAM(url, logUrl, "discard:"); ///< Destination URL of the log file EXAMPLE("http://localhost:23632/storages/m123456_987654/darkcal.log")
    PB_CONFIG_PARAM(LogLevel_t, logLevel, LogLevel_t::INFO); ///< Log severity threshold EXAMPLE("INFO")
};

struct SocketLoadingcalObject : PacBio::Configuration::PBConfig<SocketLoadingcalObject>
{
    PB_CONFIG(SocketLoadingcalObject);
    
    PB_CONFIG_OBJECT(ProcessStatusObject, processStatus);       
    PB_CONFIG_PARAM(uint64_t, movieMaxFrames, 0); ///< Movie length in frames. The values movieMaxFrames and movieMaxTime should be similar, but not exactly the same, depending on whether true elapsed time or accurate frame count is desired. One value should be the desired amount and the other value should be an emergency stop amount.  EXAMPLE(500)
    PB_CONFIG_PARAM(double, movieMaxTime, 0); ///< Maximum movie length in seconds.  The values movieMaxFrames and movieMaxTime should be similar, but not exactly the same, depending on whether true elapsed time or accurate frame count is desired. One value should be the desired amount and the other value should be an emergency stop amount.  EXAMPLE(6.0)
    PB_CONFIG_PARAM(uint32_t, movieNumber, 0); ///< Arbitrary movie number to delimit the start and end of a calibration frame set EXAMPLE(2)
    PB_CONFIG_PARAM(url, darkFrameFileUrl, "discard:"); ///< Source URL of the dark_frame calibration file EXAMPLE("http://localhost:23632/storages/m123456_987654/darkcal.h5")
    PB_CONFIG_PARAM(url, calibFileUrl, "discard:"); ///< Destination URL of the calibration file EXAMPLE("http://localhost:23632/storages/m123456_987654/loadingcal.h5")
    PB_CONFIG_PARAM(url, logUrl, "discard:"); ///< Destination URL of the log file  EXAMPLE("http://localhost:23632/storages/m123456_987654/loadingcal.log")
    PB_CONFIG_PARAM(LogLevel_t, logLevel, LogLevel_t::INFO); ///< Log severity threshold EXAMPLE("INFO")
};

struct SocketBasecallerRTMetricsObject : PacBio::Configuration::PBConfig<SocketBasecallerRTMetricsObject>
{
    PB_CONFIG(SocketBasecallerRTMetricsObject);

    PB_CONFIG_PARAM(std::string, url, "discard:" ); ///< Source URL of the most recent RT Metrics file. When the file is updated, the URL will change with the embedded timestamp EXAMPLE("http://localhost:23632/storages/m123456_987654/rtmetrics_20210625_123456.xml")
};

struct AnalogObject : PacBio::Configuration::PBConfig<AnalogObject>
{
    PB_CONFIG(AnalogObject);

    SMART_ENUM(BaseLabel_t, N, A, T, G, C);
    PB_CONFIG_PARAM(BaseLabel_t, baseLabel, BaseLabel_t::N); ///< The nucleotide that the analog is attached to EXAMPLE("C")
    PB_CONFIG_PARAM(double, relativeAmp, 1.0); ///< The relative amplitude in terms of pulse height EXAMPLE(0.3)
    PB_CONFIG_PARAM(double, interPulseDistanceSec, 0.0); ///< Average time in seconds between the falling edge of the previous pulse and rising edge of the next pulse EXAMPLE(0.14)
    PB_CONFIG_PARAM(double, excessNoiseCv, 0.0); ///< Coefficient of variation of excess noise EXAMPLE(3.0)
    PB_CONFIG_PARAM(double, pulseWidthSec, 0.0); ///< Average time in seconds of the width of pulses of this analog EXAMPLE(0.11)
    PB_CONFIG_PARAM(double, pw2SlowStepRatio, 0.15); ///< Rate constant ratio for two-step distribution of pulse width EXAMPLE(0.19)
    PB_CONFIG_PARAM(double, ipd2SlowStepRatio, 0.15); ///< Rate constant ratio for two-step distribution of interPulse distance EXAMPLE(0.14)
};


struct SocketBasecallerObject : PacBio::Configuration::PBConfig<SocketBasecallerObject>
{
    PB_CONFIG(SocketBasecallerObject);

    PB_CONFIG_PARAM(std::string, mid, ""); ///< Movie context ID used to create this object EXAMPLE("m123456_987654")
    PB_CONFIG_PARAM(std::string, uuid, ""); ///< subreadset UUID EXAMPLE("123e4567-e89b-12d3-a456-426614174000")

    PB_CONFIG_PARAM(uint64_t, movieMaxFrames, 0); ///< Movie length in frames EXAMPLE(3600000)
    PB_CONFIG_PARAM(double, movieMaxSeconds, 0); ///< Movie length in seconds EXAMPLE(36000)
    PB_CONFIG_PARAM(uint32_t, movieNumber, 0); ///< arbitrary movie number to delimite the start and end EXAMPLE(567)
    PB_CONFIG_PARAM(url, bazUrl, "discard:"); ///< Destination URL for the baz file EXAMPLE("http://localhost:23632/storages/m123456_987654/thefile.baz")
    PB_CONFIG_PARAM(url, traceFileUrl, "discard:"); ///< Destination URL for the trace file (optional) EXAMPLE("discard:")
    PB_CONFIG_PARAM(url, logUrl, "discard:"); ///< Destination URL for the log file EXAMPLE("http://localhost:23632/storages/m123456_987654/basecaller.log")
    PB_CONFIG_PARAM(LogLevel_t, logLevel, LogLevel_t::INFO); ///< log severity threshold EXAMPLE("DEBUG")
    PB_CONFIG_PARAM(ControlledString_t, chiplayout, ""); ///< controlled name of the sensor chip unit cell layout EXAMPLE("Minesweeper1.0") 
    PB_CONFIG_PARAM(url, darkcalFileUrl, ""); ///< Source URL for the dark calibration file EXAMPLE("http://localhost:23632/storages/m123456_987654/darkcal.h5")
    PB_CONFIG_PARAM(std::vector<std::vector<double>>, pixelSpreadFunction, 0); ///< This is required and a function of the sensor NFC tag EXAMPLE([[0.0,0.1,0.0],[0.1,0.6,0.1],[0.0,0.1,0.0]])
    PB_CONFIG_PARAM(std::vector<std::vector<double>>, crosstalkFilter, 0); ///< Optional kernel definition of the crosstalk deconvolution.  THe pixelSpreadFunction is used to automatically calculate one if this is not specified. EXAMPLE([[0.0,0.1,0.0],[0.1,0.6,0.1],[0.0,0.1,0.0]])
    PB_CONFIG_PARAM(std::vector<AnalogObject>, analogs, 0); ///< List of Analog objects, one analog for each nucleotide. Must be 4 elements for ATGC (order does not matter)
    PB_CONFIG_PARAM(std::vector<std::vector<int>>, sequencingRoi, 0); ///< ROI of the ZMWs that will be used for basecalling EXAMPLE(0,0,2048,1980)
    PB_CONFIG_PARAM(std::vector<std::vector<int>>, traceFileRoi, 0); ///< ROI of the ZMWs that will be used for trace file writing EXAMPLE(0,0,256,32)

    PB_CONFIG_PARAM(double, expectedFrameRate, 100.0); ///< The expected (not measured) canonical frame rate EXAMPLE(100.0)
    PB_CONFIG_PARAM(double, photoelectronSensitivity, 0.0); ///< The inversion of photoelectron gain of the sensor pixels. EXAMPLE(1.4)
    PB_CONFIG_PARAM(double, refSnr, 5.0); ///< Reference SNR EXAMPLE(10.0)

    PB_CONFIG_PARAM(url, simulationFileUrl, "discard:"); ///< Source URL for the file to use for transmission of simulated data. Only local files are supported currently. EXAMPLE("file://localhost/data/pa/sample_file.trc.h5")
    PB_CONFIG_PARAM(std::string, smrtBasecallerConfig, "{}"); ///< SmrtBasecallerConfig. Passed to smrt_basecaller --config. TODO: This will be a JSON object, but is a string here as a placeholder.

    PB_CONFIG_OBJECT(ProcessStatusObject, processStatus);       
    PB_CONFIG_OBJECT(SocketBasecallerRTMetricsObject, rtMetrics);
};

struct SocketObject : PacBio::Configuration::PBConfig<SocketObject>
{
    PB_CONFIG(SocketObject);

    PB_CONFIG_PARAM(std::string, socketId, "-1"); ///< The socket identifier, typically "1" thru "4". EXAMPLE("2")

    PB_CONFIG_OBJECT(SocketDarkcalObject, darkcal);
    PB_CONFIG_OBJECT(SocketLoadingcalObject, loadingcal);
    PB_CONFIG_OBJECT(SocketBasecallerObject, basecaller);
};

}}

#endif //include guard
