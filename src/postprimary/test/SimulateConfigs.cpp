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

#include <pacbio/tracefile/ScanData.h>
#include <dataTypes/configs/SmrtBasecallerConfig.h>

#include "SimulateConfigs.h"

namespace PacBio::Primary::Postprimary
{

std::string generateExperimentMetadata(size_t psfSize, size_t xtalkSize)
{
    using ScanData = PacBio::TraceFile::ScanData;
    ScanData::Data expMetadata;

    auto& runInfo = expMetadata.runInfo;
    runInfo.platformId = ScanData::RunInfoData::PlatformId::Kestrel;
    runInfo.instrumentName = "testInstrument";
    runInfo.hqrfMethod = "N2";

    auto& chipInfo = expMetadata.chipInfo;
    chipInfo.layoutName = "KestrelPOCRTO3";

    auto MakeUnity = [](boost::multi_array<float, 2>& ma, size_t dimSize) {
        ma.resize(boost::extents[dimSize][dimSize]);
        ma[dimSize / 2][dimSize / 2] = 1.0f;
    };

    MakeUnity(chipInfo.imagePsf, psfSize);
    MakeUnity(chipInfo.xtalkCorrection, xtalkSize);

    auto& dyeSet = expMetadata.dyeSet;
    const size_t numAnalogs = 4;
    dyeSet.numAnalog = static_cast<uint16_t>(numAnalogs);
    dyeSet.relativeAmp = {1.0f, 0.946f, 0.529f, 0.553f};
    dyeSet.excessNoiseCV = {0.1f, 0.1f, 0.1f, 0.1f};
    dyeSet.ipdMean = {0.2f, 0.3f, 0.4f, 0.5f};
    dyeSet.pulseWidthMean = {0.1f, 0.2f, 0.3f, 0.4f};
    dyeSet.pw2SlowStepRatio = {0, 0, 0, 0};
    dyeSet.ipd2SlowStepRatio = {0, 0, 0, 0};
    dyeSet.baseMap = "CTAG";

    expMetadata.acquisitionXML = "<acquisitionXML>test<acquisitionXML>";

    return expMetadata.Serialize().toStyledString();
}

std::string generateBasecallerConfig(bool internal)
{
    PacBio::Mongo::Data::SmrtBasecallerConfig basecallerConfig;
    basecallerConfig.internalMode = internal;
    return basecallerConfig.Serialize().toStyledString();
}

}