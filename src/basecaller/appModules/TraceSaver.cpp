// Copyright (c) 2020-2021, Pacific Biosciences of California, Inc.
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

#include <appModules/TraceSaver.h>

#include <pacbio/tracefile/TraceFile.h>
#include <pacbio/tracefile/TraceData.h>

#include <dataTypes/configs/AnalysisConfig.h>

namespace PacBio {
namespace Application {

using namespace PacBio::DataSource;
using namespace Mongo;
using namespace Mongo::Data;
 

TraceSaverBody::TraceSaverBody(const std::string& filename,
                               size_t numFrames,
                               DataSource::DataSourceBase::LaneSelector laneSelector,
                               const uint64_t frameBlockingSize,
                               const uint64_t zmwBlockingSize,
                               TraceFile::TraceDataType dataType,
                               const std::vector<uint32_t>& holeNumbers,
                               const std::vector<DataSource::DataSourceBase::UnitCellProperties>& properties,
                               const std::vector<uint32_t>& batchIds,
                               const boost::multi_array<float,2>& imagePsf,
                               const boost::multi_array<float,2>& crossTalk,
                               const Sensor::Platform& platform,
                               const std::string& instrumentName,
                               const Mongo::Data::AnalysisConfig& analysisConfig)
    : laneSelector_(std::move(laneSelector))
    , file_(
        // this lambda is a trick to call these chunking static functions before the file_ is constructed.
        // The `filename` argument is hijacked for the simple reason that it is the first argument
        // to the file_ constructor, but the filename is purely a spectator to this lambda.
        ([&](){
            if (frameBlockingSize > numFrames) throw PBException("frameBlockingSize must not be more than numFrames");
            if (zmwBlockingSize > holeNumbers.size()) throw PBException("zmwBlockingSize must not be more than numZmws (holeNumbers.size())");
            PacBio::TraceFile::TraceData::SetDefaultChunkZmwDim(zmwBlockingSize);
            PacBio::TraceFile::TraceData::SetDefaultChunkFrameDim(frameBlockingSize);
            return filename;
        }()),
        dataType,
        laneSelector_.size() * laneSize,
        numFrames)
{
    PopulateTraceData(holeNumbers, properties, batchIds, analysisConfig);
    PopulateScanData(numFrames, imagePsf, crossTalk, platform, instrumentName, analysisConfig);

    PBLOG_INFO << "TraceSaverBody created";
}

void TraceSaverBody::PopulateTraceData(const std::vector<uint32_t>& holeNumbers,
                                       const std::vector<DataSource::DataSourceBase::UnitCellProperties>& properties,
                                       const std::vector<uint32_t>& batchIds,
                                       const Mongo::Data::AnalysisConfig& analysisConfig)
{
    const size_t numZmw = laneSelector_.size() * laneSize;
    if (holeNumbers.size() != numZmw)
        throw PBException("Invalid number of hole numbers provided");
    if (properties.size() != numZmw)
        throw PBException("Invalid number of hole properties provided");
    if (batchIds.size() != numZmw)
        throw PBException("Invalid number of batchIds provided");

    // Now we need to populate all the actual data in the tracefile outside of the actual
    // traces
    boost::multi_array<int16_t, 2> holexy(boost::extents[numZmw][2]);
    std::vector<uint8_t> holeType(numZmw);
    for (size_t i = 0; i < numZmw; ++i)
    {
        // TODO: a conversion from "flags" to "holeType". The connection needs to be made here:
        //holetype[i] = properties[i].flags;
        holeType[i] = 0;
        // TODO change UnitCellProperties.x and y to be 16 bits to get rid of these casts
        holexy[i][0] = static_cast<int16_t>(properties[i].x);
        holexy[i][1] = static_cast<int16_t>(properties[i].y);
    }
    file_.Traces().Pedestal(analysisConfig.pedestal);
    file_.Traces().HoleXY(holexy);
    file_.Traces().HoleType(holeType);
    file_.Traces().HoleNumber(holeNumbers);
    file_.Traces().AnalysisBatch(batchIds);
}

void TraceSaverBody::PopulateScanData(size_t numFrames,
                                      const boost::multi_array<float,2>& imagePsf,
                                      const boost::multi_array<float,2>& crossTalk,
                                      const Sensor::Platform& platform,
                                      const std::string& instrumentName,
                                      const Mongo::Data::AnalysisConfig& analysisConfig)
{
    using ScanData = TraceFile::ScanData;

    ScanData::RunInfoData runInfo;
    runInfo.platformId = ScanData::RunInfoData::ToPlatformId(platform);
    runInfo.instrumentName = instrumentName;
    // FIXME: Stub in a default HQRFMethod for now.
    runInfo.hqrfMethod = "N2";
    file_.Scan().RunInfo(runInfo);

    ScanData::AcqParamsData acqParams;
    acqParams.aduGain = analysisConfig.movieInfo.photoelectronSensitivity;
    acqParams.frameRate = analysisConfig.movieInfo.frameRate;
    acqParams.numFrames = numFrames;
    file_.Scan().AcqParams(acqParams);

    // FIXME: For running on Sequel, the chip layout name is needed here.
    // We want to eventually move to storing directly into the trace file the
    // chip layout information.
    constexpr std::string_view defaultLayoutName = "KestrelPOCRTO3";

    ScanData::ChipInfoData chipInfo;
    chipInfo.layoutName = defaultLayoutName;
    chipInfo.analogRefSnr = analysisConfig.movieInfo.refSnr;
    chipInfo.imagePsf.resize(boost::extents[imagePsf.shape()[0]][imagePsf.shape()[1]]);
    chipInfo.imagePsf = imagePsf;
    chipInfo.xtalkCorrection.resize(boost::extents[crossTalk.shape()[0]][crossTalk.shape()[1]]);
    chipInfo.xtalkCorrection = crossTalk;
    file_.Scan().ChipInfo(chipInfo);

    ScanData::DyeSetData dyeSet;
    const size_t numAnalogs = analysisConfig.movieInfo.analogs.size();
    dyeSet.numAnalog = static_cast<uint16_t>(numAnalogs);
    dyeSet.relativeAmp.resize(numAnalogs);
    dyeSet.excessNoiseCV.resize(numAnalogs);
    dyeSet.ipdMean.resize(numAnalogs);
    dyeSet.pulseWidthMean.resize(numAnalogs);
    dyeSet.pw2SlowStepRatio.resize(numAnalogs);
    dyeSet.ipd2SlowStepRatio.resize(numAnalogs);
    dyeSet.baseMap = "";
    for (size_t i = 0; i < numAnalogs; i++)
    {
        const auto& am = analysisConfig.movieInfo.analogs[i];
        dyeSet.relativeAmp[i] = am.relAmplitude;
        dyeSet.excessNoiseCV[i] = am.excessNoiseCV;
        dyeSet.ipdMean[i] = am.interPulseDistance;
        dyeSet.pulseWidthMean[i] = am.pulseWidth;
        dyeSet.pw2SlowStepRatio[i] = am.pw2SlowStepRatio;
        dyeSet.ipd2SlowStepRatio[i] = am.ipd2SlowStepRatio;
        dyeSet.baseMap += am.baseLabel;
    }
    file_.Scan().DyeSet(dyeSet);

}

void TraceSaverBody::Process(const Mongo::Data::TraceBatchVariant& traceVariant)
{
    auto writeTraces = [&](const auto& traceBatch)
    {
        using T = typename std::remove_reference_t<decltype(traceBatch)>::HostType;

        const auto zmwOffset = traceBatch.Metadata().FirstZmw();
        const auto frameOffset = traceBatch.Metadata().FirstFrame();
        const DataSourceBase::LaneIndex laneBegin = zmwOffset / traceBatch.LaneWidth();
        const DataSourceBase::LaneIndex laneEnd = laneBegin + traceBatch.LanesPerBatch();

        for (const auto laneIdx : laneSelector_.SelectedLanes(laneBegin, laneEnd))
        {
            PBLOG_DEBUG << "TraceSaverBody::Process, laneIdx" << laneIdx;
            const auto blockIdx = laneIdx - laneBegin;
            Mongo::Data::BlockView<const T> blockView = traceBatch.GetBlockView(blockIdx);
#if 0
            PBLOG_NOTICE << "blockView data, zmwOffset:" << zmwOffset << "frameOffset:" << frameOffset;
            for(uint32_t x=0;x<32;x++) 
            {
                PBLOG_NOTICE <<  std::hex << blockView.Data()[x];
            }
#endif

            // The TraceFile::Traces API uses transposed blocks of data. The traceBatch data needs to be transposed to
            // work with the API.  TODO: perform this transpose inside the TraceFile::Traces() class.
            boost::const_multi_array_ref<T, 2> data {
                blockView.Data(), boost::extents[blockView.NumFrames()][blockView.LaneWidth()]};

            typedef boost::multi_array<T, 2> array_ref;
            array_ref transpose {boost::extents[blockView.LaneWidth()][blockView.NumFrames()]};
            for (uint32_t iframe = 0; iframe < blockView.NumFrames(); iframe++)
            {
                for (uint32_t izmw = 0; izmw < blockView.LaneWidth(); izmw++)
                {
                    transpose[izmw][iframe] = data[iframe][izmw];
                }
            }

            // this is messy and could be improved. It simply does a lookup of the laneIdx to get the lane offset within
            // the trace file. TODO
            // (MTL) I think this would better be implemented by having the laneSelector_.SelectedLanes() return a
            // std::pair<int,int> where the first index is the index within the selected lanes container, and the second
            // is the actual lane.  Then the `traceFileLane` just below is `iterator->first`, and `laneIdx = iterator->second`.
            // This will allow laneSelector_.SelectedLanes() to randomly interate.
            auto position = std::lower_bound(laneSelector_.begin(), laneSelector_.end(), laneIdx);
            const int64_t traceFileLane = position - laneSelector_.begin();

            const int64_t traceFileZmwOffset = traceFileLane * traceBatch.LaneWidth();
            boost::array<typename array_ref::index, 2> bases = {{traceFileZmwOffset, frameOffset}};
            transpose.reindex(bases);
            file_.Traces().WriteTraceBlock<T>(transpose);

        }
    };

    std::visit(writeTraces, traceVariant.Data());
}


}  // namespace Application
}  // namespace PacBio
