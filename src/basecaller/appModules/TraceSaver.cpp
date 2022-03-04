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

#include <pacbio/file/TraceFile.h>
#include <pacbio/file/TraceData.h>

#include <dataTypes/configs/AnalysisConfig.h>

namespace PacBio {
namespace Application {

using namespace PacBio::DataSource;
using namespace PacBio::File;

using namespace Mongo;
using namespace Mongo::Data;
 

TraceSaverBody::TraceSaverBody(const std::string& filename,
                               uint64_t numFrames,
                               DataSource::DataSourceBase::LaneSelector laneSelector,
                               const uint64_t frameBlockingSize,
                               const uint64_t zmwBlockingSize,
                               File::TraceDataType dataType,
                               const std::vector<uint32_t>& holeNumbers,
                               const std::vector<DataSource::DataSourceBase::UnitCellProperties>& properties,
                               const std::vector<uint32_t>& batchIds,
                               const File::ScanData::Data& experimentMetadata,
                               const Mongo::Data::AnalysisConfig& analysisConfig)
    : laneSelector_(std::move(laneSelector))
    , file_(
        // this lambda is a trick to call these chunking static functions before the file_ is constructed.
        // The `filename` argument is hijacked for the simple reason that it is the first argument
        // to the file_ constructor, but the filename is purely a spectator to this lambda.
        ([&](){
            if (frameBlockingSize > numFrames) throw PBException("frameBlockingSize must not be more than numFrames");
            if (zmwBlockingSize > holeNumbers.size()) throw PBException("zmwBlockingSize must not be more than numZmws (holeNumbers.size())");
            PacBio::File::TraceData::SetDefaultChunkZmwDim(zmwBlockingSize);
            PacBio::File::TraceData::SetDefaultChunkFrameDim(frameBlockingSize);
            return filename;
        }()),
        dataType,
        laneSelector_.size() * laneSize,
        numFrames)
    , numFrames_(numFrames)    
{
    PopulateTraceData(holeNumbers, properties, batchIds, analysisConfig);
    PopulateScanData(experimentMetadata);

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
    std::vector<uint32_t> holeFeaturesMask(numZmw);
    for (size_t i = 0; i < numZmw; ++i)
    {
        holeType[i] = properties[i].type;
        holeFeaturesMask[i] = properties[i].flags;
        // TODO change UnitCellProperties.x and y to be 16 bits to get rid of these casts
        holexy[i][0] = static_cast<int16_t>(properties[i].x);
        holexy[i][1] = static_cast<int16_t>(properties[i].y);
    }
    file_.Traces().Pedestal(analysisConfig.pedestal);
    file_.Traces().HoleXY(holexy);
    file_.Traces().HoleType(holeType);
    file_.Traces().HoleFeaturesMask(holeFeaturesMask);
    file_.Traces().HoleNumber(holeNumbers);
    file_.Traces().AnalysisBatch(batchIds);
}

void TraceSaverBody::PopulateScanData(const ScanData::Data& experimentMetadata)
{
    file_.Scan().RunInfo(experimentMetadata.runInfo);
    file_.Scan().AcqParams(experimentMetadata.acqParams);
    file_.Scan().ChipInfo(experimentMetadata.chipInfo);
    file_.Scan().DyeSet(experimentMetadata.dyeSet);
    file_.Scan().AcquisitionXML(experimentMetadata.acquisitionXML);
}

void TraceSaverBody::Process(const Mongo::Data::TraceBatchVariant& traceVariant)
{
    auto writeTraces = [&](const auto& traceBatch)
    {
        using T = typename std::remove_reference_t<decltype(traceBatch)>::HostType;

        if (traceBatch.Metadata().FirstFrame() < 0) throw PBException("First frame can not be negative");
        const auto zmwOffset = traceBatch.Metadata().FirstZmw();
        const uint32_t frameOffset = traceBatch.Metadata().FirstFrame();
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

            // The File::Traces API uses transposed blocks of data. The traceBatch data needs to be transposed to
            // work with the API.  TODO: perform this transpose inside the File::Traces() class.
            boost::const_multi_array_ref<T, 2> data {
                blockView.Data(), boost::extents[blockView.NumFrames()][blockView.LaneWidth()]};

            typedef boost::multi_array<T, 2> array_ref;
            uint64_t traceBlockFrames = blockView.NumFrames();
            if (frameOffset > numFrames_)
            {
                PBLOG_ERROR << "The frameOffset=" << frameOffset << " of the current chunk lies past the end of the trace file,"
                    << " numFrames:" << numFrames_ << ". Skipping this chunk";
                traceBlockFrames = 0;
            } 
            else if (frameOffset + traceBlockFrames > numFrames_)
            {
                // limit the frames to the original requested size
                traceBlockFrames = numFrames_ - frameOffset;
            }
            if (traceBlockFrames>0)
            {
                array_ref transpose {boost::extents[blockView.LaneWidth()][traceBlockFrames]};
                for (uint32_t iframe = 0; iframe < traceBlockFrames; iframe++)
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
        }
    };

    std::visit(writeTraces, traceVariant.Data());
}


}  // namespace Application
}  // namespace PacBio
