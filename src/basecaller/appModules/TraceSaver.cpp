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

namespace PacBio::Application {

using namespace PacBio::DataSource;
using namespace PacBio::File;

using namespace Mongo;
using namespace Mongo::Data;

TracePrepBody::TracePrepBody(DataSource::DataSourceBase::LaneSelector laneSelector,
                             uint64_t maxFrames)
    : laneSelector_(std::move(laneSelector))
    , maxFrames_(maxFrames)
{}

TraceSaverBody::TraceSaverBody(const std::string& filename,
                               uint64_t numFrames,
                               const uint64_t frameBlockingSize,
                               const uint64_t zmwBlockingSize,
                               File::TraceDataType dataType,
                               const std::vector<uint32_t>& holeNumbers,
                               const std::vector<DataSource::DataSourceBase::UnitCellProperties>& properties,
                               const std::vector<uint32_t>& batchIds,
                               const File::ScanData::Data& experimentMetadata,
                               const Mongo::Data::AnalysisConfig& analysisConfig,
                               uint32_t maxQueueSize)
    : file_(
        // this lambda is a trick to call these chunking static functions before the file_ is constructed.
        // The `filename` argument is hijacked for the simple reason that it is the first argument
        // to the file_ constructor, but the filename is purely a spectator to this lambda.
        ([&](){
            PacBio::File::TraceData::SetDefaultChunkZmwDim(std::min(zmwBlockingSize, holeNumbers.size()));
            PacBio::File::TraceData::SetDefaultChunkFrameDim(std::min(frameBlockingSize, numFrames));
            return filename;
        }()),
        dataType,
        holeNumbers.size(),
        numFrames)
    , numFrames_(numFrames)
    , numZmw_(holeNumbers.size())
    , maxQueueSize_(maxQueueSize)
{
    PopulateTraceData(holeNumbers, properties, batchIds, analysisConfig);
    PopulateScanData(experimentMetadata);

    if (maxQueueSize_ > 0)
    {
        enableWriterThread_ = true;
        writeFuture_ = std::async(std::launch::async, [this]()
        {
            uint64_t unitCellsWritten = 0;
            PreppedTracesVariant data;
            while (enableWriterThread_)
            {
                if (queue_.Pop(data, std::chrono::milliseconds{100}))
                {
                    std::visit([&](auto&& traces)
                    {
                        using T = std::remove_pointer_t<decltype(traces->data())>;
                        file_.Traces().WriteTraceBlock<T>(*traces);
                        unitCellsWritten += traces->size();
                    }, data);
                }
            }
            return unitCellsWritten;
        });
    };

    PBLOG_INFO << "TraceSaverBody created";
}

void TraceSaverBody::PopulateTraceData(const std::vector<uint32_t>& holeNumbers,
                                       const std::vector<DataSource::DataSourceBase::UnitCellProperties>& properties,
                                       const std::vector<uint32_t>& batchIds,
                                       const Mongo::Data::AnalysisConfig& analysisConfig)
{
    const size_t numZmw = holeNumbers.size();
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

PreppedTracesVariant TracePrepBody::Process(const Mongo::Data::TraceBatchVariant& traceVariant)
{
    auto writeTraces = [&](const auto& traceBatch) -> PreppedTracesVariant
    {
        using T = typename std::remove_reference_t<decltype(traceBatch)>::HostType;

        if (traceBatch.Metadata().FirstFrame() < 0) throw PBException("First frame can not be negative");
        const auto zmwOffset = traceBatch.Metadata().FirstZmw();
        const uint32_t frameOffset = traceBatch.Metadata().FirstFrame();
        const DataSourceBase::LaneIndex laneBegin = zmwOffset / traceBatch.LaneWidth();
        const DataSourceBase::LaneIndex laneEnd = laneBegin + traceBatch.LanesPerBatch();

        uint64_t traceBlockFrames = traceBatch.NumFrames();
        if (frameOffset > maxFrames_)
        {
            PBLOG_ERROR << "The frameOffset=" << frameOffset << " of the current chunk lies past the end of the trace file,"
                << " numFrames:" << maxFrames_ << ". Skipping this chunk";
            traceBlockFrames = 0;
        }
        else if (frameOffset + traceBlockFrames > maxFrames_)
        {
            // limit the frames to the original requested size
            traceBlockFrames = maxFrames_ - frameOffset;
        }

        const auto& selection = laneSelector_.SelectedLanes(laneBegin, laneEnd);
        uint32_t traceFileStartLane = std::lower_bound(laneSelector_.begin(), laneSelector_.end(), *selection.begin()) - laneSelector_.begin();
        uint32_t pos = 0;
        auto ret = std::make_unique<boost::multi_array<T, 2>>(boost::extents[traceBatch.LaneWidth() * selection.size()][traceBatch.NumFrames()]);
        for (const auto laneIdx : selection)
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

            if (traceBlockFrames > 0)
            {
                auto startZmwIdx = pos * blockView.LaneWidth();
                for (uint32_t iframe = 0; iframe < traceBlockFrames; iframe++)
                {
                    for (uint32_t izmw = 0; izmw < blockView.LaneWidth(); izmw++)
                    {
                        (*ret)[startZmwIdx + izmw][iframe] = data[iframe][izmw];
                    }
                }
                pos++;
            }
        }
        boost::array<boost::multi_array_types::index, 2> bases;
        bases[0] = traceFileStartLane * traceBatch.LaneWidth();
        bases[1] = frameOffset;
        ret->reindex(bases);
        return ret;
    };

    return std::visit(writeTraces, traceVariant.Data());
}

void TraceSaverBody::Process(PreppedTracesVariant traceVariant)
{
    auto writeTraces = [&](auto& traces)
    {
        using T = std::remove_pointer_t<decltype(traces->data())>;
        if (traces->shape()[1] > 0)
        {
            if (traces->shape()[0] + traces->index_bases()[0] > numZmw_ ||
                traces->shape()[1] + traces->index_bases()[1] > numFrames_)
            {
                throw PBException("Received trace data does not fit inside dimensions of trace file");
            }

            if (enableWriterThread_)
            {
                uint32_t waitCount = 0;
                while (queue_.Size() >= maxQueueSize_)
                {
                    if (waitCount % 10 == 0)
                    {
                        assert(writeFuture_.valid());
                        if (writeFuture_.wait_for(std::chrono::milliseconds{0}) != std::future_status::timeout)
                        {
                            PBLOG_ERROR << "Trace Saving queue is full and the dedicated thread has died";
                            PBLOG_ERROR << "Checking for exception...";
                            assert(writeFuture_.valid());
                            auto writtenPixels = writeFuture_.get();
                            PBLOG_ERROR << "No Exception found.  Thread terminated early after writing "
                                        << writtenPixels << " unitCells";
                            throw PBException("Unknown failure in TraceSaver");
                        }
                        PBLOG_WARN << "Trace Saving queue is full... Sleeping!";
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds{100});
                    waitCount++;
                }
                queue_.Push(std::move(traces));
            }
            else
            {
                file_.Traces().WriteTraceBlock<T>(*traces);
            }
        }
    };
    std::visit(writeTraces, traceVariant);
}

TraceSaverBody::~TraceSaverBody()
{
    if (enableWriterThread_)
    {
        while (!queue_.Empty())
        {
            PBLOG_INFO << "Waiting for trace writing to complete";
            std::this_thread::sleep_for(std::chrono::seconds{1});
        }
        enableWriterThread_ = false;
        try
        {
            PBLOG_INFO << "Waiting for SaverThread to complete";
            assert(writeFuture_.valid());
            auto unitCellsWritten = writeFuture_.get();
            if (unitCellsWritten == numFrames_ * numZmw_)
                PBLOG_INFO << "SaverThread done, TraceFile is complete";
            else
                PBLOG_WARN << "SaverThread finished without error, but only wrote " << unitCellsWritten << " unitCells out of " << numFrames_ * numZmw_;
        }
        catch (const std::exception& e)
        {
            PBLOG_ERROR << "Exception in TraceSaver thread: ";
            PBLOG_ERROR << e.what();
            PBLOG_ERROR << "We're in a destructor, so swallowing exception...";
        }
    }
}


}  // namespace PacBio::Application::TraceSaver
