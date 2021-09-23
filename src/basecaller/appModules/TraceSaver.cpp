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

#include <pacbio/tracefile/TraceFile.h>
#include <appModules/TraceSaver.h>

namespace PacBio {
namespace Application {

using namespace PacBio::DataSource;

TraceSaverBody::TraceSaverBody(std::unique_ptr<PacBio::TraceFile::TraceFile>&& writer,
                               const std::vector<PacBio::DataSource::DataSourceBase::UnitCellProperties>& features,
                               PacBio::DataSource::DataSourceBase::LaneSelector&& laneSelector)
    : writer_(std::move(writer))
    , laneSelector_(std::move(laneSelector))
{
    if (writer_)
    {
        const uint32_t numZMWs = writer_->Traces().NumZmws();
        if (numZMWs != features.size())
        {
            PBLOG_ERROR << "ROI ZMWS is not equal to trace file ZMWs. Something is not right. "
                        << "trace.NumZmws:" << numZMWs << " feature ZMWS:" <<  features.size();
            throw PBException("ROI miscalculation");
        }
        boost::multi_array<int16_t, 2> holexy(boost::extents[numZMWs][2]);
        std::vector<uint8_t> holeType(numZMWs);
        // TODO: add holeNumber to the UnitCellProperties, then uncomment the related lines below.
        // std::vector<uint32_t> holeNumber;
        for(uint32_t i=0;i<numZMWs;i++)
        {
            holexy[i][0] = static_cast<int16_t>(features[i].x); // TODO change UnitCellProperties.x and y to be 16 bits to get rid of these casts
            holexy[i][1] = static_cast<int16_t>(features[i].y);
            // TODO: a conversion from "flags" to "holeType". The connection needs to be made here:
            (void) features[i].flags;
            holeType[i] = 0;
            // TODO: holeNumber[i] = features[i].holeNumber;
        }
        writer_->Traces().HoleXY(holexy);
        writer_->Traces().HoleType(holeType);
        // TODO: writer_->Traces().HoleNumber(holeNumber);
    }
    PBLOG_INFO << "TraceSaverBody created";
}


void TraceSaverBody::Process(const Mongo::Data::TraceBatchVariant& traceVariant)
{
    auto writeTraces = [&](const auto& traceBatch)
    {
        using T = typename std::remove_reference_t<decltype(traceBatch)>::DataType;

        const auto zmwOffset = traceBatch.Metadata().FirstZmw();
        const auto frameOffset = traceBatch.Metadata().FirstFrame();
        const DataSourceBase::LaneIndex laneBegin = zmwOffset / traceBatch.LaneWidth();
        const DataSourceBase::LaneIndex laneEnd = laneBegin + traceBatch.LanesPerBatch();

        for (const auto laneIdx : laneSelector_.SelectedLanes(laneBegin, laneEnd))
        {
            PBLOG_DEBUG << "TraceSaverBody::Process, laneIdx" << laneIdx;
            const auto blockIdx = laneIdx - laneBegin;
            Mongo::Data::BlockView<const T> blockView = traceBatch.GetBlockView(blockIdx);

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
            writer_->Traces().WriteTraceBlock<T>(transpose);

        }
    };

    if (writer_)
    {
        std::visit(writeTraces, traceVariant);
    }
    else
    {
        throw PBException("fix me, the tracwriter was not constructed");
    }
}


}  // namespace Application
}  // namespace PacBio
