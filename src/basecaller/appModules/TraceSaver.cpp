// Copyright (c) 2020, Pacific Biosciences of California, Inc.
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
        for(uint32_t i=0;i<numZMWs;i++)
        {
            holexy[i][0] = features[i].x;
            holexy[i][1] = features[i].y;
            holeType[i] = features[i].flags; // FIXME there should be a conversion between bits and enumeration
        }
        writer_->Traces().HoleXY(holexy);
        writer_->Traces().HoleType(holeType);

#if 0
        std::vector<uint32_t> holeNumber;
        writer_->Traces().HoleNumber(holeNumber);
#endif
    }
    PBLOG_INFO << "TraceSaverBody created";
}


void TraceSaverBody::Process(const Mongo::Data::TraceBatch<int16_t>& traceBatch)
{
    if (writer_)
    {
        const auto zmwOffset = traceBatch.Metadata().FirstZmw();
        const auto frameOffset = traceBatch.Metadata().FirstFrame();
        const DataSourceBase::LaneIndex laneBegin = zmwOffset / traceBatch.LaneWidth();
        const DataSourceBase::LaneIndex laneEnd = laneBegin + traceBatch.LanesPerBatch();

#if 0
        if (zmwOffset < 10000)
        {
            PBLOG_INFO << "TraceSaverBody::Process, zmwOffset:" << zmwOffset << " frameOffset:" << frameOffset << " laneBegin:" << laneBegin << " laneEnd:" << laneEnd;
        }
#endif

        for (const auto laneIdx : laneSelector_.SelectedLanes(laneBegin, laneEnd))
        {
#if 1
            PBLOG_INFO << "TraceSaverBody::Process, laneIdx" << laneIdx;
#endif
            const auto blockIdx = laneIdx - laneBegin;
            Mongo::Data::BlockView<const int16_t> blockView = traceBatch.GetBlockView(blockIdx);
#if 1
            // do the transpose here.  All of this is a bit ugly and could be compartmentalized I think. TODO
            boost::const_multi_array_ref<int16_t, 2> data {
                blockView.Data(), boost::extents[blockView.NumFrames()][blockView.LaneWidth()]};

            typedef boost::multi_array<int16_t, 2> array_ref;
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
            auto position = std::lower_bound(laneSelector_.begin(), laneSelector_.end(), laneIdx);
            const int64_t traceFileLane = position - laneSelector_.begin();

            const int64_t traceFileZmwOffset = traceFileLane * traceBatch.LaneWidth();
            boost::array<array_ref::index, 2> bases = {{traceFileZmwOffset, frameOffset}};
            transpose.reindex(bases);
            writer_->Traces().WriteTraceBlock<int16_t>(transpose);
#else
            // no transpose
            {
                typedef boost::const_multi_array_ref<int16_t, 2> array_ref;
                array_ref data {blockView.Data(), boost::extents[blockView.LaneWidth()][blockView.NumFrames()]};
                boost::array<array_ref::index, 2> bases = {{laneOffset + laneIdx, frameOffset}};
                data.reindex(bases);
                writer_->Traces().WriteTraceBlock(data);
            }
#endif
        }
    }
    else
    {
        throw PBException("fix me, the tracwriter was not constructed");
    }
}


}  // namespace Application
}  // namespace PacBio