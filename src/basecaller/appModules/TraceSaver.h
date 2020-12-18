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

#ifndef PACBIO_APPLICATION_TRACE_SAVER_H
#define PACBIO_APPLICATION_TRACE_SAVER_H

#include <pacbio/datasource/PacketLayout.h>
#include <pacbio/datasource/SensorPacket.h>
#include <pacbio/tracefile/DataFileInterface.h>
#include <pacbio/tracefile/TraceFile.h>
#include <pacbio/logging/Logger.h>
#include <pacbio/sensor/SequelROI.h>

#include <common/graphs/GraphNodeBody.h>

#include <dataTypes/TraceBatch.h>
#include <dataTypes/configs/ROIConfig.h>

#include <boost/multi_array.hpp>

namespace PacBio {
namespace Application {

class NoopTraceSaverBody final : public Graphs::LeafBody<const Mongo::Data::TraceBatch<int16_t>>
{
public:
    size_t ConcurrencyLimit() const override { return 1; }
    float MaxDutyCycle() const override { return 0.01; }

    void Process(const Mongo::Data::TraceBatch<int16_t>&) override
    {

    }
};

class TraceSaverBody final : public Graphs::LeafBody<const Mongo::Data::TraceBatch<int16_t>>
{
public:
    TraceSaverBody(std::unique_ptr<PacBio::TraceFile::DataFileWriterInterface>&& writer,
                   std::unique_ptr<PacBio::Sensor::SequelROI> roi,
                   const std::vector<uint32_t> blockIndices)
        : writer_(std::move(writer))
        , roi_(std::move(roi))
        , blockIndices_(blockIndices)
    {
        auto* tracewriter = dynamic_cast<PacBio::TraceFile::TraceFileWriter*>(writer_.get());
        if (tracewriter) {
            const uint32_t numZMWs = tracewriter->Traces().NumZmws();
            if (numZMWs !=roi_->CountZMWs())
            {
                PBLOG_WARN << "ROI ZMWS is not equal to trace file ZMWs. Something is not right. " <<
                           "trace.NumZmws:" << numZMWs << " ROI ZMWS:" << roi_->CountZMWs();
            }
            boost::multi_array<int16_t,2> holexy(boost::extents[numZMWs][2]);
            PacBio::Sensor::SequelROI::Enumerator e(*roi_,0,numZMWs);
            for(;e.Index() != numZMWs; e++)
            {
                const int i = e.Index();
                holexy[i][0] = static_cast<int16_t>(e.GetPixelRow().Value());
                holexy[i][1] = static_cast<int16_t>(e.GetPixelCol().Value());
            }
            tracewriter->Traces().HoleXY(holexy);
        }
        PBLOG_INFO << "TraceSaverBody created with ";
        PacBio::Logging::LogStream logstream;
        writer_->OutputSummary(logstream);
    }

    size_t ConcurrencyLimit() const override { return 1; }
    float MaxDutyCycle() const override { return 0.01; }

    void Process(const Mongo::Data::TraceBatch<int16_t>& traceBatch) override
    {
        // FIX ME. this is a hack and no polymorphic, but I want to get this functional first.
        // The dynamic_cast should be removed, and possibly there should be a WriteTraceBatch()
        // method written in the interface.

        auto* tracewriter = dynamic_cast<PacBio::TraceFile::TraceFileWriter*>(writer_.get());
        if (tracewriter)
        {
            const auto zmwOffset = traceBatch.Metadata().FirstZmw();
            const auto frameOffset = traceBatch.Metadata().FirstFrame();

            const uint32_t laneOffset = zmwOffset / traceBatch.LaneWidth();
            const uint32_t maxLanesOffset = laneOffset + traceBatch.LanesPerBatch();

            // loops over all lanes selected by the ROI
            for(const uint32_t laneIdx : blockIndices_)
            {
                // checking to see that these blocks are in the current batch bounds
                if (laneIdx >= laneOffset && laneIdx < maxLanesOffset)
                {
                    Mongo::Data::BlockView<const int16_t> blockView = traceBatch.GetBlockView(laneIdx);
#if 1
                    // do the transpose here.
                    boost::const_multi_array_ref<int16_t, 2>
                        data{blockView.Data(), boost::extents[blockView.NumFrames()][blockView.LaneWidth()]};

                    typedef boost::multi_array<int16_t, 2> array_ref;
                    array_ref transpose{boost::extents[blockView.LaneWidth()][blockView.NumFrames()]};
                    for(uint32_t iframe=0;iframe < blockView.NumFrames(); iframe++)
                    {
                        for(uint32_t izmw = 0;izmw < blockView.LaneWidth(); izmw++)
                        {
                            transpose[izmw][iframe] = data[iframe][izmw];
                        }
                    }
                    boost::array<array_ref::index, 2> bases = {{laneOffset + laneIdx, frameOffset}};
                    transpose.reindex(bases);
                    tracewriter->Traces().WriteTraceBlock<int16_t>(transpose);
#else
                    // no transpose
                    {
                        typedef boost::const_multi_array_ref<int16_t, 2> array_ref;
                        array_ref data{blockView.Data(), boost::extents[blockView.LaneWidth()][blockView.NumFrames()]};
                        boost::array<array_ref::index, 2> bases = {{laneOffset + laneIdx, frameOffset}};
                        data.reindex(bases);
                        tracewriter->Traces().WriteTraceBlock(data);
                    }
#endif
                }
            }
        }
        else
        {
            throw PBException("fix me");
        }
    }
private:
    std::unique_ptr<PacBio::TraceFile::DataFileWriterInterface> writer_;
    std::unique_ptr<PacBio::Sensor::SequelROI> roi_;
    std::vector<uint32_t> blockIndices_;
};

}}

#endif //PACBIO_APPLICATION_TRACE_SAVER_H
