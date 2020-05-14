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

#ifndef PACBIO_APPLICATION_REPACKER_H
#define PACBIO_APPLICATION_REPACKER_H

#include <pacbio/datasource/PacketLayout.h>
#include <pacbio/datasource/SensorPacket.h>

#include <common/graphs/GraphNodeBody.h>

#include <dataTypes/TraceBatch.h>

namespace PacBio {
namespace Application {

// Repacker implementation that does nothing but repackage a SensorPacket into a
// trace batch.  Requires the SensorPacket to have the correct dimensions
class TrivialRepackerBody final : public Graphs::MultiTransformBody<DataSource::SensorPacket, const Mongo::Data::TraceBatch<int16_t>>
{
public:
    TrivialRepackerBody(Mongo::Data::BatchDimensions expectedDims)
        : expectedDims_(expectedDims)
    {}

    size_t ConcurrencyLimit() const override { return 1; }
    float MaxDutyCycle() const override { return 0.01; }

    void Process(DataSource::SensorPacket packet) override
    {
        // Make sure packet is valid
        if (packet.Layout().Encoding() != DataSource::PacketLayout::INT16)
            throw PBException("TrivialRepacker only supports INT16 encoding");
        if (packet.Layout().Type() != DataSource::PacketLayout::BLOCK_LAYOUT_DENSE)
            throw PBException("TrivialRepacker only supports BLOCK_LAYOUT_DENSE");
        if (expectedDims_.lanesPerBatch != packet.Layout().NumBlocks())
            throw PBException("TrivialRepacker expected " + std::to_string(expectedDims_.lanesPerBatch) +
                              " blocks but received " + std::to_string(packet.Layout().NumBlocks()));
        if (expectedDims_.framesPerBatch != packet.Layout().NumFrames())
            throw PBException("TrivialRepacker expected " + std::to_string(expectedDims_.framesPerBatch) +
                              " frames but received " + std::to_string(packet.Layout().NumFrames()));
        if (expectedDims_.laneWidth != packet.Layout().BlockWidth())
            throw PBException("TrivialRepacker expected " + std::to_string(expectedDims_.laneWidth) +
                              " blockWidth but received " + std::to_string(packet.Layout().BlockWidth()));

        Mongo::Data::BatchMetadata meta(
            packet.StartZmw() / (expectedDims_.lanesPerBatch * expectedDims_.laneWidth),
            packet.StartFrame(),
            packet.StartFrame() + packet.NumFrames(),
            packet.StartZmw());
        PushOut(Mongo::Data::TraceBatch<int16_t>(std::move(packet),
                                    meta,
                                    expectedDims_,
                                    PacBio::Cuda::Memory::SyncDirection::HostWriteDeviceRead,
                                    SOURCE_MARKER()));
    }
private:
    Mongo::Data::BatchDimensions expectedDims_;
};


}}

#endif //PACBIO_APPLICATION_REPACKER_H
