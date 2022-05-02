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

#ifndef PACBIO_APPLICATION_TRIVIAL_REPACKER_H
#define PACBIO_APPLICATION_TRIVIAL_REPACKER_H

#include <pacbio/datasource/PacketLayout.h>
#include <pacbio/datasource/SensorPacket.h>

#include <appModules/Repacker.h>

#include <common/graphs/GraphNodeBody.h>

#include <dataTypes/TraceBatch.h>

namespace PacBio {
namespace Application {

// Repacker implementation that does nothing but repackage a SensorPacket into a
// trace batch.  Requires the SensorPacket to have the correct dimensions
class TrivialRepackerBody final : public RepackerBody
{
public:
    TrivialRepackerBody(const std::map<uint32_t, DataSource::PacketLayout>& expectedLayouts)
    {
        for (const auto& kv : expectedLayouts)
        {
            auto id = kv.first;
            const auto& layout = kv.second;
            if (layout.Type() == DataSource::PacketLayout::FRAME_LAYOUT)
                throw PBException("Frame layout not supported");
            if (layout.Encoding() != DataSource::PacketLayout::INT16
                && layout.Encoding() != DataSource::PacketLayout::UINT8)
            {
                throw PBException("Unsupported layout encoding for TrivialRepacker");
            }
            if (layout.BlockWidth() != Mongo::laneSize)
                throw PBException("Invalid block width");
            Mongo::Data::BatchDimensions dims;
            dims.framesPerBatch = layout.NumFrames();
            dims.laneWidth = layout.BlockWidth();
            dims.lanesPerBatch = layout.NumBlocks();

            expectedDims_[id] = dims;
        }

    }

    std::map<uint32_t, Mongo::Data::BatchDimensions> BatchLayouts() const override
    {
        return expectedDims_;
    }

    size_t ConcurrencyLimit() const override { return 1; }
    float MaxDutyCycle() const override { return 0.01; }

    void Process(DataSource::SensorPacket packet) override
    {
        const auto& dims = expectedDims_[packet.PacketID()];
        // Make sure packet is valid
        if (packet.Layout().Encoding() != DataSource::PacketLayout::INT16
            && packet.Layout().Encoding() != DataSource::PacketLayout::UINT8)
        {
            throw PBException("TrivialRepacker only supports INT16 or INT8 encoding");
        }
        if (dims.lanesPerBatch != packet.Layout().NumBlocks())
            throw PBException("TrivialRepacker expected " + std::to_string(dims.lanesPerBatch) +
                              " blocks but received " + std::to_string(packet.Layout().NumBlocks()));
        if (dims.framesPerBatch != packet.Layout().NumFrames())
            throw PBException("TrivialRepacker expected " + std::to_string(dims.framesPerBatch) +
                              " frames but received " + std::to_string(packet.Layout().NumFrames()));
        if (dims.laneWidth != packet.Layout().BlockWidth())
            throw PBException("TrivialRepacker expected " + std::to_string(dims.laneWidth) +
                              " blockWidth but received " + std::to_string(packet.Layout().BlockWidth()));

        Mongo::Data::BatchMetadata meta(
            packet.PacketID(),
            packet.StartFrame(),
            packet.StartFrame() + packet.NumFrames(),
            packet.StartZmw());
        meta.SetTimeStamp(packet.TimeStamp());
        switch (packet.Layout().Encoding())
        {
            case DataSource::PacketLayout::INT16:
            {
                PushOut(Mongo::Data::TraceBatch<int16_t>(std::move(packet),
                                                         meta,
                                                         dims,
                                                         PacBio::Cuda::Memory::SyncDirection::HostWriteDeviceRead,
                                                         SOURCE_MARKER()));
                break;
            }
            case DataSource::PacketLayout::UINT8:
            {
                PushOut(Mongo::Data::TraceBatch<uint8_t>(std::move(packet),
                                                         meta,
                                                         dims,
                                                         PacBio::Cuda::Memory::SyncDirection::HostWriteDeviceRead,
                                                         SOURCE_MARKER()));
                break;
            }
            default:
                throw PBException("Unsupported packet encoding in TrivialRepacker");
        }
    }
private:
    std::map<uint32_t, Mongo::Data::BatchDimensions> expectedDims_;
};

}}

#endif //PACBIO_APPLICATION_TRIVIAL_REPACKER_H
