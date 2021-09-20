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

#ifndef PACBIO_APPLICATION_BLOCK_REPACKER_H
#define PACBIO_APPLICATION_BLOCK_REPACKER_H

#include <pacbio/datasource/PacketLayout.h>
#include <pacbio/datasource/SensorPacket.h>

#include <appModules/Repacker.h>

#include <common/graphs/GraphNodeBody.h>

#include <dataTypes/TraceBatch.h>

namespace PacBio {
namespace Application {

// A Moderately general block repacker, intended to handle a number of situations:
// - inbound SensorPackets: are BLOCK_LAYOUT_DENSE.
// - inbound SensorPackets: have a block width that is a multiple of a cache line,
//                          which has pixels that are either int16_t or uint8_t
// - inbound SensorPackets: have a uniform frame count
// - inbound SensorPackets: have a frame count that is equal to, or an even divisor of,
//                          the output TraceBatch frame count
//
// The input SensorPackets do *not* have to:
// - Have multiple blocks
// - Have the same number of blocks with respect to each other.
//
// Notes:
// - The implementation is parallel up to `concurrencyLimit` (specified in the ctor).
// - There is a general purpose implementation that is the most flexible, but perhaps
//   not the most performant.  It should serve well for experiments, but one could
//   possibly do better for more limited scenarios.  The implementation file
//   provides a mechanism for re-writing the low-level implementation for specific
//   target cases, without having to re-write the higher level multi-threading logic
// - The implementation will not convert between 16 and 8 bit data types.  The output
//   type will be the same as the input, we're merely shuffling data around.
class BlockRepacker final : public RepackerBody
{
public:
    /// \param expectedInputLayouts The expected input packet layouts.  It is required
    ///                             that all blocks have the same number of frames.
    /// \param outputLayout         The desired output layout for this repacker.
    ///                             All output batches will be of the same shape, save
    ///                             perhaps the last one which might have fewer blocks
    /// \param numZmw               The number of ZMW in a chip
    /// \param concurrencyLimit     The maximum number of threads allowed to work at
    ///                             once in this node of the analysis graph.
    BlockRepacker(const std::map<uint32_t, DataSource::PacketLayout>& expectedInputLayouts,
                  Mongo::Data::BatchDimensions outputDims,
                  size_t numZmw,
                  size_t concurrencyLimit);

    size_t ConcurrencyLimit() const override { return concurrencyLimit_; }
    float MaxDutyCycle() const override { return 1.0; }

    void Process(DataSource::SensorPacket packet) override;

    std::map<uint32_t, Mongo::Data::BatchDimensions> BatchLayouts() const override;

    ~BlockRepacker() override;

    // The actual implementation will be a set of
    // template classes.  Providing another virtual
    // slip layer here so that the constructor can
    // chose while implementation is most appropriate
    // to instantiate
    class Impl
    {
    public:
        virtual void Process(DataSource::SensorPacket packet) = 0;
        virtual std::map<uint32_t, Mongo::Data::BatchDimensions> BatchLayouts() const = 0;
        virtual ~Impl() = default;
    };

private:

    std::unique_ptr<Impl> impl_;
    size_t concurrencyLimit_;
};


}}

#endif //PACBIO_APPLICATION_BLOCK_REPACKER_H
