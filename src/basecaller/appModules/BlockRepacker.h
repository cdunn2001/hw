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

#ifndef PACBIO_APPLICATION_BLOCK_REPACKER_H
#define PACBIO_APPLICATION_BLOCK_REPACKER_H

#include <pacbio/datasource/PacketLayout.h>
#include <pacbio/datasource/SensorPacket.h>

#include <common/graphs/GraphNodeBody.h>

#include <dataTypes/TraceBatch.h>

namespace PacBio {
namespace Application {

// A Moderately general block repacker, intended to handle a number of situations:
// - inbound SensorPackets: are BLOCK_LAYOUT_DENSE.
// - inbound SensorPackets: have a block width that is a multiple of 32 16 bit pixels
// - inbound SensorPackets: have a uniform frame count
// - inbound SensorPackets: have a frame count that is equal to, or an even divisor of,
//                          the output TraceBatch frame count
// - inbound SensorPackets: have a uniform block size.  That is they can vary the number
//                          of blocks, but not the shape of blocks.
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
class BlockRepacker final : public Graphs::MultiTransformBody<DataSource::SensorPacket,
                                                              const Mongo::Data::TraceBatch<int16_t>>
{
public:
    /// \param expectedInputLayout a nominal layout expected for input packets.  In
    ///                            practice the number of blocks can change, but
    ///                            the incoming packets must match the expected block
    ///                            dimensions
    /// \param outputLayout        The output layout that this repacker will produce.
    ///                            All output batches will be of the same shape, save
    ///                            perhaps the last one which might have fewer blocks
    /// \param numZmw              The number of ZMW in a chip
    /// \param concurrencyLimit    The maximum number of threads allowed to work at
    ///                            once in this node of the analysis graph.
    BlockRepacker(DataSource::PacketLayout expectedInputLayout,
                  Mongo::Data::BatchDimensions outputDims,
                  size_t numZmw,
                  size_t concurrencyLimit);

    size_t ConcurrencyLimit() const override { return concurrencyLimit_; }
    float MaxDutyCycle() const override { return 1.0; }

    void Process(DataSource::SensorPacket packet) override;

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
    };

private:

    std::unique_ptr<Impl> impl_;
    size_t concurrencyLimit_;
};


}}

#endif //PACBIO_APPLICATION_BLOCK_REPACKER_H
