// Copyright (c) 2021, Pacific Biosciences of California, Inc.
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

#ifndef PACBIO_APPLICATION_BATCH_DATA_SOURCE_H
#define PACBIO_APPLICATION_BATCH_DATA_SOURCE_H

#include <cstdint>
#include <pacbio/datasource/DataSourceBase.h>
#include <vector>

#include <dataTypes/TraceBatch.h>

namespace PacBio::Mongo {

/// Helper extension to the DataSourceBase API, primarily
/// for test purposes.  It adds the ability to automatically
/// generate batches rather than packets, and tweaks the API
/// to remove the need for a dedicated thread
///
/// Note: This API is only suitable for use in implementations
///       that do not have realtime constraints on how frequently
///       the ContinueProcessing function is called. Data is
///       only generated lazily on demand
/// Note: This class defines some pseudo-iterators that are only
///       suitable for enabling range based for loops.  In particular,
///       we don't know how many chunks will be produced, so the "end"
///       iterator is just the "empty" or default constructed iterator,
///       and the "not equal" operator only checks if both arguemnts
///       are "empty" or not.  This works fine for a range based for
///       loop cycling over all the data, but would be horribly broken
///       for any more general application.  The types involved are all
///       private intentionally to make them more difficult for external
///       code to name them.
///
/// A basic example:
///   const uint64_t framesPerChunk = 512;
///   const uint64_t lanesPerPool = 64;
///   auto layout = PacketLayout(PacketLayout::BlockLayoutDense,
///                              PacketLayout::INT16_t,
///                              {lanesPerPool, framesPerchunk, Mongo::laneSize});
///   auto cfg = DataSourceBase::Configuration(layout,
///                                            std::make_unique<MallocAllocator>());
///
///   // Produce 4 chunks
///   const uint32_t movieFrames = framesPerChunk * 4;
///   // Each chunk will just have two batches
///   const uint32_t numZmw = layout.NumZmw() * 2;
///   auto source = TraceFileDataSource(std::move(cfg),
///                                     "MyFavoriteData.trc.h5",
///                                     movieFrames, numZmw,
///                                     true /* cache the data */)
///
///   if (processByChunk)
///   {
///       for (const std::vector<TraceBatch<int16_t>>& chunk : source.AllChunks())
///       {
///           ProcessFullChunk(chunk);
///       }
///   } else {
///       for (const TraceBatch<int16_t>& batch : source.AllBatches)
///       {
///           // batch.Meta() contains full information about
///           // what frames/zmw belong to this batch.
///           ProcessSingleBatch(batch);
///       }
///   }
class BatchDataSource : public DataSource::DataSourceBase
{
    /// Private function, for generating the next chunk.
    /// It calls `ContinueProcessing` in a loop until the next
    /// SensorPacketsChunk is produced, after which it
    /// converts it into an std::vector of TraceBatch objects
    ///
    /// \return vector of trace batches to span the chunk.  Once
    ///         the child DataSource is finished, this function
    ///         will return an empty vector;
    std::vector<Data::TraceBatchVariant> NextChunk()
    {
        while (IsRunning() && !ChunksReady())
        {
            ContinueProcessing();
        }
        if (!ChunksReady()) return {};

        DataSource::SensorPacketsChunk chunk;
        PopChunk(chunk, std::chrono::milliseconds{10});
        std::vector<Data::TraceBatchVariant> ret;
        for (auto& packet : chunk)
        {
            auto meta = Data::BatchMetadata(
                packet.PacketID(),
                packet.StartFrame(),
                packet.StartFrame() + packet.NumFrames(),
                packet.StartZmw());

            Data::BatchDimensions dims;
            dims.framesPerBatch = packet.Layout().NumFrames();
            dims.lanesPerBatch = packet.Layout().NumBlocks();
            dims.laneWidth = packet.Layout().BlockWidth();
            assert(dims.laneWidth == laneSize);

            switch (packet.Layout().Encoding())
            {
                case PacBio::DataSource::PacketLayout::INT16:
                {
                    Data::TraceBatch<int16_t> trc(std::move(packet),
                                                  meta,
                                                  dims,
                                                  Cuda::Memory::SyncDirection::HostWriteDeviceRead,
                                                  SOURCE_MARKER());
                    ret.emplace_back(std::move(trc));
                    break;
                }
                case PacBio::DataSource::PacketLayout::UINT8:
                {
                    Data::TraceBatch<uint8_t> trc(std::move(packet),
                                                  meta,
                                                  dims,
                                                  Cuda::Memory::SyncDirection::HostWriteDeviceRead,
                                                  SOURCE_MARKER());
                    ret.emplace_back(std::move(trc));
                    break;
                }
                default:
                    throw PBException("Unexpected trace encoding");
            }
        }

        return ret;
    }

    /// Pseudo-Iterator type for iterating over all the chunks produced by
    /// a DataSource.  This type is *only* suitable for implicit use
    /// during range based for loops.  See class comments for more details
    template <typename T>
    class ChunkIterator
    {
    public:
        ChunkIterator(BatchDataSource* source)
            : source_(source)
        {}

        const std::vector<T>& operator*() const
        {
            return data;
        }
        std::vector<T>& operator*()
        {
            return data;
        }
        ChunkIterator& operator++()
        {
            if constexpr (std::is_same_v<T, Data::TraceBatchVariant>)
            {
                data = source_->NextChunk();
            } else {
                auto tmp = source_->NextChunk();
                data.clear();
                for (auto& v : tmp)
                    data.push_back(std::move(std::get<T>(v.Data())));
            }
            return *this;
        }
        bool operator!=(ChunkIterator& o)
        {
            return data.size() != o.data.size();
        }
    private:
        BatchDataSource* source_;
        std::vector<T> data;
    };

    /// Helper class, whos only purpose is to be fed
    /// into a range based for loop.  Explicit capture
    /// and/or other more advanced usage of this class
    /// is strongly discouraged.
    template <typename T>
    class ChunkContainer
    {
        static_assert(std::is_same_v<T, int16_t>
                      || std::is_same_v<T, uint8_t>
                      || std::is_same_v<T, void>);
        using Batch_t = std::conditional_t<std::is_same_v<T, void>,
                                           Data::TraceBatchVariant,
                                           Data::TraceBatch<T>>;
    public:
        ChunkContainer(BatchDataSource* ptr)
            : source(ptr)
        {}

        ChunkIterator<Batch_t> begin() const
        {
            // Note the weirdness! `end` just creates an
            // "empty" iterator, but incrementing that
            // (the first time) will kick things off and
            // load the first bit of data
            auto ret = end();
            ++ret;
            return ret;
        }
        ChunkIterator<Batch_t> end() const
        {
            return ChunkIterator<Batch_t>(source);
        }
    private:
        BatchDataSource* source;
    };

    /// Pseudo-Iterator type for iterating over all the chunks produced by
    /// a DataSource.  This type is *only* suitable for implicit use
    /// during range based for loops.  See class comments for more details
    template <typename T>
    struct BatchIterator
    {
        BatchIterator(BatchDataSource* source)
            : source_(source)
        {}

        const T& operator*() const
        {
            if constexpr(std::is_same_v<T, Data::TraceBatchVariant>)
                return data[idx];
            else
                return std::get<T>(data[idx].Data());
            __builtin_unreachable();
        }
        T& operator*()
        {
            if constexpr(std::is_same_v<T, Data::TraceBatchVariant>)
                return data[idx];
            else
                return std::get<T>(data[idx].Data());
            __builtin_unreachable();
        }
        BatchIterator& operator++()
        {
            idx++;
            if (idx >= data.size())
            {
                data = source_->NextChunk();
                idx = 0;
            }
            return *this;
        }
        bool operator!=(BatchIterator& o)
        {
            return data.size() != o.data.size();
        }
    private:
        BatchDataSource* source_;
        std::vector<Data::TraceBatchVariant> data;
        size_t idx = 0;
    };

    /// Helper class, whos only purpose is to be fed
    /// into a range based for loop.  Explicit capture
    /// and/or other more advanced usage of this class
    /// is strongly discouraged.
    template <typename T>
    class BatchContainer
    {
        static_assert(std::is_same_v<T, int16_t>
                      || std::is_same_v<T, uint8_t>
                      || std::is_same_v<T, void>);
        using Batch_t = std::conditional_t<std::is_same_v<T, void>,
                                           Data::TraceBatchVariant,
                                           Data::TraceBatch<T>>;
    public:
        BatchContainer(BatchDataSource* ptr)
            : source(ptr)
        {}

        BatchDataSource* source;

        BatchIterator<Batch_t> begin() const
        {
            // Note the weirdness! `end` just creates an
            // "empty" iterator, but incrementing that
            // (the first time) will kick things off and
            // load the first bit of data
            auto ret = end();
            ++ret;
            return ret;
        }
        BatchIterator<Batch_t> end() const
        {
            return BatchIterator<Batch_t>(source);
        }
    };

public:

    using DataSourceBase::DataSourceBase;

    // The following two functions are provided to enable looping
    // through all the data, either by batch or by chunk.  Do
    // beware this is a convenience API provided to make certain
    // tests easier to write, and will not behave well if you stray
    // outside of the expected usage patterns.
    //
    // * Only make a single call to one of these functions.  Anything
    //   else will cause an exception to be thrown.
    // * Do not mix calls to these functions with the usage of a
    //   DataSourceRunner.  Doing so will cause an exception to be
    //   thrown.
    //
    // Beyond that, these functions will default to providing
    // TraceBatchVariants.  You can however specify int16_t or
    // uint8_t as the template parameter and get a specific type
    // of batch directly.  However an exception will be thrown
    // if the requested type does not match the data generated
    template <typename T = void>
    ChunkContainer<T> AllChunks()
    {
        if (this->IsStarted() || startedSelf_)
            throw PBException("Unexpected usage\n");

        Start();
        startedSelf_ = true;

        return ChunkContainer<T>(this);
    }

    template <typename T = void>
    BatchContainer<T> AllBatches()
    {
        if (this->IsStarted() || startedSelf_)
            throw PBException("Unexpected usage\n");

        Start();
        startedSelf_ = true;

        return BatchContainer<T>(this);
    }

private:
    /// This class needed to implement vStart for some correctness guarantees.
    /// Any children that need to do any work during the top level `Start` call
    /// shall override this function instead.
    virtual void vStartImpl() {}

    void vStart() override final
    {
        // If we're already started, then we're currently being invoked by the
        // DataSourceRunner, but we've already previously called AllBatches or
        // AllChunks
        if(startedSelf_) throw PBException("Detected invalid use of DataSourceRunner");

        vStartImpl();
    }
    bool startedSelf_ = false;
};

}  // namespace PacBio::Mongo

#endif  // PACBIO_APPLICATION_BATCH_DATA_SOURCE_H
