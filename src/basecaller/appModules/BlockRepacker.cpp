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

#include <appModules/BlockRepacker.h>

#include <immintrin.h>

using namespace PacBio::Application;
using namespace PacBio::DataSource;
using namespace PacBio::Mongo::Data;
using namespace PacBio::Cuda::Memory;

namespace {

// Helper factory function to create a TraceBatch for us
TraceBatch<int16_t> CreateBatch(BatchDimensions dims, size_t startFrame, size_t startZmw, size_t maxZmw)
{
    const uint32_t poolId = startZmw / dims.ZmwsPerBatch();
    BatchMetadata meta(poolId, startFrame, startFrame + dims.framesPerBatch, startZmw);

    // All batches, except maybe the last one, will have the
    // full number of blocks specified by the specified dims
    dims.lanesPerBatch = std::min(static_cast<size_t>(dims.lanesPerBatch),
                                 (maxZmw - startZmw) / dims.laneWidth);
    return TraceBatch<int16_t>(meta, dims, SyncDirection::HostWriteDeviceRead, SOURCE_MARKER());
}

// Implements a mutex that can be used in movable objects.
// Theoretically this opens us up to race conditions
// during move operations themselves, so it is not suitable
// for general purpose use (this is why std::mutex is not
// movable itself).  That said, in this file this class
// enables us to put members with mutexes inside standard
// containers like vectors and queues.  The moves only
// happen while those containers are being first populated
// (in a serial fashion), and all move operations are completed
// by the time multiple threads are in play.
class MovableMutex
{
public:
    MovableMutex()
        : m_(std::make_unique<std::mutex>())
    {}

    MovableMutex(const MovableMutex&) = delete;
    MovableMutex(MovableMutex&& o)
        : m_(std::move(o.m_))
    {
        o.m_ = std::make_unique<std::mutex>();
    }

    MovableMutex& operator=(const MovableMutex&) = delete;
    MovableMutex& operator=(MovableMutex&& o)
    {
        if (&o != this)
        {
            m_ = std::move(o.m_);
            o.m_ = std::make_unique<std::mutex>();
        }
        return *this;
    }

    operator std::mutex&() { return *m_; }

private:
    std::unique_ptr<std::mutex> m_;
};

// Helper RAII class to make sure SensorPackets return
// their allocation to the allocation pools.
//
// TODO: This could be avoided with a bit of a refactor.
// Right now SmartAllocation has an internal RAII object
// to return the allocation to a custom deleter function.
// If that was tweaked so the "custom deleter" accepted
// a bit of metadata (like allocation location), then
// we could automatically hand things back to the
// allocation caching framework, instead of having to
// do things manually.
class CachedPacket
{
public:
    CachedPacket(PacBio::DataSource::SensorPacket packet)
        : packet_(std::move(packet))
    {}

    CachedPacket(const CachedPacket&) = delete;
    CachedPacket(CachedPacket&&) = default;
    CachedPacket& operator=(const CachedPacket&) = delete;
    CachedPacket& operator=(CachedPacket&&) = default;

    ~CachedPacket()
    {
        IMongoCachedAllocator::ReturnHostAllocation(std::move(packet_).RelinquishAllocation());
    }

    operator PacBio::DataSource::SensorPacket&() { return packet_; }
    operator const PacBio::DataSource::SensorPacket&() const { return packet_; }

private:
    PacBio::DataSource::SensorPacket packet_;
};


// Thread safe priority queue for ordering Packets.  Packets
// are ordered by ascending start zmw, and within the same
// start zmw by ascending start frame.
class PacketPriorityQueue
{
public:
    PacketPriorityQueue() = default;

    PacketPriorityQueue(const PacketPriorityQueue&) = delete;
    PacketPriorityQueue(PacketPriorityQueue&&) = default;
    PacketPriorityQueue& operator=(const PacketPriorityQueue&) = delete;
    PacketPriorityQueue& operator=(PacketPriorityQueue&&) = default;

    void Push(std::shared_ptr<CachedPacket> packet)
    {
        assert(packet);
        std::lock_guard<std::mutex> lm(m_);
        data_.emplace(packet);
    }

    // Returns and pops the top of the priority
    // queue if the predicate `f` is true for that
    // value.  If the predicate returns false,
    // (or the queue is empty), and empty shared
    // pointer is returned.
    //
    // `f` should have the callable signature
    // bool operator()(const CachedPacket& p)
    template <typename Func>
    std::shared_ptr<CachedPacket> PopIf(Func&& f)
    {
        std::shared_ptr<CachedPacket> ret;
        std::lock_guard<std::mutex> lm(m_);
        if (!data_.empty())
        {
            if (f(const_cast<const CachedPacket&>(*data_.top())))
            {
                ret = data_.top();
                data_.pop();
            }
        }
        return ret;
    }

    // Returns and pops the top of the queue,
    // unless the queue is empty in which case
    // an empty shared_ptr is returned.
    std::shared_ptr<CachedPacket> Pop()
    {
        std::shared_ptr<CachedPacket> ret;
        std::lock_guard<std::mutex> lm(m_);
        if (!data_.empty())
        {
            ret = data_.top();
            data_.pop();
        }
        return ret;
    }

    bool Empty() const
    {
        std::lock_guard<std::mutex> lm(m_);
        return data_.empty();
    }
private:
    // If operator() returns false, then `left` will be ordered before `right`
    struct Compare
    {
        bool operator()(const std::shared_ptr<CachedPacket>& left,
                        const std::shared_ptr<CachedPacket>& right) const
        {
            const SensorPacket& l = *left;
            const SensorPacket& r = *right;
            if (l.StartZmw() == r.StartZmw())
            {
                return l.StartFrame() > r.StartFrame();
            } else
            {
                return l.StartZmw() > r.StartZmw();
            }
        }
    };

    using Queue = std::priority_queue<std::shared_ptr<CachedPacket>,
                                      std::deque<std::shared_ptr<CachedPacket>>,
                                      Compare>;
    Queue data_;
    mutable MovableMutex m_;
};

// Enum for the different repacker specializations.  DEFAULT
// is the most general, and new entries should be added
// for any special cases you want to handle separately.
enum class RepackerMode
{
    DEFAULT
};

// Low level implementation class, handling the actual repacking of data from a packet
// into a batch.  Right here is the general implementation, that puts few contraints
// on the layout of incoming packets.  It is entirely possible one could write a faster
// implementation for a less general case.  If that is the desire then simply
// implement a specialization for this class, and have the top-level BlockRepacker
// constructor instantiate things with that values of RepackerMode when the necessary
// input conditions are true.  This way all specialized implementations have to do
// is re-write the low-level data shuffle itself, while still leveraging all the
// existing higher level bookkeeping and threading model
//
// This class is not thread safe.  It is the responsibility of calling classes to
// make sure only one thread uses any given BatchFiller at one time.
template <RepackerMode mode>
class BatchFiller
{
public:
    BatchFiller(const BatchDimensions& dims, size_t startFrame,
                size_t startZmw, size_t maxZmw)
        : batch_(CreateBatch(dims, startFrame, startZmw, maxZmw))
        , currZmw_(startZmw)
        , currFrame_(startFrame)
    {}

    // These are the current zmw/frame we are waiting
    // for a packet to provide
    size_t CurrZmw() const { return currZmw_; }
    size_t CurrFrame() const { return currFrame_; }

    // Determines if a given packet is the next desired packet.
    // This implementation processes packets in order, first
    // doing all frames for a given start zmw, and then over
    // zmw until the batch is filled
    bool ReadyFor(const SensorPacket& packet) const
    {
        if (packet.StartFrame() != currFrame_) return false;
        if (packet.StartZmw() <= currZmw_ && packet.StopZmw() > currZmw_) return true;

        return false;
    }

    // Predicate to check if the batch is full and ready to be
    // placed in the output queue
    bool Done() const
    {
        if (currFrame_ != batch_.GetMeta().LastFrame()) return false;
        if (currZmw_ != batch_.GetMeta().FirstZmw() + batch_.StorageDims().ZmwsPerBatch()) return false;

        return true;
    }

    const TraceBatch<int16_t>& Batch() const
    {
        return batch_;
    }

    // Only callable on rvalue references.  It will move
    // out the batch.  Leaves this object in a moved-from
    // state that isn't good for anything except destruction
    TraceBatch<int16_t> ExtractBatch() &&
    {
        assert(Done());
        return std::move(batch_);
    }

    // Does the repacking of any data present in `packet`
    // that belongs to this batch
    void Process(const SensorPacket& packet)
    {
        // We only handle 16 bit data, so there
        // are 32 zmw in a cache line.
        constexpr size_t cacheLineZmw = 32;

        assert(packet.StartZmw() <= currZmw_);
        // Find the zmw range that is the union of this batch and the packet
        const size_t startZmw = std::max(packet.StartZmw(), currZmw_);
        const size_t stopZmw = std::min(packet.StopZmw(),
                                        static_cast<size_t>(batch_.GetMeta().FirstZmw()
                                                          + batch_.StorageDims().ZmwsPerBatch()));

        static constexpr size_t batchLaneWidth = 64;
        assert(batch_.LaneWidth() == batchLaneWidth);

        const size_t batchFirstFrame = batch_.GetMeta().FirstFrame();
        const size_t batchFirstZmw = batch_.GetMeta().FirstZmw();
        const size_t batchLastZmw = batchFirstZmw + batch_.StorageDims().ZmwsPerBatch();

        const size_t packetBlockWidth = packet.Layout().BlockWidth();
        const size_t packetNumFrames = packet.NumFrames();
        const size_t packetStartZmw = packet.StartZmw();

        const size_t packetStartBlock = (startZmw - packet.StartZmw()) / packetBlockWidth;
        const size_t packetEndBlock = [&]()
        {
            auto ret = (stopZmw - packet.StartZmw()) / packetBlockWidth;
            if (ret * packet.Layout().BlockWidth() + packetStartZmw != stopZmw) ret++;
            return ret;
        }();

        // Loop in a order that streams the input, to help out the prefetcher
        for (size_t packetBlockIdx = packetStartBlock; packetBlockIdx < packetEndBlock; ++packetBlockIdx)
        {
            const auto  packetBlockZmw = packetBlockIdx * packetBlockWidth + packetStartZmw;
            const auto* packetBlockData = reinterpret_cast<const int16_t*>(packet.BlockData(packetBlockIdx).Data());
            for (size_t packetFrameIdx = 0; packetFrameIdx < packetNumFrames; ++packetFrameIdx, packetBlockData += packetBlockWidth)
            {
                size_t batchFrameIdx = packetFrameIdx + (currFrame_ - batchFirstFrame);
                for (size_t packetZmwIdx = 0; packetZmwIdx < packetBlockWidth; packetZmwIdx += cacheLineZmw)
                {
                    // We've made sure to select only packet blocks that overlap the batch,
                    // but it might be a partial overlap.  Need to neglect all ZMW's that
                    // don't belong
                    const size_t zmwNum = packetBlockZmw + packetZmwIdx;
                    if (zmwNum < batchFirstZmw) continue;
                    if (zmwNum >= batchLastZmw) break;

                    size_t batchBlockIdx = (zmwNum - batchFirstZmw) / batchLaneWidth;
                    size_t batchBlockZmw = batchFirstZmw + batchBlockIdx * batchLaneWidth;
                    size_t batchBlockZmwIdx = zmwNum - batchBlockZmw;
                    auto * batchBlockData = reinterpret_cast<__m128i*>(
                          batch_.GetBlockView(batchBlockIdx).Data()
                          + batchFrameIdx * batchLaneWidth + batchBlockZmwIdx);

                    //memcpy(bData, pData + pZmwIdx, 2*cacheLineZmw);
                    const auto* ppData = reinterpret_cast<const __m128i*>(packetBlockData + packetZmwIdx);
                    // Not sure if these prefetches are doing anything useful.  Initial checks
                    // indicate they might be, but the data was noisy and I'm not sure.
                    // At the least it's not hurting.
                    _mm_prefetch(reinterpret_cast<const char*>(ppData + 16), _MM_HINT_T0);
                    _mm_prefetch(reinterpret_cast<const char*>(ppData + 32), _MM_HINT_T0);

                    // Copy 64 bytes from the source to the destinatoin.
                    // Potentially could do better with avx512 and/or aligned storage.
                    auto d1 = _mm_loadu_si128(ppData);
                    auto d2 = _mm_loadu_si128(ppData+1);
                    auto d3 = _mm_loadu_si128(ppData+2);
                    auto d4 = _mm_loadu_si128(ppData+3);
                    _mm_storeu_si128(batchBlockData,   d1);
                    _mm_storeu_si128(batchBlockData+1, d2);
                    _mm_storeu_si128(batchBlockData+2, d3);
                    _mm_storeu_si128(batchBlockData+3, d4);
                }
            }
        }

        currFrame_ += packet.NumFrames();
        if (currFrame_ == batch_.GetMeta().LastFrame())
        {
            currZmw_ = stopZmw;
            if (currZmw_ != batch_.GetMeta().FirstZmw() + batch_.StorageDims().ZmwsPerBatch())
            {
                currFrame_ = batch_.GetMeta().FirstFrame();
            }
        }
    }

private:

    TraceBatch<int16_t> batch_;
    size_t currZmw_;
    size_t currFrame_;
};

// Mid-level implementation class, handling an "Arena" of pools.
// An Arena handles a subset of a chunk, and is used to enable
// a course grained level of parallelism, with different threads
// working in different arenas
//
// The primary concerns of this class is with managing the logic
// of ordering incoming packets and creation/delivery of outgoing
// batches.  All public functions are thread safe to call at any
// time, though the `Process` function will return early if another
// thread is already executing that function.
//
// Incoming packets will be ordered and processed according to their
// start zmw/frame.  If packets happen to come in ordered then they
// will be repacked into batches as they come in.  Otherwise they
// will be stored and ordered until the expected "next" packet shows
// up.  In practice entire chunks are ready at once, so even with a
// "hostile" input order, the sorting/queing of work stalls real
// progress only for a trivial amount of time compared to the actual
// data shuffle.  Enforcing a sort ourselves makes the logic for
// ensuring all expected data pieces has arrived that much simpler.
template <RepackerMode mode>
class PoolArena
{
public:
    PoolArena(BatchDimensions batchDims,
              size_t startZmw, size_t endZmw,
              BlockRepacker* repacker)
        : batchDims_(batchDims)
        , startZmw_(startZmw)
        , stopZmw_(endZmw)
        , currStartFrame_(0)
        , currStartZmw_(startZmw)
        , repacker_(repacker)
    {
        assert(stopZmw_ > startZmw_);
    }

    PoolArena(const PoolArena&) = delete;
    PoolArena(PoolArena&&) = default;
    PoolArena& operator=(const PoolArena&) = delete;
    PoolArena& operator=(PoolArena&&) = default;

    // Determines if any of the data in a packet overlaps
    // with this Arena
    bool OverlapsWith(const SensorPacket& packet) const
    {
        return packet.StopZmw()  > startZmw_
            && packet.StartZmw() < stopZmw_;
    }

    // Queues up a packet to be processed by this arena.
    // This function is thread-safe to call at any time,
    // and even if another thread has the processing lock,
    // that thread can see and handle this new data (assuming
    // this Arena is ready to process it) without that thread
    // having to return and re-enter.
    void AddPacket(std::shared_ptr<CachedPacket> packet)
    {
        assert(OverlapsWith(*packet));
        assert(static_cast<SensorPacket&>(*packet).NumZmw() != 0);
        assert(static_cast<SensorPacket&>(*packet).BlockData(0));

        packetsToWrite_.Push(packet);
    }

    // Processes any ready packets, and returns true if any useful
    // work was done.  There are three reasons we might not do any
    // useful work:
    // 1. This Arena is currently empty, either no data has arrived
    //    or all data has been processed
    // 2. This Arena has some data, but we are still expecting a
    //    packet with an earlier start zmw/frame.  We will do no
    //    work until that expected packet comes in.
    // 3. Another thread already owns the lock and is busy repacking
    //    data.
    //
    // If none of those three conditions cause an early return, then
    // we will process as many packets as we can, up until the point
    // that the packet queue is empty and there is no more work, or
    // we come to a packet that is out-of-order, and we stop until
    // the expected next packet comes in.
    bool ProcessReady()
    {
        std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);

        // There is nothing to do if we're either out of work
        // or another thread owns the lock.
        if (packetsToWrite_.Empty() || !lock.owns_lock())
        {
            return false;
        }

        // We need at least the next output batch ready at all times,
        // so that we can query if the next packet available is
        // the one we want.
        if (currBatches_.empty()) PrepareNext();

        bool didSomething = false;
        std::shared_ptr<CachedPacket> packet;
        auto ReadyPredicate = [this](const CachedPacket& p)
                              {
                                  return currBatches_.front().ReadyFor(p);
                              };
        // Loop and process all packets available as long as they are in the expected order.
        // If an expected packet is missing then we'll just terminate, and resume later
        // when that packet arrives.
        while ((packet = packetsToWrite_.PopIf(ReadyPredicate)))
        {
            didSomething = true;
            ProcessPacket(*packet);

            // Find all fully populated batches, and output them.
            while (currBatches_.front().Done())
            {
                repacker_->PushOut(std::move(currBatches_[0]).ExtractBatch());
                currBatches_.pop_front();
                if (currBatches_.empty()) break;
            }
            bool moreWork = !packetsToWrite_.Empty();
            if (currBatches_.empty() && moreWork)
                PrepareNext();
            if (!moreWork) break;
        }

        return didSomething;
    }

    ~PoolArena()
    {
        // Things are probably FUBAR if someone is currently holding a lock,
        // but try and recover anyway...
        {
            std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
            if (!lock.owns_lock())
            {
                PBLOG_ERROR << "Destroying PoolArena while worker threads are active!  "
                            << "Blocking until we can obtain the lock...";
            }
        }
        std::lock_guard<std::mutex> lm(mutex_);

        // Something has gone wrong.  Either an incomplete chunk came through,
        // or we have a critical internal bug!
        if(!currBatches_.empty() || !packetsToWrite_.Empty())
        {
            PBLOG_ERROR << "BasicBlockRepacker Error";
            PBLOG_ERROR << "entries in Write queue: startFrame, startZme";
            while (!packetsToWrite_.Empty())
            {
                auto pdata = packetsToWrite_.Pop();
                const SensorPacket& packet = *pdata;
                PBLOG_ERROR << packet.StartFrame() << " " << packet.StartZmw();
            }
            PBLOG_ERROR << "Current batches in progress: currFrame, currZmw, firstFrame, firstZmw";
            for (const auto& batchFiller : currBatches_)
            {
                const size_t currFrame = batchFiller.CurrFrame();
                const size_t currZmw = batchFiller.CurrZmw();
                const auto& batch = batchFiller.Batch();
                PBLOG_ERROR << currFrame << " " << currZmw << " " << batch.GetMeta().FirstFrame() << " " << batch.GetMeta().FirstZmw();
            }
            assert(false);
        }
    }
private:
    // Non-threadsafe function, only to be called while holding the lock.
    // Creates the next BatchFiller we expect to use, and updates the
    // data members recording our progress.
    void PrepareNext()
    {
        currBatches_.emplace_back(batchDims_, currStartFrame_, currStartZmw_, stopZmw_);
        currStartZmw_ += batchDims_.ZmwsPerBatch();
        if (currStartZmw_ >= stopZmw_)
        {
            currStartZmw_ = startZmw_;
            currStartFrame_ += batchDims_.framesPerBatch;
        }
    };

    // Non-threadsafe function, only to be called while holding the lock.
    // Handles sharing out a packet with all necessary BatchFillers.
    void ProcessPacket(const SensorPacket& packet)
    {
        assert(!currBatches_.empty());

        // make sure first batch overlaps packet
        assert(currBatches_.front().Batch().GetMeta().FirstZmw() < packet.StopZmw());
        assert(currBatches_.front().Batch().GetMeta().FirstZmw() + batchDims_.ZmwsPerBatch() > packet.StartZmw());
        auto lastZmw = std::min(stopZmw_, packet.StopZmw());
        while (lastZmw > currBatches_.back().Batch().GetMeta().FirstZmw() + batchDims_.ZmwsPerBatch())
        {
            PrepareNext();
        }
        // We also expect no batches that don't overlap with this packet
        assert(lastZmw > currBatches_.back().Batch().GetMeta().FirstZmw());
        assert(lastZmw <= currBatches_.back().Batch().GetMeta().FirstZmw() + batchDims_.ZmwsPerBatch());

        for (auto& filler : currBatches_) filler.Process(packet);
    };

    // Members describing the layout of the overal arena
    BatchDimensions batchDims_;
    size_t startZmw_;
    size_t stopZmw_;

    // start frame for the current chunk being populated
    size_t currStartFrame_;
    // zmw of the next batch to be created and placed into
    // currBatches_
    size_t currStartZmw_;

    // packets waiting to be repacked into batches
    PacketPriorityQueue packetsToWrite_;
    // The set of batches currently being populated
    std::deque<BatchFiller<mode>> currBatches_;

    // Pointer to the top level BlockRepacker.  This inherits
    // from a GraphBodyNode, and has the output queue we need
    // to push data into
    BlockRepacker* repacker_;

    // Mutex used to ensure only one worker thread is populating
    // batches at any given point in time
    MovableMutex mutex_;
};

// Top level implementation for the BlockRepacker.
// This class mostly concerns itself with setting
// up a set of PoolArena objects that span a whole
// chunk.  Any packets that come through are just
// forwarded to the different arenas, where different
// threads can work in different arenas concurrently
template <RepackerMode mode>
class ImplChild : public BlockRepacker::Impl
{
public:
    ImplChild(BatchDimensions batchDims,
              size_t numZmw,
              BlockRepacker* parent,
              size_t numWorkers)
    {
        // Figure out number of batches required to span all the zmw, with care
        // taken for integer division.
        const auto numBatches = (numZmw + batchDims.ZmwsPerBatch() - 1) / batchDims.ZmwsPerBatch();
        // Statically chunk the batches into arenas.  Again take care for
        // integer division, and keep the arenas as close in size as possible
        // (i.e. we'll opt for several arenas 1 smaller than the norm, rather
        // that one single runt arena that can be arbitrarily small)
        const auto minBatchesPerArena = numBatches / numWorkers;
        const auto numSmallBatches = numWorkers - numBatches % numWorkers;

        // Create all the arenas.  Need to take care, both because not all
        // arenas have the same number of batches, but there will potentially
        // be a runt batch at the end, depending on how evenly the chip divides
        // into batches
        size_t startZmw = 0;
        if (minBatchesPerArena > 0)
        {
            for (size_t i = 0; i < numSmallBatches; ++i)
            {
                size_t endZmw = std::min(startZmw + minBatchesPerArena * batchDims.ZmwsPerBatch(), numZmw);
                arenas_.emplace_back(batchDims, startZmw, endZmw, parent);
                startZmw = endZmw;
            }
        }
        for (size_t i = numSmallBatches; i < numWorkers; ++i)
        {
            size_t endZmw = std::min(startZmw + (minBatchesPerArena+1) * batchDims.ZmwsPerBatch(), numZmw);
            arenas_.emplace_back(batchDims, startZmw, endZmw, parent);
            startZmw = endZmw;
        }
        assert(startZmw == numZmw);
    }

    // Processes a new packet.  We do two things:
    // 1. The packet is queued up inside any arenas that
    //    overlap with this packet.  Since a packet may
    //    overlap with pools from separate arenas, the
    //    packet is first moved into shared_ptr so all
    //    interested parties can share ownership
    // 2. We enter a loop work, where we continuously do
    //    work as long as we are making forward progress.
    //    Once we manage to visit all existing arenas
    //    without doing usefull work (e.g. they are all empty)
    //    then we'll exit.
    void Process(SensorPacket packet) override
    {
        // Need to turn things into a CachedPacket while
        // we're at it, to make sure the underlying allocation
        // get's returned to the allocation pools.
        //
        // TODO the allocation pools can be refactored to make
        //      this unecessary.
        auto sharedPacket = std::make_shared<CachedPacket>(std::move(packet));
        for (auto& worker : arenas_)
        {
            if (worker.OverlapsWith(*sharedPacket))
            {
                worker.AddPacket(sharedPacket);
            }
        }

        bool didSomething = false;
        do
        {
            didSomething = false;
            for (size_t i = 0; i < arenas_.size(); ++i)
            {
                // ProcessReady returns true if we made some
                // real forward progress.
                didSomething |= arenas_[i].ProcessReady();
            }
        } while(didSomething);
    }

private:
    std::vector<PoolArena<mode>> arenas_;
};

} // anonymous namespace

namespace PacBio {
namespace Application {

BlockRepacker::BlockRepacker(PacketLayout /*expectedInputLayout*/,
                             BatchDimensions outputDims,
                             size_t numZmw,
                             size_t concurrencyLimit)
    : concurrencyLimit_(concurrencyLimit)
{
    // When/If multiple modes are implemented, the runtime switch
    // will go here.  We'll instantiate a special mode as appropriate
    // and fall back to DEFAULT if there is no applicable special
    // implementation
    impl_ = std::make_unique<ImplChild<RepackerMode::DEFAULT>>(
                outputDims,
                numZmw,
                this,
                concurrencyLimit);
}

void BlockRepacker::Process(SensorPacket packet)
{
    impl_->Process(std::move(packet));
}

BlockRepacker::~BlockRepacker() {}

}}
