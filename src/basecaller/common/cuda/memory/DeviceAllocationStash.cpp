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

#include <common/cuda/memory/DeviceAllocationStash.h>

#include <numeric>

namespace PacBio {
namespace Cuda {
namespace Memory {

// Implementation notes:
//    The registered allocations are currently stored as a
//    map of maps of vectors of allocations.  The outtermost key
//    is an allocation size, the inner key is a pool ID, and
//    the vector contains a reference to all allocations of that
//    size belonging to that pool.
//
//    The allocations are divided into two partitions, one for data
//    that always lives on the GPU, and another for data shuttled up
//    and down just in time for a given lane analysis.  We could
//    be arbitrarily smart about which allocations end up in which
//    parition, but this function does the dumb and simple thing for
//    now.  The map of vectors for a given allocation size is
//    *always* kept together.  i.e. all of those allocations (across
//    all pools) are either all permanently on the GPU, or all shuttled
//    on demand.  This keeps the logic simple, and all of the code
//    below assumes we don't have to do things like merge different
//    maps together.
//
//    If this strategy ever proves insufficient then it can be changed.
//    I've intentionally encapsulated all this code here so any alterations
//    can be surgical.  Just keep in mind that the function below either
//    needs to be re-written entirely, or carefully gone through to make
//    sure nothing remains that relies on simplifying assumptions.
void DeviceAllocationStash::PartitionData(size_t maxResidentMiB)
{
    const size_t maxBytesResident = maxResidentMiB*(1<<20);
    // TODO this locking dance can be a single line in C++17...
    // std::scoped_lock sl(uploadMutex_, downloadMutex_, mutateMutex_);
    std::lock(uploadMutex_, downloadMutex_, mutateMutex_);
    std::lock_guard<std::mutex> lm1(uploadMutex_, std::adopt_lock);
    std::lock_guard<std::mutex> lm2(downloadMutex_, std::adopt_lock);
    std::lock_guard<std::mutex> lm3(mutateMutex_, std::adopt_lock);

    // Could probably do this more efficiently, but for simplicity toss everything
    // into the mobileData as a starting point, in case someone called this
    // function a second time with a new value.
    for (auto& kvSizeData : stationaryData_)
    {
        auto& destPoolMap = mobileData_[kvSizeData.first];
        for (auto& kvPoolVec : kvSizeData.second)
        {
            auto& dest = destPoolMap[kvPoolVec.first];
            dest.insert(dest.end(),
                        std::make_move_iterator(kvPoolVec.second.begin()),
                        std::make_move_iterator(kvPoolVec.second.end()));
        }
    }
    stationaryData_.clear();

    size_t bytesResident = 0;
    // Could probably be smarter, but for now do an all-or-nothing of transfer
    // of every allocation of a given size, as long as they all will fit.
    auto transferIfFits = [&](auto& kvSizeMap)
    {
        const size_t bytesInLot = std::accumulate(kvSizeMap.second.begin(),
                                                  kvSizeMap.second.end(),
                                                  0ull,
                                                  [&](size_t currSum, auto&& kv){
                                                      return currSum + kv.second.size()*kvSizeMap.first;
                                                  });

        if (bytesResident + bytesInLot <= maxBytesResident)
        {
            auto success = stationaryData_.emplace(std::move(kvSizeMap));
            if (!success.second)
            {
                throw PBException("Unexpected map insert failure");
            } else
            {
                bytesResident += bytesInLot;
            }
        }
    };

    auto eraseEmpty = [](auto& map) {
        for (auto it = map.begin(); it != map.end();)
        {
            if (it->second.empty())
                it = map.erase(it);
            else
                it++;
        }
    };

    // First find irregular allocations not in all pools.  These won't
    // be candidates for any sharing in the allocation cache infrastructure,
    // so are less useful to continually upload/download
    for (auto& kvSizeMap : mobileData_)
    {
        if (bytesResident >= maxBytesResident) break;
        if (kvSizeMap.second.size() == poolsSeen_.size()) continue;

        transferIfFits(kvSizeMap);
    }
    eraseEmpty(mobileData_);

    // Now march through the rest of the allocations.  For simplicitly just doing
    // an all-or-nothing move of the entire lot, this could potentially be
    // made much smarter
    for (auto& kvSizeMap : mobileData_)
    {
        if (bytesResident >= maxBytesResident) break;
        assert(kvSizeMap.second.size() == poolsSeen_.size());

        transferIfFits(kvSizeMap);

    }
    eraseEmpty(mobileData_);
}

void DeviceAllocationStash::Register(uint32_t poolId, Allocation alloc)
{
    // TODO this locking dance can be a single line in C++17...
    // std::scoped_lock sl(uploadMutex_, downloadMutex_, mutateMutex_);
    std::lock(uploadMutex_, downloadMutex_, mutateMutex_);
    std::lock_guard<std::mutex> lm1(uploadMutex_, std::adopt_lock);
    std::lock_guard<std::mutex> lm2(downloadMutex_, std::adopt_lock);
    std::lock_guard<std::mutex> lm3(mutateMutex_, std::adopt_lock);

    poolsSeen_.insert(poolId);
    auto& poolMap = mobileData_[alloc.lock()->size()];
    auto& allocList = poolMap[poolId];
    allocList.emplace_back(alloc);
}

void DeviceAllocationStash::RetrievePool(uint32_t poolId)
{
    std::lock_guard<std::mutex> lm(uploadMutex_);

    // Most of the time this loop should essentially be
    // zero work, but we still need to ensure the data
    // ends up on the GPU since someone else might have
    // manually downloaded it for whatever reason...
    for (const auto& poolMap : stationaryData_)
    {
        auto itr = poolMap.second.find(poolId);
        if (itr == poolMap.second.cend()) continue;

        for (const auto& val : itr->second)
        {
            auto alloc = val.lock();
            alloc->Retrieve();
        }
    }

    for (const auto& poolMap : mobileData_)
    {
        auto itr = poolMap.second.find(poolId);
        if (itr == poolMap.second.cend()) continue;

        for (const auto& val : itr->second)
        {
            auto alloc = val.lock();
            alloc->Retrieve();
        }
    }
}

void DeviceAllocationStash::StashPool(uint32_t poolId)
{
    std::lock_guard<std::mutex> lm(downloadMutex_);

    for (const auto& poolMap : mobileData_)
    {
        auto itr = poolMap.second.find(poolId);
        if (itr == poolMap.second.cend()) continue;

        for (const auto& val : itr->second)
        {
            auto alloc = val.lock();
            alloc->Stash();
        }
    }
}

}}}
