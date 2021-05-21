// Copyright (c) 2014-2016, Pacific Biosciences of California, Inc.
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted (subject to the limitations in the
// disclaimer below) provided that the following conditions are met:
//
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
//  * Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
//  * Neither the name of Pacific Biosciences nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE
// GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY PACIFIC
// BIOSCIENCES AND ITS CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL PACIFIC BIOSCIENCES OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
// USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
// OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGE.

// Programmer: Armin Töpfer

#pragma once

#include <map>
#include <mutex>
#include <iostream>
#include <condition_variable>

namespace PacBio
{

template <class K, class V, class Compare = std::less<K>,
          class Allocator = std::allocator<std::pair<const K, V>>>
class ThreadSafeMap
{
private:
    std::map<K, V, Compare, Allocator> map_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;

public:
    void Set(K key, V&& value)
    {
        std::lock_guard<std::mutex> lk(this->mutex_);
        this->map_[key] = std::move(value);
    }

    bool Empty()
    {
        std::lock_guard<std::mutex> lk(this->mutex_);
        return this->map_.empty();
    }

    size_t Size()
    {
        std::lock_guard<std::mutex> lk(this->mutex_);
        return this->map_.size();
    }

    bool HasKey(const K key)
    {
        std::lock_guard<std::mutex> lk(this->mutex_);
        return this->map_.find(key) != this->map_.cend();
    }

    V Get(const K key)
    {
        std::lock_guard<std::mutex> lk(this->mutex_);

        V item = std::move(map_[key]);
        // Remove dangling pointer!
        map_.erase(key);
        return std::move(item);
    }

    void Pop(const K key)
    {
        std::lock_guard<std::mutex> lk(this->mutex_);

        map_.erase(key);
    }

    V& Ref(const K key)
    {
        std::lock_guard<std::mutex> lk(this->mutex_);

        return map_[key];
    }
};

} // ::PacBio

