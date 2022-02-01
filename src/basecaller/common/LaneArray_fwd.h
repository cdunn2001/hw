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

#ifndef mongo_common_LaneArray_fwd_H_
#define mongo_common_LaneArray_fwd_H_

#include <cstddef>

#include <common/MongoConstants.h>

namespace PacBio {
namespace Simd {

// Forward declare the classes in the namespace they will be defined
template<size_t ScalarCount_ = Mongo::laneSize>
class LaneMask;

template <typename T, size_t ScalarCount = Mongo::laneSize>
class LaneArray;

template <typename T, size_t ScalarCount>
class MemoryRange;

template <typename T>
union ArrayUnion;

}}

// Now export the public types to the Mongo namespace
namespace PacBio {
namespace Mongo {

template<size_t ScalarCount = laneSize>
using LaneMask = Simd::LaneMask<ScalarCount>;

template <typename T, size_t ScalarCount = laneSize>
using LaneArray = Simd::LaneArray<T, ScalarCount>;

template <typename T, size_t ScalarCount>
using MemoryRange = Simd::MemoryRange<T, ScalarCount>;

template <typename T>
using ArrayUnion = Simd::ArrayUnion<T>;

}}


#endif // mongo_common_LaneArray_fwd_H_
