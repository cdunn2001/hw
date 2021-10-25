#ifndef mongo_dataTypes_BatchMetadata_H_
#define mongo_dataTypes_BatchMetadata_H_

// Copyright (c) 2019, Pacific Biosciences of California, Inc.
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
//
//  Description:
//  Defines class BatchMetadata.

#include <cassert>
#include <cstdint>

namespace PacBio {
namespace Mongo {
namespace Data {

// Class that identifies the "location" of a batch of data
// within an acquisition
class BatchMetadata
{
public:
    BatchMetadata() = default;

    BatchMetadata(uint32_t poolId, int32_t firstFrame, int32_t lastFrame, uint32_t firstZmw)
        : poolId_(poolId)
        , firstFrame_(firstFrame)
        , lastFrame_(lastFrame)
        , firstZmw_(firstZmw)
    {
        assert(firstFrame <= lastFrame);
    }

    /// An identifier for a specific pool of ZMWs.
    uint32_t PoolId() const { return poolId_; }

    int32_t FirstFrame() const { return firstFrame_; }
    int32_t LastFrame() const { return lastFrame_; }

    uint32_t FirstZmw() const { return firstZmw_; }

private:
    // Not set in stone, just examples
    uint32_t poolId_;      // Identifier of pool of ZMWs.
    int32_t firstFrame_;
    int32_t lastFrame_;
    uint32_t firstZmw_;
};


inline bool operator==(const BatchMetadata& lhs, const BatchMetadata& rhs)
{
    if (lhs.PoolId() != rhs.PoolId()) return false;
    if (lhs.FirstFrame() != rhs.FirstFrame()) return false;
    if (lhs.LastFrame() != rhs.LastFrame()) return false;
    return true;
}

}}}     // namespace PacBio::Mongo::Data

#endif // mongo_dataTypes_BatchMetadata_H_
