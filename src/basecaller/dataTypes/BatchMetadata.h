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
//  Defines classes BatchMetadata and BatchDimensions.

namespace PacBio {
namespace Mongo {
namespace Data {

// Class that identifies the "location" of a batch of data
// within an acuiqisition
class BatchMetadata
{
public:
    BatchMetadata() = default;

    BatchMetadata(uint32_t poolId, uint32_t firstFrame, uint32_t lastFrame)
        : poolId_(poolId)
        , firstFrame_(firstFrame)
        , lastFrame_(lastFrame)
    {
        assert(firstFrame <= lastFrame);
    }

    uint32_t PoolId() const { return poolId_; }
    uint32_t FirstFrame() const { return firstFrame_; }
    uint32_t LastFrame() const { return lastFrame_; }

private:
    // Not set in stone, just examples
    uint32_t poolId_;      // Identifier of pool of ZMWs.
    uint32_t firstFrame_;
    uint32_t lastFrame_;
};


class BatchDimensions
{
public:     // Functions
    uint32_t zmwsPerBatch() const
    {
        // TODO: Strictly speaking, there's an overflow risk here. Pretty sure,
        // however, that we won't have more than four billion ZMWs per batch in
        // the forseeable future.
        return laneWidth * lanesPerBatch;
    }

public:
    uint32_t laneWidth;
    uint32_t framesPerBatch;
    uint32_t lanesPerBatch;
};

}}}     // namespace PacBio::Mongo::Data

#endif // mongo_dataTypes_BatchMetadata_H_
