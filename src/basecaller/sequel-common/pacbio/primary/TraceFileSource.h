#ifndef Sequel_Primary_TraceFileSource_H_
#define Sequel_Primary_TraceFileSource_H_

// Copyright (c) 2016 Pacific Biosciences of California, Inc.
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
/// \file	TraceFileSource.h
/// \brief  A block source backed by an hdf5 trace file.


#include <assert.h>

// TODO: Replace this include with a forward declaration of SchunkLaserPowerChangeSet.
#include <pacbio/primary/LaserPowerChange.h>

#include "SequelTraceFile.h"
#include "SequelDefinitions.h"

namespace PacBio {
namespace Primary {

/// The API of this class defines the BlockSource concept.
/// From a trace file, blocks are served up as chunks, though the
/// size of the chunking is configurable
///
class TraceFileSource
{
public:
    explicit TraceFileSource(const std::string& file, size_t chunkSize) 
    : chunkIndex_(0)
    , laneIndex_(0)
    , numLanes_(0)
    , numChunks_(0)
    , chunkSize_(chunkSize)
    , traceFile_(file)
    {
        assert(traceFile_.valid);
        numLanes_ = traceFile_.NUM_HOLES / zmwsPerTranche;
        assert(traceFile_.NUM_HOLES % zmwsPerTranche == 0);

        numChunks_ = (traceFile_.NFRAMES + chunkSize - 1) / chunkSize;
    }

    size_t BlockSize()
    {
        return chunkSize_; 
    }

    // templated to avoid dependency of common on basecaller/common
    // Could potentially move this outside of common so that we can just
    // include the Traceblock directly
    template <class TraceBlock>
    size_t NextBlock(TraceBlock* ci) 
    {
        ci->LaneIndex(this->laneIndex_);
        ci->ChunkIndex(this->chunkIndex_);
        ci->SetZmwNumber(this->laneIndex_ * zmwsPerTranche);
        ci->BlockIndex(chunkIndex_);

        ci->IsLastBlock(true);
        ci->IsTerminus(chunkIndex_ >= numChunks_-1);

        size_t nextSize = chunkSize_;
        if (chunkIndex_ * chunkSize_ + nextSize > traceFile_.NFRAMES) 
        {
            // Write out a partial chunk to eat up the rest of the data
            if (traceFile_.NFRAMES > chunkIndex_ * chunkSize_)
            {
                nextSize = traceFile_.NFRAMES - chunkIndex_ * chunkSize_;
            } else {
                nextSize = 0;
            }
        }

        return nextSize;
    }

    size_t NextBlockStart()
    { 
        return chunkIndex_ * chunkSize_; 
    }
    
    size_t ReadBlock(m512s* vbuf, size_t count)
    { 
        if (count == 0) return 0;
        
        size_t frameOffset = chunkIndex_ * chunkSize_;
        size_t zmwOffset = laneIndex_ * zmwsPerTranche;
        size_t numRead = traceFile_.ReadLaneSegment(
                zmwOffset, frameOffset, 
                count, reinterpret_cast<int16_t*>(vbuf));

        // Increment for the next read
        if (laneIndex_ == numLanes_ - 1)
        {
            ++chunkIndex_;
            laneIndex_ = 0;
        }
        else
        {
            ++laneIndex_;
        }

        return numRead;
    }

    size_t Lanes() { return numLanes_; }

    // TODO: Populate the array with LPCs read from the trace file.
    /// Shared pointer to immutable array of laser power changes for
    /// the superchunk.
    std::shared_ptr<const SchunkLaserPowerChangeSet>
    LaserPowerChanges() const
    { return std::make_shared<const SchunkLaserPowerChangeSet>(); }

private:
    size_t chunkIndex_;
    size_t laneIndex_;
    size_t numLanes_;
    size_t numChunks_;
    size_t chunkSize_;

    SequelTraceFileHDF5 traceFile_;

};

}} // ::PacBio::Primary

#endif // Sequel_Primary_TraceFileSource_H_

