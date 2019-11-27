#ifndef Sequel_Primary_ChunkFileSource_H_
#define Sequel_Primary_ChunkFileSource_H_

// Copyright (c) 2015, Pacific Biosciences of California, Inc.
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
/// \file	ChunkFileSource.h
/// \brief  A block source backed by the binary chunk-file.


#include <assert.h>
#include <iterator>
#include <pacbio/PBException.h>
#include <pacbio/logging/Logger.h>
#include "Chunking.h"
#include "SequelDefinitions.h"
#include <iostream>

// TODO: Replace include with forward declaration of SchunkLaserPowerChangeSet.
#include <pacbio/primary/LaserPowerChange.h>

namespace PacBio {
namespace Primary {

/// The API of this class defines the BlockSource concept.
/// From a "chunk_file", blocks are served up as complete chunks.
///
class ChunkFileSource
{
public:     // Structors

    ChunkFileSource(const std::string& fileName)
        : fstrm_(fileName, std::ios::in | std::ios::binary)
        , laneIndex_(0), chunkIndex_(0)
        , sizeofSample_(0), remSamples_(0)
    {
        if (!fstrm_.is_open())
        {
            throw PBException("Failed to open input chunk file.");
        }

        ReadHeader();
    }

    ~ChunkFileSource() = default;

public:     // BlockSource concept methods

    // read the header
    void ReadHeader()
    {
        // Read the file header
        fstrm_ >> dims_;
        if (!fstrm_)
        {
            throw PBException("Failed to read header");
        }
        sizeofSample_ = dims_.sizeofSample;
        PBLOG_DEBUG << "ReadHeader() sizeofsample:" << sizeofSample_;
        // This source does not support multi-channel (by chunk) input.
        assert(dims_.channelNum == 1);
    }

    /// go back to the first lane
    void Rewind()
    {
        laneIndex_ = 0;
        chunkIndex_ = 0;
        sizeofSample_ = 0;
        remSamples_ = 0;
        fstrm_.clear();
        fstrm_.seekg(0,fstrm_.beg);
        ReadHeader();
    }

    /// The default n-of-samples per block read
    size_t BlockSize() const { return dims_.sampleNum; }

    /// Prepare to read the next block:
    /// Return block size and bookkeeping indices.
    ///
    size_t NextBlock(size_t& laneIdx, size_t& chunkIdx,
                     bool& chunkCompleted, bool& terminus)
    {
        // For chunk files, blocks==chunks, so the next block completes the chunk, by definition.
        chunkCompleted = true;

        // The API allows a partial read, but the termination logic doesn't really support it.
        assert(remSamples_ == 0);
        if (remSamples_ > 0)
        {
            // We remain in the current chunk...somehow.
            laneIdx = laneIndex_;
            chunkIdx = chunkIndex_;
            terminus = chunkIdx >= (dims_.chunkNum - 1);

            return remSamples_;
        }

        // Read the next chunk header
        Chunking rec;
        fstrm_ >> rec;

        // At EOF, advance bookkeeping in (chunk, trace) odometer style
        if (fstrm_.eof())
        {
            if (laneIndex_ == dims_.laneNum - 1)
            {
                laneIndex_ = 0;
                ++chunkIndex_;
            }
            else
            {
                ++laneIndex_;
            }
            remSamples_ = 0;
        }
        else
        {
            // Use the chunk header to update bookkeeping
            laneIndex_ = rec.laneNum;
            assert(laneIndex_ < dims_.laneNum);

            chunkIndex_ = rec.chunkNum;
            assert(chunkIndex_ < dims_.chunkNum);

            remSamples_ = rec.sampleNum;
            sizeofSample_ = rec.sizeofSample;
        }


        laneIdx = laneIndex_;
        chunkIdx = chunkIndex_;
        terminus = chunkIdx >= (dims_.chunkNum - 1);

        return remSamples_;
    }

    // templated to avoid dependency of common on basecaller/common
    // Could potentially move this outside of common so that we can just
    // include the Traceblock directly
    template <class TraceBlock>
    size_t NextBlock(TraceBlock* ci) 
    {

        // Read the next chunk header
        assert(remSamples_ == 0);
        Chunking rec;
        fstrm_ >> rec;

        // At EOF.  Still have to let current data filter through the pipe, so
        // increment lane odometer style, but keep chunk as is.
        if (fstrm_.eof())
        {
            if (laneIndex_ == dims_.laneNum - 1)
            {
                laneIndex_ = 0;
            }
            else
            {
                ++laneIndex_;
            }
        }
        else
        {
            // Use the chunk header to update bookkeeping
            laneIndex_ = rec.laneNum;
            assert(laneIndex_ < dims_.laneNum);

            chunkIndex_ = rec.chunkNum;
            assert(chunkIndex_ < dims_.chunkNum);

            remSamples_ = rec.sampleNum;
            sizeofSample_ = rec.sizeofSample;
        }
        
        ci->LaneIndex(this->laneIndex_);
        ci->ChunkIndex(this->chunkIndex_);
        ci->SetZmwNumber(this->laneIndex_ * zmwsPerTranche);
        ci->BlockIndex(chunkIndex_);

        ci->IsLastBlock(true);
        ci->IsTerminus(chunkIndex_ >= (dims_.chunkNum - 1));

        return remSamples_;
    }

    size_t NextBlockStart() {
        return dims_.sampleNum * chunkIndex_;
    }

    /// Read at most the requested number of samples of type V;
    /// Return the number of samples actually read. 
    ///
    template <typename V>
    size_t ReadBlock(V* vbuf, size_t count)
    {
        // Provide some semblance of type-checking
        if(sizeofSample_ != sizeof(V))
        {
            throw PBException("sizeofsample=" + std::to_string(sizeofSample_) + " expected sizeof(V)=" + std::to_string(sizeof(V)));
        }

        if (fstrm_.eof())
            return 0;

        // Cast to a buffer of bytes.
        char* buf = reinterpret_cast<char*>(vbuf);

        // Get the initial stream pos
        std::streampos before = fstrm_.tellg();

        // Read data into the container; never read past the end of the block.
        count = std::min(count, remSamples_);
        fstrm_.read(buf, sizeofSample_*count);

        size_t nSamplesRead = (fstrm_.tellg() - before) / sizeofSample_;
        remSamples_ -= nSamplesRead;

        return nSamplesRead;
    }

    void SkipBlock()
    {
        size_t bytes = remSamples_ * sizeofSample_;
        fstrm_.seekg(bytes,fstrm_.cur);
        remSamples_ = 0;
    }

public:     // Auxiliary properties

    /// The total number of lanes provided by the source
    size_t NumLanes() const { return dims_.laneNum; }

    /// The total number of (super)chunks provided by the source
    size_t NumChunks() const { return dims_.chunkNum; }

    size_t SizeOfSample() const { return sizeofSample_;}

    /// Shared pointer to immutable array of laser power changes for
    /// the superchunk. The array is always empty because chunk files do not
    /// support LPCs.
    std::shared_ptr<const SchunkLaserPowerChangeSet>
    LaserPowerChanges() const
    { return std::make_shared<const SchunkLaserPowerChangeSet>(); }

private:    // Data
    std::ifstream fstrm_;
    Chunking dims_;

    size_t laneIndex_;
    size_t chunkIndex_;

    size_t sizeofSample_;	// sizeof(sample_type) in the current block
    size_t remSamples_;		// N-of-samples remaining to be read
};

}} // ::PacBio::Primary

#endif // Sequel_Primary_ChunkFileSource_H_
