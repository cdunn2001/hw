#ifndef ChunkFileSink_H_
#define ChunkFileSink_H_

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
/// \file	ChunkFileSink.h
/// \brief  A block sink backed by the binary chunk-file.

#include <assert.h>
#include "Chunking.h"

namespace PacBio {
namespace Primary {
		
	/// The public API of this class defines the BlockSink Concept.
	class ChunkFileSink
	{
	public:
		ChunkFileSink(const std::string& fileName, const Chunking& dims)
			: fstrm_(fileName, std::ios::out | std::ios::binary)
			, dims_(dims)
		{
			if (!fstrm_.is_open())
				throw std::runtime_error("Failed to open output chunk file."); // TODO - FIXME - exceptions.
			
			// Write the file header
			fstrm_ << dims_;
		}

		virtual ~ChunkFileSink()
		{ }

		/// The total number of traces to be consumed by the sink
		size_t NumLanes() { return dims_.laneNum; }

		///  Write count samples from the buffer as the next block of data
		template <typename V>
		size_t WriteBlock(const V* vbuf, size_t count,
						  size_t laneIdx, size_t chunkIdx, uint16_t channelNum = 0)
		{
			// Don't write empty blocks.
			if (count == 0)
				return 0;

			Chunking hdr(dims_);
			hdr.sizeofSample = sizeof(V);
			hdr.channelNum = channelNum;
			hdr.sampleNum = count;
			hdr.laneNum = laneIdx;
			hdr.chunkNum = chunkIdx;

			// Write the block header
			fstrm_ << hdr;

			// Write the block data
			const char* buf = reinterpret_cast<const char*>(vbuf);
			fstrm_.write(buf, hdr.sizeofSample*count);

			return count;
		}
		
	private:
		std::ofstream fstrm_;
		Chunking dims_;
	};

}} // ::PacBio::Primary::Basecaller

#endif // ChunkFileSink_H_
