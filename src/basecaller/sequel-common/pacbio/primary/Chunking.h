#ifndef Chunking_H_
#define Chunking_H_

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
/// \file	Chunking.h
/// \brief	The class Chunking is used to represent header data stored in the
///         internal (non-production, testing purposes) binary chunk-file format.

#include <iostream>
#include <fstream>
#include <string>
#include <stdint.h>

namespace PacBio {
namespace Primary {

	struct Chunking
	{
		enum struct ElemType : uint16_t
		{
			UNSPECIFIED,
			INT8,  UINT8,
			INT16, UINT16,
			INT32, UINT32,
			FLOAT, DOUBLE,
		};

		struct ElemDescriptor
		{
			const char* name;
			const size_t size;
			ElemDescriptor(const char* nam, size_t siz) : name(nam), size(siz) { }
		};

		// 32-Byte Record
		uint32_t formatID;			// [ 4] Reserved (FORMAT_ID)
		uint16_t recordSize;		// [ 6] Size of this record-header in bytes [32]		
		uint16_t uword1;			// [ 8]
		uint16_t uword2;			// [10]
		uint16_t uword3;            // [12]
		
		uint16_t sampleType;		// [14] Unused, reserved for possible sample_type code.
		ElemType simdDataType;		// [16] data_type code (scalar data type for SIMD samples)
		uint16_t sizeofSample;		// [18] sizeof(sample_type) in bytes
		
		uint16_t channelNum;		// [20] ChannelIndex or NumChannels (in main header)
		uint32_t sampleNum;			// [24] N-of-samples in chunk or MaxNumSamples (in main header) 		
		uint32_t chunkNum;			// [28] ChunkIndex or NumChunks (in main header)
		uint32_t laneNum;			// [32] LaneIndex or NumLanes (in main header)

		Chunking()
			: formatID(0)
			, recordSize(32)
			, uword1(0), uword2(0), uword3(0)
			, sampleType(0)
			, simdDataType(ElemType::UNSPECIFIED)
			, sizeofSample(0)
			, channelNum(0)
			, sampleNum(0)
			, chunkNum(0)
			, laneNum(0)
		{ }

		ElemDescriptor SimdDataType() const
		{
			return SimdDataType(simdDataType);
		}

		size_t SimdChannelsPerSample() const
		{
			if (simdDataType == ElemType::UNSPECIFIED)
				return 1;

			return sizeofSample / SimdDataType().size;
		}

		bool IsSimd() const
		{
			return (SimdChannelsPerSample() > 1);
		}

		static ElemDescriptor SimdDataType(const Chunking::ElemType& t)
		{
			switch (t)
			{
			case ElemType::INT8:		return ElemDescriptor("Int8", 1);
			case ElemType::UINT8:		return ElemDescriptor("UInt8", 1);
			case ElemType::INT16:		return ElemDescriptor("Int16", 2);
			case ElemType::UINT16:		return ElemDescriptor("UInt16", 2);
			case ElemType::INT32:		return ElemDescriptor("Int32", 4);
			case ElemType::UINT32:		return ElemDescriptor("UInt32", 4);
			case ElemType::FLOAT:		return ElemDescriptor("Float", 4);
			case ElemType::DOUBLE:		return ElemDescriptor("Double", 8);
			default:					return ElemDescriptor("Unspecified", 0);
			}
		}
	};

	inline std::ostream& operator << (std::ostream& strm, const Chunking& r)
	{
		const char* p = reinterpret_cast<const char*>(static_cast<const Chunking*>(&r));
		strm.write(p, sizeof(Chunking));
		return strm;
	}
	
	inline std::ofstream& operator << (std::ofstream& strm, const Chunking& r)
	{
		const char* p = reinterpret_cast<const char*>(static_cast<const Chunking*>(&r));
		strm.write(p, sizeof(Chunking));
		return strm;
	}
	
	inline std::istream& operator >> (std::istream& strm, Chunking& r)
	{
		char* p = reinterpret_cast<char*>(static_cast<Chunking*>(&r));
		strm.read(p, sizeof(Chunking));
		return strm;
	}

	inline std::ifstream& operator >> (std::ifstream& strm, Chunking& r)
	{
		char* p = reinterpret_cast<char*>(static_cast<Chunking*>(&r));
		strm.read(p, sizeof(Chunking));
		return strm;
	}

	inline std::ostream& operator << (std::ostream& strm, const Chunking::ElemType& t)
	{
		strm << Chunking::SimdDataType(t).name;
		return strm;
	}

	inline std::istream& operator >> (std::istream& strm, Chunking::ElemType& t)
	{
		unsigned int itype;
		strm >> itype;
		
		if (itype > static_cast<unsigned int>(Chunking::ElemType::DOUBLE))
			t = Chunking::ElemType::UNSPECIFIED;
		else
			t = static_cast<Chunking::ElemType>(itype);

		return strm;
	}

}} // ::PacBio::Primary

#endif // Chunking_H_
