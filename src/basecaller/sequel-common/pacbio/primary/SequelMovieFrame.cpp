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
/// \brief  Implementation of frame object, used in sequel movies.


#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <iostream>
#include <iomanip>
#include <random>

#include <pacbio/primary/FrameClass.h>
#include <pacbio/primary/SequelMovieFrame.h>
#include <pacbio/primary/SequelROI.h>
#include <pacbio/PBException.h>
#include <pacbio/text/String.h>
#include <pacbio/logging/Logger.h>

using namespace PacBio;

#define NOT_IMPLEMENTED {throw PBException("Not implemented");}

namespace PacBio {
namespace Primary {

static uint32_t GetRowPduMagicWord(FrameClass frameClass)
{
    switch(frameClass)
    {
    case FrameClass::Format1C4A:
        return 0x00010002;
    case FrameClass::Format2C2A:
        return 0x00000002;
    default:
        throw PBException("FrameClass not supported");
    }
}


    template<class T>
    SequelMovieFrame<T>::SequelMovieFrame(uint32_t rows , uint32_t cols )
        : NROW(rows),
          NCOL(cols),
          index(0),
          timestamp(0),
          cameraConfiguration(0xFFFFFFFF),
          data(nullptr),
          valid(true)
    {
        int32_t total = rows * cols;
        data = new T[total];
    }
    
    template<class T>
    SequelMovieFrame<T>::SequelMovieFrame(const SequelRectangularROI& roi)
        : NROW(roi.NumPixelRows()),
          NCOL(roi.NumPixelCols()),
          index(0),
          timestamp(0),
          cameraConfiguration(0xFFFFFFFF),
          data(nullptr),
          valid(true)
    {
        int32_t total = NROW * NCOL;
        data = new T[total];
    }

    /// Constructor from existing buffer
    ///
    template<class T>
    SequelMovieFrame<T>::SequelMovieFrame(const int16_t* rawBuffer, uint32_t rows, uint32_t cols)
        : NROW(rows),
          NCOL(cols),
          index(0),
          timestamp(0),
          cameraConfiguration(0),
          valid(true)
    {
        int32_t total = rows * cols;
        data = new T[total];
        memcpy(data, rawBuffer, total * sizeof(T));
    }

    /// Copy constructor (deep copy) (lvalue) differing types
    ///

    template<class T>
    template<class Y>
    SequelMovieFrame<T>::SequelMovieFrame(const SequelMovieFrame<Y>& frame)
        : NROW(frame.NROW),
          NCOL(frame.NCOL),
          index(frame.index),
          timestamp(frame.timestamp),
          cameraConfiguration(frame.cameraConfiguration),
          data(nullptr),
          valid(frame.valid)
    {
        int32_t total = NROW * NCOL;
        data = new T[total];

        for (int32_t i = 0; i < total; i++)
        {
            data[i] = static_cast<T>(frame.data[i]);
        }
    }

    /// Copy constructor (deep copy) (lvalue) same types
    template<class T>
    SequelMovieFrame<T>::SequelMovieFrame(const SequelMovieFrame<T>& frame)
        : NROW(frame.NROW),
          NCOL(frame.NCOL),
          index(frame.index),
          timestamp(frame.timestamp),
          cameraConfiguration(frame.cameraConfiguration),
          data(nullptr),
          valid(frame.valid)
    {
        int32_t total = NROW * NCOL;
        data = new T[total];
        if (data == nullptr)
        {
            throw PBException("could not allocate " + std::to_string(NROW) + " x " + std::to_string(NCOL));
        }
        memcpy(data, frame.data, total * sizeof(T));
    }

    /// Copy constructor (with move) (rvalue)
    ///
    template<class T>
    SequelMovieFrame<T>::SequelMovieFrame::SequelMovieFrame(SequelMovieFrame<T>&& frame)
        : NROW(frame.NROW),
          NCOL(frame.NCOL),
          index(frame.index),
          timestamp(frame.timestamp),
          cameraConfiguration(frame.cameraConfiguration),
          data(nullptr),
          valid(frame.valid)
    {
        std::swap(data, frame.data);
    }

    /// copy assignment, different type
    ///
    template<class T>
    template<class Y>
    SequelMovieFrame<T>& SequelMovieFrame<T>::operator=(const SequelMovieFrame<Y>& frame)
    {
        NROW = frame.NROW;
        NCOL = frame.NCOL;
        index = frame.index;
        timestamp = frame.timestamp;
        cameraConfiguration = frame.cameraConfiguration;
        valid = frame.valid;

        int32_t total = NROW * NCOL;
        if (data) delete[] data;
        data = new T[total];

        for (int32_t i = 0; i < total; i++)
        {
            data[i] = static_cast<T>(frame.data[i]);
        }
        return *this;
    }

/// copy assignment, same type
///
template<class T>
SequelMovieFrame<T>& SequelMovieFrame<T>::operator=(const SequelMovieFrame<T>& frame)
{
    NROW = frame.NROW;
    NCOL = frame.NCOL;
    index = frame.index;
    timestamp = frame.timestamp;
    cameraConfiguration = frame.cameraConfiguration;
    valid = frame.valid;

    int32_t total = NROW * NCOL;
    if (data) delete[] data;
    data = new T[total];

    memcpy(data, frame.data, total * sizeof(T));
    return *this;
}

    /// move assignment
    ///
    template<class T>
    template<typename Y>
    SequelMovieFrame<T>& SequelMovieFrame<T>::operator=(SequelMovieFrame<Y>&& frame)
    {
        NROW = frame.NROW;
        NCOL = frame.NCOL;
        index = frame.index;
        timestamp = frame.timestamp;
        cameraConfiguration = frame.cameraConfiguration;
        valid = frame.valid;
        data = nullptr;

        std::swap(data, frame.data);
        return *this;
    }

    template<class T>
    SequelMovieFrame<T>::~SequelMovieFrame()
    {
        delete[] data;
        data = NULL;
    }

    template<class T>
    void SequelMovieFrame<T>::Resize(uint32_t rows, uint32_t cols)
    {
        NROW = rows;
        NCOL = cols;
        delete[] data;
        int32_t total = rows * cols;
        data = new T[total];
    }

    template<class T>
    template<typename Y>
    void SequelMovieFrame<T>::TiledReplicate(const SequelMovieFrame<Y>& frame)
    {
        for(uint32_t row=0;row<NROW;row++)
        {
            for(uint32_t col=0;col<NCOL;col++)
            {
                size_t dst = col + row*NCOL;
                size_t src = (col % frame.NCOL) + (row % frame.NROW) * frame.NCOL;
                data[dst] = frame.data[src];
            }
        }
    }

    template<class T>
    template<typename Y>
    void SequelMovieFrame<T>::Insert(uint32_t rowOffset, uint32_t colOffset, const SequelMovieFrame<Y>& frame)
    {
        if (rowOffset + frame.NROW > NROW)
        {
            std::stringstream ss;
            ss << "Insert failed because rowOffset:" << rowOffset << "frame.NROW:" << frame.NROW <<
                " > NROW:" << NROW;
            throw PBException(ss.str());
        }
        if (colOffset + frame.NCOL > NCOL)
        {
            std::stringstream ss;
            ss << "Insert failed because colOffset:" << rowOffset << "frame.NCOL:" << frame.NCOL <<
                " > NCOL:" << NCOL;
            throw PBException(ss.str());
        }

        for(uint32_t row=0;row<frame.NROW;row++)
        {
            for(uint32_t col=0;col<frame.NCOL;col++)
            {
                size_t dst = (col  + colOffset) + (row + rowOffset) * frame.NCOL;
                size_t src = col + row * frame.NCOL;
                data[dst] = frame.data[src];
            }
        }
    }

    template<class T>
    SequelMovieFrame<T>& SequelMovieFrame<T>::SetDefaultValue(T defaultValue)
    {
        int32_t total = NROW * NCOL;
        if (defaultValue == 0)
        {
            memset(data,0,total * sizeof(T));
        }
        else
        {
            for (int32_t i = 0; i < total; i++)
            {
                data[i] = defaultValue;
            }
        }
        return *this;
    }

    template<class T>
    void SequelMovieFrame<T>::DumpSummary(std::ostream& s, const std::string& prefix , uint32_t rows , uint32_t cols ) const
    {
        uint32_t i, j;
        s << prefix << "MovieFrame Index:" << index << " Time:" << timestamp << " Config:" << cameraConfiguration <<
            " NROW:" << NROW << " NCOL:" << NCOL << "\n";

        for (j = 0; j < NROW; j++)
        {
            if (j < rows || j >= NROW - rows)
            {
                s << prefix;
                for (i = 0; i < cols; i++)
                    s << ' ' << std::setw(5) << data[j * NCOL + i];
                s << " ... ";
                for (i = NCOL - cols; i < NCOL; i++)
                    s << ' ' << std::setw(5) << data[j * NCOL + i];
                s << std::endl;
            }
            else if (j == rows)
            {
                s << prefix;
                for (i = 0; i < cols; i++)
                    s << ' ' << std::setw(5) << "  ...";
                s << " ... ";
                for (i = NCOL - cols; i < NCOL; i++)
                    s << ' ' << std::setw(5) << "  ...";
                s << std::endl;
            }
        }
    }


#if 0
    // extracts just the index
    template<class T>
    uint64_t SequelMovieFrame<T>::PeekIndex(const void* streamBuffer0)
    {
        const uint8_t* streamBuffer = (const uint8_t*) streamBuffer0;
        uint32_t magicWord;
        memcpy(&magicWord, streamBuffer, 4);
        streamBuffer += 4;
        if (magicWord == 0x01000001)
        {
            uint64_t index;
            memcpy(&index, streamBuffer, 8);
            return index;
        }
        else
        {
            return 0;
        }
    }
#endif

    // returns number of bytes parsed from streamBuffer0
    template<class T>
    size_t SequelMovieFrame<T>::Unpack(const void* streamBuffer0, uint32_t rowsTransfered, uint32_t maxLength)
    {
        const uint8_t* streamBuffer = (const uint8_t*) streamBuffer0;
        const uint8_t* const start = streamBuffer;

        valid = false;
        bool headerValid = false;
        uint32_t missingRows = 0;

          if (NCOL & 7)
          {
              throw PBException("ncols must be multiple of 8");
          }

#ifdef VERBOSE
        for(uint32_t i=0;i<maxLength;i+=8)
        {
            if ((i % 64) == 0) std::cout << "Unpack(): bytes:" << std::hex << std::setw(16)<< std::setfill('0')<< (uint64_t)streamBuffer0 + i <<": ";
            std::cout << " "    << std::setw(16)<< std::setfill('0')<< *(uint64_t*)&streamBuffer[i] ;
            if ((i % 64) == 56) std::cout << std::endl;
        }
        std::cout << "\nunpacking !..." << std::dec << std::endl;
#endif


          uint32_t magicWord;
          memcpy(&magicWord, streamBuffer, 4);
          streamBuffer += 4;
          if (magicWord == 0x01000001)
          {
              headerValid = true;
          }
          else
          {
              headerValid = false;
              std::cout << "ERROR: bad Header PDU magic word: " << std::hex << magicWord << std::dec << std::endl;
          }

          memcpy(&index, streamBuffer, 8);
          streamBuffer += 8;
          memcpy(&timestamp, streamBuffer, 8);
          streamBuffer += 8;
          memcpy(&cameraConfiguration, streamBuffer, 4);
          streamBuffer += 4;


#ifdef VERBOSE
        std::cout << "unpack: frame header, index:" << index << " timestamp:" << timestamp << " confg:" << cameraConfiguration << std::endl;
#endif

#define XFER_SIZE 64
        uint32_t pageOffset = (streamBuffer - start) & (XFER_SIZE - 1);
        if (pageOffset != 0)
        {
            streamBuffer += (XFER_SIZE - pageOffset);
        }

         uint32_t row;
         uint32_t goodRows = 0;
         for (row = 0; row < NROW && goodRows < rowsTransfered; row++)
         {
              memcpy(&magicWord, streamBuffer, 4);
              streamBuffer += 4;
              FrameClass frameClass = FrameClass::UNKNOWN;
              uint32_t maskedMagicWord = (magicWord & 0x00FFFFFF);
              if (maskedMagicWord == 0x00000002)
              {
                  frameClass = FrameClass::Format2C2A;
                  goodRows++;
              }
              else if (maskedMagicWord == 0x00010002)
              {
                  frameClass = FrameClass::Format1C4A;
                  goodRows++;
              }
              else
              {
                  std::cout << std::hex << "unpack: BAD RowPDU magic word: " << magicWord << std::dec << std::endl;
                  break;
              }
              int bitsPerPixel = magicWord >> 24;
              if (bitsPerPixel > 16) throw PBException("Unsupported pixel bit size " + std::to_string(bitsPerPixel));
              uint32_t rowLen = GetPackedRowPduSize(bitsPerPixel,frameClass);

              if ((uint32_t)(streamBuffer - start) + rowLen > maxLength)
              {
                  std::cout << std::hex << "unpack: rowLen:" << rowLen << " maxLength:" << maxLength << " pos:" <<
                  rowLen << std::dec << std::endl;
                  throw PBException("unpack: buffer unpack error. Length too big! corrupted buffer!");
              }


              // skip padding for spider
              if (magicWord == 0xC010002)
                  streamBuffer += 4;

              uint32_t rowCompare;
              memcpy(&rowCompare, streamBuffer, 4);
              streamBuffer += 4;
#ifdef VERBOSE
            std::cout << "unpack: frame row " << rowCompare << std::endl;
#endif

            if (rowCompare != row)
            {
                if (rowCompare < NROW && rowCompare > row)
                {
                    uint32_t missing = (rowCompare - row);
                    std::cout << "unpack: MISSING ROWs " << row << " to " << (rowCompare - 1) << std::endl;
                    if (FillMissingRows)
                    {
                        void* ptr = &data[row * NCOL];
                        memset(ptr, 0, missing * sizeof(T) * NCOL);
                    }
                    missingRows += missing;
                    row = rowCompare;
                }
                else
                {
                    std::cout << "unpack: corrupted row PDU, got row: " << rowCompare << ", expected: " << row <<
                        ". frame will be dropped" << std::endl;
                    break;
                }
            }

            T* dst = &data[row * NCOL];
            switch (bitsPerPixel)
            {
            case 10:
                Unpack10BitRow(streamBuffer, dst);
                break;
            case 12:
                Unpack12BitRow(streamBuffer, dst);
                break;
            }


            // unpad to 32-bit (4 byte) alignment
            while (((uint64_t) streamBuffer) & 3)
            {
                streamBuffer++;
            }
            pageOffset = (streamBuffer - start) & (XFER_SIZE - 1);
            if (pageOffset != 0)
            {
                streamBuffer += (XFER_SIZE - pageOffset);
            }
#ifdef VERBOSE
            for(uint32_t j=0;j<NCOL;j++)
            {
                std::cout << data[row*NCOL + j] << " ";
            }
            std::cout << std::endl;
#endif
        }
#ifdef VERBOSE
        std::cout << "unpack: DONE" << std::endl;
#endif
        if (NROW != row)
        {
            std::cout << "unpack: MISSING ROWs " << row << " to " << (NROW - 1) << std::endl;
            missingRows += (NROW - row);
        }
        if (FillMissingRows)
        {
            valid = headerValid && (missingRows + goodRows) == NROW;
        }
        else
        {
            valid = headerValid && missingRows == 0 && goodRows == NROW;
        }
        return (size_t)(streamBuffer - start);
    }

    template<>
    void SequelMovieFrame<double>::Unpack12BitRow(const uint8_t*& /*streamBuffer*/, double* /*ptr*/) NOT_IMPLEMENTED

     template<>
    void SequelMovieFrame<float>::Unpack12BitRow(const uint8_t*& /*streamBuffer*/, float* /*ptr*/) NOT_IMPLEMENTED

    template<>
    void SequelMovieFrame<int8_t>::Unpack12BitRow(const uint8_t*& /*streamBuffer*/, int8_t* /*ptr*/) NOT_IMPLEMENTED

    template<>
    void SequelMovieFrame<int16_t>::Unpack12BitRow(const uint8_t*& streamBuffer, int16_t* ptr)
    {
        // LCM of 8 bits and 10 bits is 40 bits
        // 40 bits is 4 pixels and 5 bytes
        // so we split 4 pixels among 5 bytes

#if 1
        uint32_t lastJ = NCOL / 20 * 20;
        for (uint32_t j = 0; j < lastJ; j += 20)
        {
            uint64_t src0 = *(uint64_t * )(streamBuffer);     // bits 0 to 63
            uint64_t src1 = *(uint64_t * )(streamBuffer + 7); // bits 56 to 119
            uint64_t src2 = *(uint64_t * )(streamBuffer + 15);     // bits 0 to 63
            uint64_t src3 = *(uint64_t * )(streamBuffer + 22); // bits 56 to 119

            ptr[0] = static_cast<int16_t>(src0 & 0xFFF);
            ptr[1] = static_cast<int16_t>((src0 >> 12) & 0xFFF);
            ptr[2] = static_cast<int16_t>((src0 >> 24) & 0xFFF);
            ptr[3] = static_cast<int16_t>((src0 >> 36) & 0xFFF);
            ptr[4] = static_cast<int16_t>((src0 >> 48) & 0xFFF);

            ptr[5] = static_cast<int16_t>((src1 >> 4) & 0xFFF); // 60
            ptr[6] = static_cast<int16_t>((src1 >> 16) & 0xFFF); // 72
            ptr[7] = static_cast<int16_t>((src1 >> 28) & 0xFFF); // 84
            ptr[8] = static_cast<int16_t>((src1 >> 40) & 0xFFF); // 96
            ptr[9] = static_cast<int16_t>((src1 >> 52) & 0xFFF); // 108

            ptr[10] = static_cast<int16_t>(src2 & 0xFFF);
            ptr[11] = static_cast<int16_t>((src2 >> 12) & 0xFFF);
            ptr[12] = static_cast<int16_t>((src2 >> 24) & 0xFFF);
            ptr[13] = static_cast<int16_t>((src2 >> 36) & 0xFFF);
            ptr[14] = static_cast<int16_t>((src2 >> 48) & 0xFFF);

            ptr[15] = static_cast<int16_t>((src3 >> 4) & 0xFFF); // 60
            ptr[16] = static_cast<int16_t>((src3 >> 16) & 0xFFF); // 72
            ptr[17] = static_cast<int16_t>((src3 >> 28) & 0xFFF); // 84
            ptr[18] = static_cast<int16_t>((src3 >> 40) & 0xFFF); // 96
            ptr[19] = static_cast<int16_t>((src3 >> 52) & 0xFFF); // 108

            streamBuffer += 30;
            ptr += 20;
        }
        uint32_t mod = NCOL - lastJ;
        if (mod > 0)
        {
            uint64_t src0 = *(uint64_t * )(streamBuffer);     // bits 0 to 63
            uint64_t src1 = *(uint64_t * )(streamBuffer + 7); // bits 56 to 119
            uint64_t src2 = *(uint64_t * )(streamBuffer + 15);     // bits 0 to 63
            uint64_t src3 = *(uint64_t * )(streamBuffer + 22); // bits 56 to 119

            ptr[0] = src0 & 0xFFF;
            ptr[1] = (src0 >> 12) & 0xFFF;
            ptr[2] = (src0 >> 24) & 0xFFF;
            ptr[3] = (src0 >> 36) & 0xFFF;
            ptr[4] = (src0 >> 48) & 0xFFF;

            ptr[5] = (src1 >> 4) & 0xFFF; // 60
            ptr[6] = (src1 >> 16) & 0xFFF; // 72
            ptr[7] = (src1 >> 28) & 0xFFF; // 84
            streamBuffer += 10;
            if (mod > 8)
            {
                ptr[8] = (src1 >> 40) & 0xFFF; // 80
                ptr[9] = (src1 >> 52) & 0xFFF; // 90

                ptr[10] = src2 & 0xFFF;
                ptr[11] = (src2 >> 12) & 0xFFF;
                ptr[12] = (src2 >> 24) & 0xFFF;
                ptr[13] = (src2 >> 36) & 0xFFF;
                ptr[14] = (src2 >> 48) & 0xFFF;

                ptr[15] = (src3 >> 4) & 0xFFF; // 60
                streamBuffer += 10;
                if (mod > 16)
                {
                    ptr[16] = (src3 >> 16) & 0xFFF; // 72
                    ptr[17] = (src3 >> 28) & 0xFFF; // 84
                    ptr[18] = (src3 >> 40) & 0xFFF; // 96
                    ptr[19] = (src3 >> 52) & 0xFFF; // 108
                    streamBuffer += 10;
                }
            }
        }
#else
        uint16_t* ptr = &data[row*NCOL];
        for(uint32_t j=0;j<NCOL;j+=4)
        {
            ptr[0] =  streamBuffer[0]      | ((streamBuffer[1] & 0x03)<<8);
            ptr[1] = (streamBuffer[1] >>2) | ((streamBuffer[2] & 0x0F)<<6);
            ptr[2] = (streamBuffer[2] >>4) | ((streamBuffer[3] & 0x3F)<<4);
            ptr[3] = (streamBuffer[3] >>6) | ((streamBuffer[4] & 0xFF)<<2);
            streamBuffer += 5;
            ptr += 4;
        }
#endif
    }

    template<>
    void SequelMovieFrame<float>::Unpack10BitRow(const uint8_t*& /*streamBuffer*/, float* /*ptr*/) NOT_IMPLEMENTED

    template<>
    void SequelMovieFrame<double>::Unpack10BitRow(const uint8_t*& /*streamBuffer*/, double* /*ptr*/) NOT_IMPLEMENTED

    template<>
    void SequelMovieFrame<int8_t>::Unpack10BitRow(const uint8_t*& /*streamBuffer*/, int8_t* /*ptr*/) NOT_IMPLEMENTED

    template<>
    void SequelMovieFrame<int16_t>::Unpack10BitRow(const uint8_t*& streamBuffer, int16_t* ptr)
    {
#if 1
        uint32_t lastJ = NCOL / 24 * 24;
        for (uint32_t j = 0; j < lastJ; j += 24)
        {
            uint64_t src0 = *(uint64_t * )(streamBuffer);     // bits 0 to 63
            uint64_t src1 = *(uint64_t * )(streamBuffer + 7); // bits 56 to 119
            uint64_t src2 = *(uint64_t * )(streamBuffer + 15);     // bits 0 to 63
            uint64_t src3 = *(uint64_t * )(streamBuffer + 22); // bits 56 to 119

            ptr[0] = src0 & 0x3FF;
            ptr[1] = (src0 >> 10) & 0x3FF;
            ptr[2] = (src0 >> 20) & 0x3FF;
            ptr[3] = (src0 >> 30) & 0x3FF;
            ptr[4] = (src0 >> 40) & 0x3FF;
            ptr[5] = (src0 >> 50) & 0x3FF;

            ptr[6] = (src1 >> 4) & 0x3FF; // 60
            ptr[7] = (src1 >> 14) & 0x3FF; // 70
            ptr[8] = (src1 >> 24) & 0x3FF; // 80
            ptr[9] = (src1 >> 34) & 0x3FF; // 90
            ptr[10] = (src1 >> 44) & 0x3FF; // 100
            ptr[11] = (src1 >> 54) & 0x3FF; // 110

            ptr[12] = src2 & 0x3FF;
            ptr[13] = (src2 >> 10) & 0x3FF;
            ptr[14] = (src2 >> 20) & 0x3FF;
            ptr[15] = (src2 >> 30) & 0x3FF;
            ptr[16] = (src2 >> 40) & 0x3FF;
            ptr[17] = (src2 >> 50) & 0x3FF;

            ptr[18] = (src3 >> 4) & 0x3FF; // 60
            ptr[19] = (src3 >> 14) & 0x3FF; // 70
            ptr[20] = (src3 >> 24) & 0x3FF; // 80
            ptr[21] = (src3 >> 34) & 0x3FF; // 90
            ptr[22] = (src3 >> 44) & 0x3FF; // 100
            ptr[23] = (src3 >> 54) & 0x3FF; // 110

            streamBuffer += 30;
            ptr += 24;
        }
        uint32_t mod = NCOL - lastJ;
        if (mod > 0)
        {
            uint64_t src0 = *(uint64_t * )(streamBuffer);     // bits 0 to 63
            uint64_t src1 = *(uint64_t * )(streamBuffer + 7); // bits 56 to 119
            uint64_t src2 = *(uint64_t * )(streamBuffer + 15);     // bits 0 to 63
            uint64_t src3 = *(uint64_t * )(streamBuffer + 22); // bits 56 to 119

            ptr[0] = src0 & 0x3FF;
            ptr[1] = (src0 >> 10) & 0x3FF;
            ptr[2] = (src0 >> 20) & 0x3FF;
            ptr[3] = (src0 >> 30) & 0x3FF;
            ptr[4] = (src0 >> 40) & 0x3FF;
            ptr[5] = (src0 >> 50) & 0x3FF;

            ptr[6] = (src1 >> 4) & 0x3FF; // 60
            ptr[7] = (src1 >> 14) & 0x3FF; // 70
            streamBuffer += 10;
            if (mod > 8)
            {
                ptr[8] = (src1 >> 24) & 0x3FF; // 80
                ptr[9] = (src1 >> 34) & 0x3FF; // 90
                ptr[10] = (src1 >> 44) & 0x3FF; // 100
                ptr[11] = (src1 >> 54) & 0x3FF; // 110

                ptr[12] = src2 & 0x3FF;
                ptr[13] = (src2 >> 10) & 0x3FF;
                ptr[14] = (src2 >> 20) & 0x3FF;
                ptr[15] = (src2 >> 30) & 0x3FF;
                streamBuffer += 10;
                if (mod > 16)
                {
                    ptr[16] = (src2 >> 40) & 0x3FF;
                    ptr[17] = (src2 >> 50) & 0x3FF;

                    ptr[18] = (src3 >> 4) & 0x3FF; // 60
                    ptr[19] = (src3 >> 14) & 0x3FF; // 70
                    ptr[20] = (src3 >> 24) & 0x3FF; // 80
                    ptr[21] = (src3 >> 34) & 0x3FF; // 90
                    ptr[22] = (src3 >> 44) & 0x3FF; // 100
                    ptr[23] = (src3 >> 54) & 0x3FF; // 110
                    streamBuffer += 10;
                }
            }
        }
#else
        uint16_t* ptr = &data[row*NCOL];
        for(uint32_t j=0;j<NCOL;j+=4)
        {
            ptr[0] =  streamBuffer[0]      | ((streamBuffer[1] & 0x03)<<8);
            ptr[1] = (streamBuffer[1] >>2) | ((streamBuffer[2] & 0x0F)<<6);
            ptr[2] = (streamBuffer[2] >>4) | ((streamBuffer[3] & 0x3F)<<4);
            ptr[3] = (streamBuffer[3] >>6) | ((streamBuffer[4] & 0xFF)<<2);
            streamBuffer += 5;
            ptr += 4;
        }
#endif

#ifdef VERBOSE
        for(uint32_t j=0;j<NCOL;j++)
        {
            std::cout << data[row*NCOL + j] << " ";
        }
        std::cout << std::endl;
#endif


#ifdef VERBOSE
        std::cout << "unpack: DONE" << std::endl;
#endif
    }

    // packed format is
    // <number of words>
    // <words>
    // <number of words>
    // <words>
    // <number of words>
    // <words>
    // 0x00000000
     
    // returns number of bytes packed into streamBuffer
    template<class T>
    size_t SequelMovieFrame<T>::Pack(uint8_t* streamBuffer, size_t maxLen, uint32_t BitsPerPixel, FrameClass frameClass) const
    {
        uint8_t* startPtr = streamBuffer;
        uint32_t x = 0;

        uint32_t headerLen = PackHeader(streamBuffer);
#ifdef VERBOSE
        std::cout << "pack: headerLen:" << headerLen << std::endl;
#endif
        uint16_t paddedHeaderLen = GetPackedHeaderPduSize();
        if (headerLen > paddedHeaderLen)
        {
            std::cout << "pack: GetPackedHeaderPduSize" << GetPackedHeaderPduSize() << std::endl;
            throw PBException("pack: Header PDU too long");
        }
        streamBuffer += (paddedHeaderLen);
        x += (paddedHeaderLen);

        uint32_t pageOffset = (streamBuffer - startPtr) & (XFER_SIZE - 1);
        if (pageOffset != 0)
        {
            streamBuffer += (XFER_SIZE - pageOffset);
        }

        uint32_t rowLen = GetPackedRowPduSize(BitsPerPixel, frameClass);
        for (uint32_t row = 0; row < NROW; row++)
        {
            uint32_t len = PackLine(row, streamBuffer, BitsPerPixel, frameClass);
            if (len > rowLen)
            {
                throw PBException("pack: row pdu mismatch");
            }
            streamBuffer += (rowLen);
            x += (rowLen);
            if (x > maxLen)
            {
                throw PBException("pack: buffer overflow");
            }
            pageOffset = (streamBuffer - startPtr) & (XFER_SIZE - 1);
            if (pageOffset != 0)
            {
                streamBuffer += (XFER_SIZE - pageOffset);
            }
#if 0
            {
                static int rolling = 0;
                if (rolling++ == 900) {
                    std::cout << "DROPPED ROW!" << std::endl;
                    row++;
                    rolling=0;
                }
            }
#endif

        }

        if (x > maxLen)
        {
            throw PBException("pack: buffer overflow");
        }
        if ((streamBuffer - startPtr) & (XFER_SIZE - 1))
        {
            std::cout << (uint64_t) streamBuffer << ":" << (uint64_t) startPtr << " "
                      << (uint64_t)((streamBuffer - startPtr)) << " "
                      << (uint64_t)((streamBuffer - startPtr) & (XFER_SIZE - 1)) << std::endl;
            throw PBException("pack: padding is not correct");
        }

#if 0
        for(uint32_t i=0;i<x;i+=4)
        {
            *(uint32_t*)(startPtr+i) = i/4;
        }
#endif
#ifdef VERBOSE
        for(uint32_t i=0;i<x;i+=8)
        {
            if ((i % 64) == 0) std::cout << "Pack() bytes@" << std::hex<< std::setw(16) << std::setfill('0') << (uint64_t)(startPtr + i) << ": ";
            std::cout << " " << std::setw(16) << std::setfill('0') << std::hex << *(uint64_t*)&startPtr[i] ;
            if ((i % 64) == 56) std::cout << std::endl;
        }
        std::cout << std::dec << std::endl;
#endif
          return (streamBuffer - startPtr);
      }

      template<class T>

      uint32_t SequelMovieFrame<T>::PackHeader(uint8_t* streamBuffer) const
      {
          return PackHeaderStatic(streamBuffer,24, index,timestamp,cameraConfiguration);
      }
      template<class T>
      uint32_t SequelMovieFrame<T>::PackHeaderStatic(uint8_t* streamBuffer, size_t headerSize, uint64_t frameindex, uint64_t timestampIn, uint32_t cameraConfigurationIn)
      {
          if (headerSize < 24) throw PBException("header buffer size too small");
          uint8_t* startPtr = streamBuffer;

          // pack into stream format
          uint32_t magicWord = 0x01000001;
          memcpy(streamBuffer, &magicWord, 4);
          streamBuffer += 4;

          memcpy(streamBuffer, &frameindex, 8);
          streamBuffer += 8;
          memcpy(streamBuffer, &timestampIn, 8);
          streamBuffer += 8;
          memcpy(streamBuffer, &cameraConfigurationIn, 4);
          streamBuffer += 4;
          return static_cast<uint32_t>(streamBuffer - startPtr);
      }

      template<>
      uint32_t SequelMovieFrame<double>::PackLine(int32_t /*row*/, uint8_t* /*streamBuffer*/,
                                                  uint32_t /*BitsPerPixel*/, FrameClass /* class*/) const NOT_IMPLEMENTED

      template<>
      uint32_t SequelMovieFrame<float>::PackLine(int32_t /*row*/, uint8_t* /*streamBuffer*/,
                                                 uint32_t /*BitsPerPixel*/, FrameClass /* class*/) const NOT_IMPLEMENTED

      template<>
      uint32_t SequelMovieFrame<int8_t>::PackLine(int32_t /*row*/, uint8_t* /*streamBuffer*/,
                                           uint32_t /*BitsPerPixel*/, FrameClass /* class*/) const NOT_IMPLEMENTED

      template<>
      uint32_t SequelMovieFrame<int16_t>::PackLine(int32_t row, uint8_t* streamBuffer, uint32_t BitsPerPixel, FrameClass frameClass) const
      {
          // assume streamBuffer is 32bit aligned, so that SSE instructions work
          uint8_t* startPtr = streamBuffer;

          // pack into stream format
          uint32_t magicWord = (static_cast<uint32_t>(BitsPerPixel) << 24) | GetRowPduMagicWord(frameClass);
          memcpy(streamBuffer, &magicWord, 4);
#if 0
          if (magicWord != 0x0C000002)
          {
            std::cout << "row:" << row << " magicword:" << std::hex << magicWord << std::dec << " " << frameClass.toString() << std::endl;
          }
#endif
          streamBuffer += 4;

          if (frameClass == FrameClass::Format1C4A)
          {
              uint32_t padding = 0;
              memcpy(streamBuffer, &padding, 4);
              streamBuffer += 4;
          }
          memcpy(streamBuffer, &row, 4);
          streamBuffer += 4;

        // LCM of 8 bits and 10 bits is 40 bits
        // 40 bits is 4 pixels and 5 bytes
        // so we split 4 pixels among 5 bytes

        int16_t* ptr = &data[row * NCOL];
#if defined(__SSE2__)
        if (NCOL & 7)
        {
            throw PBException("pack: ncols must be multiple of 8");
        }

        if (BitsPerPixel == 12)
        {
            const __m128i m0 = _mm_set_epi16(0, 0xFFF, 0, 0xFFF, 0, 0xFFF, 0, 0xFFF);
            const __m128i m1 = _mm_set_epi16(0xFFF, 0, 0xFFF, 0, 0xFFF, 0, 0xFFF, 0);
            const __m128i m2 = _mm_set_epi32(0, 0xFFFFFFFF, 0, 0xFFFFFFFF);
            const __m128i m3 = _mm_set_epi32(0xFFFFFFFF, 0, 0xFFFFFFFF, 0);
            const __m128i m4 = _mm_set_epi32(0, 0, 0xFFFFFFFF, 0xFFFFFFFF);
            const __m128i m5 = _mm_set_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0, 0);
            __m128i s0, t0, r0, x0, x1;

            // unrolled and normal loop gives the same result
            for (uint32_t j = 0; j < NCOL; j += 8)
            {
                // load 8 samples into s0
                s0 = _mm_loadu_si128((__m128i*) ptr);            // s0=00070006_00050004_00030002_00010000

                // join 16-bit samples into 32-bit words
                x0 = _mm_and_si128(s0, m0);                     // x0=-----006_-----004_-----002_-----000
                x1 = _mm_and_si128(s0, m1);                     // x1=-007----_-005----_-003----_-001----
                t0 = _mm_or_si128(x0, _mm_srli_epi32(x1, 4));   // t0=--007006_--005004_--003002_--001000

                // join 32-bit words into 64-bit dwords
                x0 = _mm_and_si128(t0, m2);                     // x0=--------_--005004_--------_--001000
                x1 = _mm_and_si128(t0, m3);                     // x1=--007006_--------_--003002_--------
                t0 = _mm_or_si128(x0, _mm_srli_epi64(x1, 8));   // t0=----0070_06005004_----0030_02001000

                // join 64-bit dwords
                x0 = _mm_and_si128(t0, m4);                     // x0=--------_--------_----0030_02001000
                x1 = _mm_and_si128(t0, m5);                     // x1=----0070_06005004_--------_--------
                r0 = _mm_or_si128(x0, _mm_srli_si128(x1, 2));   // r0=00000000_00700600_50040030_02001000

                // and store result
                _mm_storeu_si128((__m128i*) streamBuffer, r0);

                streamBuffer += 12;
                ptr += 8;
            }
        }
        else if (BitsPerPixel == 10)
        {
            const __m128i m0 = _mm_set_epi16(0, 0x3FF, 0, 0x3FF, 0, 0x3FF, 0, 0x3FF);
            const __m128i m1 = _mm_set_epi16(0x3FF, 0, 0x3FF, 0, 0x3FF, 0, 0x3FF, 0);
            const __m128i m2 = _mm_set_epi32(0, 0xFFFFFFFF, 0, 0xFFFFFFFF);
            const __m128i m3 = _mm_set_epi32(0xFFFFFFFF, 0, 0xFFFFFFFF, 0);
            const __m128i m4 = _mm_set_epi32(0, 0, 0xFFFFFFFF, 0xFFFFFFFF);
            const __m128i m5 = _mm_set_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0, 0);
            __m128i s0, t0, r0, x0, x1;

            // unrolled and normal loop gives the same result
            for (uint32_t j = 0; j < NCOL; j += 8)
            {
                // load 8 samples into s0
                s0 = _mm_loadu_si128((__m128i*) ptr);            // s0=00070006_00050004_00030002_00010000

                // join 16-bit samples into 32-bit words
                x0 = _mm_and_si128(s0, m0);                     // x0=00000006_00000004_00000002_00000000
                x1 = _mm_and_si128(s0, m1);                     // x1=00070000_00050000_00030000_00010000
                t0 = _mm_or_si128(x0, _mm_srli_epi32(x1, 6));   // t0=00001c06_00001404_00000c02_00000400

                // join 32-bit words into 64-bit dwords
                x0 = _mm_and_si128(t0, m2);                     // x0=00000000_00001404_00000000_00000400
                x1 = _mm_and_si128(t0, m3);                     // x1=00001c06_00000000_00000c02_00000000
                t0 = _mm_or_si128(x0, _mm_srli_epi64(x1, 12));  // t0=00000001_c0601404_00000000_c0200400

                // join 64-bit dwords
                x0 = _mm_and_si128(t0, m4);                     // x0=00000000_00000000_00000000_c0200400
                x1 = _mm_and_si128(t0, m5);                     // x1=00000001_c0601404_00000000_00000000
                r0 = _mm_or_si128(x0, _mm_srli_si128(x1, 3));   // r0=00000000_000001c0_60140400_c0200400

                // and store result
                _mm_storeu_si128((__m128i*) streamBuffer, r0);

                streamBuffer += 10;
                ptr += 8;
            }
        }
#elif defined(__SSE2__)
#error
        if (NCOL & 7)
        {
            throw PBException("pack: ncols must be multiple of 8");
        }
        const __m128i mask0 = _mm_set_epi16(0,0,0,0,0,0,0,0x3FF);
        const __m128i mask1 = _mm_set_epi16(0,0,0,0,0,0,0x3FF,0);
        const __m128i mask2 = _mm_set_epi16(0,0,0,0,0,0x3FF,0,0);
        const __m128i mask3 = _mm_set_epi16(0,0,0,0,0x3FF,0,0,0);
        const __m128i mask4 = _mm_set_epi16(0,0,0,0x3FF,0,0,0,0);
        const __m128i mask5 = _mm_set_epi16(0,0,0x3FF,0,0,0,0,0);
        const __m128i mask6 = _mm_set_epi16(0,0x3FF,0,0,0,0,0,0);
        const __m128i mask7 = _mm_set_epi16(0x3FF,0,0,0,0,0,0,0);

        for(uint32_t j=0;j<NCOL;j+=16)
        {
            __m128i s = _mm_load_si128((__m128i*)ptr); // load 8 16 bit values
            __m128i s2 = _mm_load_si128((__m128i*)(ptr+8)); // load 8 16 bit values

            __m128i a = _mm_and_si128(s,mask0);
            a = _mm_or_si128( a, _mm_srli_epi64 (_mm_and_si128(s, mask1),6));
            a = _mm_or_si128( a, _mm_srli_epi64 (_mm_and_si128(s, mask2),12));
            a = _mm_or_si128( a, _mm_srli_epi64 (_mm_and_si128(s, mask3),18));
            a = _mm_or_si128( a, _mm_srli_si128 (_mm_and_si128(s, mask4),24/8)); // special shift 24 bits to the right, staddling the middle. luckily use just on 128 byte shift (24/8)
            a = _mm_or_si128( a, _mm_srli_si128 (_mm_srli_epi64 (_mm_and_si128(s, mask5),6),24/8)); // special. shift net 30 bits. first shift 6 bits, then 3 bytes.
            a = _mm_or_si128( a, _mm_srli_si128 (_mm_srli_epi64 (_mm_and_si128(s, mask6),4),32/8)); // special. shift net 36 bits. first shift 4 bits, then 4 bytes (32 bits).
            a = _mm_or_si128( a, _mm_srli_epi64 (_mm_and_si128(s, mask7),42));

            _mm_storeu_si128((__m128i*)streamBuffer, a);

            __m128i a2 = _mm_and_si128(s2,mask0);
            a2 = _mm_or_si128( a2, _mm_srli_epi64 (_mm_and_si128(s2, mask1),6));
            a2 = _mm_or_si128( a2, _mm_srli_epi64 (_mm_and_si128(s2, mask2),12));
            a2 = _mm_or_si128( a2, _mm_srli_epi64 (_mm_and_si128(s2, mask3),18));
            a2 = _mm_or_si128( a2, _mm_srli_si128 (_mm_and_si128(s2, mask4),24/8)); // special shift 24 bits to the right, staddling the middle. luckily use just on 128 byte shift (24/8)
            a2 = _mm_or_si128( a2, _mm_srli_si128 (_mm_srli_epi64 (_mm_and_si128(s2, mask5),6),24/8)); // special. shift net 30 bits. first shift 6 bits, then 3 bytes.
            a2 = _mm_or_si128( a2, _mm_srli_si128 (_mm_srli_epi64 (_mm_and_si128(s2, mask6),4),32/8)); // special. shift net 36 bits. first shift 4 bits, then 4 bytes (32 bits).
            a2 = _mm_or_si128( a2, _mm_srli_epi64 (_mm_and_si128(s2, mask7),42));

            _mm_storeu_si128((__m128i*)(streamBuffer+10), a2);

            streamBuffer += 20 ;
            ptr += 16 ;
        }

#else
        if (NCOL & 3)
        {
            throw PBException("pack: ncols must be multiple of 4");
        }
        if (BitsPerPixel == 10)
        {
            for (uint32_t j = 0; j < NCOL; j += 4 * 4)
            {
                uint64_t* dst;
                uint64_t src[4][4];

                // __m128i s01 = _mm_set_epi64(ptr[0], ptr[1]);
                // __m128i s23 = _mm_set_epi64(ptr[2], ptr[3]);
                // ---- or ----
                // __m128i s0123 = _mm_load_si128(ptr[0])
                // __m128i s01   = _?????_(s0123) // some instruction to extract s01 from s0123
                // __m128i s23   = _?????_(s0123) // some instruction to extract s23

                src[0][0] = ptr[0] & 0x3ff;
                src[0][1] = ptr[1] & 0x3ff;
                src[0][2] = ptr[2] & 0x3ff;
                src[0][3] = ptr[3] & 0x3ff;

                src[1][0] = ptr[4] & 0x3ff;
                src[1][1] = ptr[5] & 0x3ff;
                src[1][2] = ptr[6] & 0x3ff;
                src[1][3] = ptr[7] & 0x3ff;

                src[2][0] = ptr[8] & 0x3ff;
                src[2][1] = ptr[9] & 0x3ff;
                src[2][2] = ptr[10] & 0x3ff;
                src[2][3] = ptr[11] & 0x3ff;

                src[3][0] = ptr[12] & 0x3ff;
                src[3][1] = ptr[13] & 0x3ff;
                src[3][2] = ptr[14] & 0x3ff;
                src[3][3] = ptr[15] & 0x3ff;

                // looks like _mm_maskmoveu_si128 can store result efficiently
                dst = (uint64_t*) streamBuffer;
                dst[0] = src[0][0] | (src[0][1] << 10) | (src[0][2] << 20) | (src[0][3] << 30);

                dst = (uint64_t * )(streamBuffer + 5);
                dst[0] = src[1][0] | (src[1][1] << 10) | (src[1][2] << 20) | (src[1][3] << 30);

                dst = (uint64_t * )(streamBuffer + 10);
                dst[0] = src[2][0] | (src[2][1] << 10) | (src[2][2] << 20) | (src[2][3] << 30);

                dst = (uint64_t * )(streamBuffer + 15);
                dst[0] = src[3][0] | (src[3][1] << 10) | (src[3][2] << 20) | (src[3][3] << 30);

                streamBuffer += 5 * 4;
                ptr += 4 * 4;
            }
        } else if (BitsPerPixel == 12)
        {
            throw PBException("Bit depth not supported:" + std::to_string(BitsPerPixel));
        }
        else
        {
            throw PBException("Bit depth not supported:" + std::to_string(BitsPerPixel));
        }
#endif
        // pad to 32-bit alignment
        while (((uint64_t) streamBuffer) & 3)
        {
            *streamBuffer++ = 0;
        }
        return static_cast<uint32_t>(streamBuffer - startPtr);
    }

    template<class T>
    int SequelMovieFrame<T>::NumberOfSetBits(int i)
    {
        i = i - ((i >> 1) & 0x55555555);
        i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
        return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
    }

    template<>
    int SequelMovieFrame<float>::CountBitMismatches(float a, float b)
    {
        return NumberOfSetBits((int) a ^ (int) b);
    }

    template<>
    int SequelMovieFrame<double>::CountBitMismatches(double a, double b)
    {
        return NumberOfSetBits((int) a ^ (int) b);
    }

    template<>
    int SequelMovieFrame<int16_t>::CountBitMismatches(int16_t a, int16_t b)
    {
        return NumberOfSetBits((int)(static_cast<uint16_t>(a) ^ static_cast<uint16_t>(b)));
    }

    template<>
    int SequelMovieFrame<int8_t>::CountBitMismatches(int8_t a, int8_t b)
    {
        return NumberOfSetBits((int)(static_cast<int8_t>(a) ^ static_cast<int8_t>(b)));
    }

// returns number of errors found
    template<class T>
    uint32_t SequelMovieFrame<T>::Compare(const SequelMovieFrame& b, double tolerance, bool dataOnly) const
    {
        int errors = 0;
        int numErrsToPrint=10;  //maybe pass this in?
        errors += Compare(this->NROW, b.NROW, "NROW");
        errors += Compare(this->NCOL, b.NCOL, "NCOL");
        if (!dataOnly)
        {
            errors += Compare(this->index, b.index, "index");
            errors += Compare(this->timestamp, b.timestamp, "timestamp");
            errors += Compare(this->cameraConfiguration, b.cameraConfiguration, "cameraConfiguration");
        }
        
        if (errors == 0)
        {
#if 0
            T mask = (1<<BitsPerPixel) - 1;
            for(int i=0;i<NROW*NCOL;i++)
            {
                this->data[i] &= mask;
            }
#endif
            int maxDiff = 0;
            int minDiff = 0xffff;
            if (memcmp(this->data, b.data, NROW * NCOL * sizeof(T)))
            {
                int ptr = 0;
                for (uint32_t r = 0; r < NROW; r++)
                {
                    for (uint32_t c = 0; c < NCOL; c++)
                    {
                        bool mismatch;
                        if (tolerance == 0)
                        {
                            mismatch = this->data[ptr] != b.data[ptr];
                        }
                        else
                        {
                            mismatch = std::abs(this->data[ptr] - b.data[ptr]) > tolerance;
                        }
                        if (mismatch)
                        {
                            int bitMismatches = CountBitMismatches(this->data[ptr], b.data[ptr]);
                            errors  += 1;
                            int diff = static_cast<int>(this->data[ptr] - b.data[ptr]);
                            maxDiff  = (diff > maxDiff) ? diff : maxDiff;
                            minDiff  = (diff < minDiff) ? diff : minDiff;
                            if (errors <= numErrsToPrint)
                            {
                                std::cout << "data mismatch in frame " << this->index;
                                std::cout << " pixel(\t" << r << "\t," << c << ") :\t";
                                std::cout <<"0x"<<std::hex << this->data[ptr] << "(" << std::dec << this->data[ptr] << ")" ;
                                std::cout <<" 0x"<<std::hex << b.data[ptr] << "(" << std::dec << b.data[ptr] << ")" ;
                                std::cout << " pixel_diff:" << this->data[ptr] -b.data[ptr];
                                std::cout << std::dec << "\t" << bitMismatches << " flipped bits" << std::endl;
                            }
                        }
                        ptr++;
                    }
                }
            }
            if (errors > numErrsToPrint)
            {
                std::cout << "  * suppressed " << (errors - numErrsToPrint) << " additional pixel mismatch errors" << std::endl;
            }
            if (errors) 
            {
                std::cout << " minimum_diff was:" << std::dec << minDiff << 
                    " maximum_diff was:" << std::dec << maxDiff << std::endl; 
            }
        }
        return errors;
    }

    template<class T>
    uint16_t SequelMovieFrame<T>::GetPackedHeaderPduSize(uint32_t /*rows*/, uint32_t /*cols*/)
    {
        uint16_t headerSize =
            4 + // magic word
            8 + // index
            8 + // timestamp
            4 + // configuration
#if 0
            8 + // padding
#endif
            0;
        return headerSize;
    }

    template<class T>
    uint32_t SequelMovieFrame<T>::GetPackedRowPduSize(uint32_t /*rows*/, uint32_t cols, uint32_t bitsPerPixel, FrameClass frameClass)
    {
        uint32_t pixelbytes = (cols * bitsPerPixel + 7) / 8; // rounded up
        uint32_t rowSize =
            4 + //magic word
                ((frameClass == FrameClass::Format1C4A)?8:4) + // include padding bits /* fixme*/
            ((pixelbytes + 3) / 4 * 4); // round up to 4-byte
        return rowSize;
    }

    template<class T>
    uint32_t SequelMovieFrame<T>::GetPackedFrameSize(uint32_t rows, uint32_t cols, uint32_t bitsPerPixel, FrameClass frameClass)
    {
        uint32_t headerSize = GetPackedHeaderPduSize(rows, cols);
        uint32_t rowSize = GetPackedRowPduSize(rows, cols, bitsPerPixel, frameClass);
        uint32_t headerSizeRoundUp = (headerSize + (XFER_SIZE - 1)) / XFER_SIZE * XFER_SIZE;
        uint32_t rowSizeRoundUp = (rowSize + (XFER_SIZE - 1)) / XFER_SIZE * XFER_SIZE;
        uint32_t size = (uint32_t)(rows * rowSizeRoundUp + headerSizeRoundUp);
        return size;
    }

    template<class T>
    void SequelMovieFrame<T>::CreateRandomPattern(uint32_t BitsPerPixel)
    {
        static std::default_random_engine generator;
        int32_t mask = (1 << BitsPerPixel) - 1;
        static std::uniform_int_distribution <int32_t> distribution(0, mask);
        static int j = 0;
        int i = 0;
        for (uint32_t row = 0; row < NROW; row++)
        {
            for (uint32_t col = 0; col < NCOL; col++)
            {
                int32_t dice_roll = 0;
                if (((row % 3) == 1 && (col % 3) == 1) ||
                    ((row % 3) == 2 && (col % 3) == 2))
                {
                    dice_roll = distribution(generator);
                }
                data[i++] = static_cast<T>((dice_roll + j) & mask);
            }
        }
        j++;
    }

      template<class T>
      void SequelMovieFrame<T>::CreateRandomPatternForSequel(uint32_t BitsPerPixel)
      {
          static std::default_random_engine generator;
          int32_t mask = (1 << BitsPerPixel) - 1;
          static std::uniform_int_distribution <int32_t> distribution(0, mask);
          static int j = 0;
          int i = 0;
          for (uint32_t row = 0; row < NROW; row++)
          {
              for (uint32_t col = 0; col < NCOL; col++)
              {
                  int32_t dice_roll = 0;
                  if (((row % 4) == 1 && ((col % 32) == 0 || (col % 32) == 1)))
                  {
                      dice_roll = distribution(generator);
                  }
                  data[i++] = static_cast<T>((dice_roll + j) & mask);
              }
          }
          j++;
      }
#if 0
    template<typename Y>
    void SequelMovieFrame<T>::Add(const SequelMovieFrame<Y>& frame)
    {
        uint64_t n = NROW * NCOL;
        for (uint64_t i = 0; i < n; i++)
        {
            data[i] += frame.data[i];
        }
    }

    template<typename Y>
    void SequelMovieFrame<T>::Add(Y offset)
    {
        uint64_t n = NROW * NCOL;
        for (uint64_t i = 0; i < n; i++)
        {
            data[i] += offset;
        }
    }

    template<typename Y>
    void SequelMovieFrame<T>::Subtract(const SequelMovieFrame<Y>& frame)
    {
        uint64_t n = NROW * NCOL;
        for (uint64_t i = 0; i < n; i++)
        {
            data[i] -= frame.data[i];
        }
    }

    template<typename Y>
    void SequelMovieFrame<T>::AddSquare(const SequelMovieFrame<Y>& frame)
    {
        uint64_t n = NROW * NCOL;
        for (uint64_t i = 0; i < n; i++)
        {
            const Y value = frame.data[i];
            data[i] += value * value;
        }
    }
#endif

template <typename T>
struct TypeName
{
    static const char* Get()
    {
        return typeid(T).name();
    }
};
#define ENABLE_TYPENAME(A) template<> struct TypeName<A> { static const char *Get() { return #A; }};
ENABLE_TYPENAME(float)
ENABLE_TYPENAME(double)
ENABLE_TYPENAME(int16_t)
ENABLE_TYPENAME(int8_t)

    template<typename T>
    static void SanityCheckBitsPerPixel(uint32_t bitsPerPixel)
    {
        if ((1LL<<bitsPerPixel) -1 > std::numeric_limits<T>::max())
        {
            PBLOG_ERROR
                << "((1LL<<bitsPerPixel) -1:" <<  ((1LL<<bitsPerPixel) -1)
                << " std::numeric_limits<T>::max():" << (double)std::numeric_limits<T>::max();
            throw PBException("bitPerPixel " + std::to_string(bitsPerPixel) +
                              " is too large for " + TypeName<T>::Get());
        }
    }

    /// sets the pixels to pattern1
    template<class T>
    void SequelMovieFrame<T>::SetPattern1(uint32_t bitsPerPixel)
    {
        SanityCheckBitsPerPixel<T>(bitsPerPixel);
        T j=0;
        T maxPixelValue = static_cast<T>((1ULL<<bitsPerPixel) -1);
        size_t i=0;
        for (uint32_t row = 0; row < NROW; row++)
        {
            for (uint32_t col = 0; col < NCOL; col++)
            {
                data[i++] = j++;
                if (j > maxPixelValue) j =0;
            }
        }
    }


    /// throws an exception if the frame doesn't match pattern1
    template<class T>
    void SequelMovieFrame<T>::ValidatePattern1(uint32_t bitsPerPixel, std::ostream* of)
    {
        SanityCheckBitsPerPixel<T>(bitsPerPixel);
        T j=0;
        T maxPixelValue = static_cast<T>((1<<bitsPerPixel) - 1);
        uint32_t mismatches = 0;
        size_t i=0;
        const uint32_t MAX_MISMATCHES = 20;
        for (uint32_t row = 0; row < NROW; row++)
        {
            for (uint32_t col = 0; col < NCOL; col++)
            {
                if (data[i] != j)
                {
                    if (of != nullptr)
                    {
                        if (mismatches == 0 ) *of << "ValidatePattern1: failed!\n"; // print only on first mismatch
                        if (mismatches < MAX_MISMATCHES && of != nullptr)
                        {
                            *of << "ValidatePattern1: [" << row << "," << col << "] (i=" << i << ") " <<
                            data[i] << "!=" << j << "\n";
                        }
                        else if (mismatches == MAX_MISMATCHES)
                        {
                            *of << "ValidatePattern1: and more mismatches (suppressed)";
                        }
                    }
                    mismatches++;
                }
                i++;
                j++;
                if (j > maxPixelValue) j =0;
            }
        }
        if (mismatches != 0)
        {
            if (of) *of << "ValidatePattern1: frame mismatches pattern by " << mismatches << " mismatches";
            throw PBException("frame mismatches pattern by " + std::to_string(mismatches) + " mismatches");
        }
    }

    template<class T>
    void SequelMovieFrame<T>::ValidatePattern(uint32_t frame, std::function<T(uint32_t,uint32_t,uint32_t)> pattern) const
    {
        uint32_t mismatches = 0;
        for (uint32_t row = 0; row < NROW; row++)
        {
            for (uint32_t col = 0; col < NCOL; col++)
            {
                T expectedPixel = pattern(row, col, frame);
                T was = GetPixel(row, col);
                if (expectedPixel != was)
                {
                    if (mismatches < 10)
                    {
                        PBLOG_ERROR << "First mismatch at " << row << "," << col << "."
                                << "Expected:" << expectedPixel << " was:" << was;
                    }
                    else if (mismatches == 10)
                    {
                        PBLOG_ERROR << " Additional mismatches suppressed";
                    }
                    mismatches ++;
                }
            }
        }
        if (mismatches > 0)
        {
            throw PBException("Mismatches:" + std::to_string(mismatches));
        }
    }


    template<class T>
    void SequelMovieFrame<T>::Scale(const double factor)
    {
        uint64_t n = NROW * NCOL;
        for (uint64_t i = 0; i < n; i++)
        {
            data[i] = static_cast<T>(data[i] * factor);
        }
    }

    template<class T>
    void SequelMovieFrame<T>::Scale(const double factor, T min, T max)
    {
        uint64_t n = NROW * NCOL;

        for (uint64_t i = 0; i < n; i++)
        {
            double x = data[i] * factor;
            if (x < min) x = min;
            else if (x > max) x = max;
            data[i] = static_cast<T>(x);
        }
    }

    template<class T>
    void SequelMovieFrame<T>::Square()
    {
        uint64_t n = NROW * NCOL;
        for (uint64_t i = 0; i < n; i++)
        {
            data[i] = static_cast<T>(data[i] * data[i]);
        }
    }

    template<class T>
    void SequelMovieFrame<T>::Clamp(const T clampVal)
    {
        uint64_t n = NROW * NCOL;
        if (clampVal < 0)
        {
            for (uint64_t i = 0; i < n; i++)
            {
                if (data[i] < clampVal)
                    data[i] = clampVal;
            }
        }
        else if (clampVal > 0)
        {
            for (uint64_t i = 0; i < n; i++)
            {
                if (data[i] > clampVal)
                    data[i] = clampVal;
            }
        }
        else
        {
            for (uint64_t i = 0; i < n; i++)
            {
                data[i] = static_cast<T>(0);
            }
        }
    }

#if 0
// this performs the last stage of a linear regression
// http://stackoverflow.com/questions/5083465/fast-efficient-least-squares-fit-algorithm-in-c
    template<typename Y>
    void SequelMovieFrame<T>::LinearFitYIntercept(
        const SequelMovieFrame<Y>& sumy,
        const SequelMovieFrame<Y>& sumx2,
        const SequelMovieFrame<Y>& sumx,
        const SequelMovieFrame<Y>& sumxy,
        double recip)
    {
        uint64_t n = NROW * NCOL;
        for (uint64_t i = 0; i < n; i++)
        {
            data[i] = (T) ((sumy.data[i] * sumx2.data[i] - sumx.data[i] * sumxy.data[i]) * recip);
        }
    }

    template<typename Y>
    void SequelMovieFrame<T>::LinearFitYIntercept(
        const SequelMovieFrame<Y>& sumy,
        const Y sumx2,
        const Y sumx,
        const SequelMovieFrame<Y>& sumxy,
        double recip)
    {
        uint64_t n = NROW * NCOL;
        for (uint64_t i = 0; i < n; i++)
        {
            data[i] = (T) ((sumy.data[i] * sumx2 - sumx * sumxy.data[i]) * recip);
        }
    }

    template<typename Y>
    void SequelMovieFrame<T>::LinearFitSlope(
        const int nPoints,
        const SequelMovieFrame<Y>& sumxy,
        const Y sumx,
        const SequelMovieFrame<Y>& sumy,
        double recip)
    {
        uint64_t n = NROW * NCOL;
        for (uint64_t i = 0; i < n; i++)
        {
            data[i] = (T) ((nPoints * sumxy.data[i] - sumx * sumy.data[i]) * recip);
        }
    }
#endif

    template<class T>
    size_t SequelMovieFrame<T>::DataSize() const
    {
        return NROW * NCOL * sizeof(T);
    }


    template<class T>
    int SequelMovieFrame<T>::Compare(uint64_t a, uint64_t b, const char* field)
    {
        if (a != b)
        {
            std::cout << PacBio::Text::String::Format("%s mismatch 0x%x != 0x%x", field, a, b) << std::endl;
            return 1;
        }
        else
        {
            return 0;
        }
    }

template<typename T>
void SequelMovieFrame<T>::GetMinMax(T* min, T* max) const
{
    if (min == nullptr || max == nullptr) throw PBException("null ptrs");
    uint64_t n = NROW * NCOL;
    if (n == 0) throw PBException("cant find min max of 0 size frame");
    *min = data[0];
    *max = data[0];
    for (uint64_t i = 1; i < n; i++)
    {
        if (data[i] < *min) { *min = data[i];}
        if (data[i] > *max) { *max = data[i];}
    }
}

template<typename T>
bool SequelMovieFrame<T>::FillMissingRows = true;

template SequelMovieFrame<double>::SequelMovieFrame(const SequelMovieFrame<int16_t>&);
template SequelMovieFrame<float>::SequelMovieFrame(const SequelMovieFrame<int16_t>&);
template SequelMovieFrame<int16_t>::SequelMovieFrame(const SequelMovieFrame<float>&);
template SequelMovieFrame<float>::SequelMovieFrame(const SequelMovieFrame<double>&);
template SequelMovieFrame<double>::SequelMovieFrame(const SequelMovieFrame<float>&);

template SequelMovieFrame<double>& SequelMovieFrame<double>::operator=(const SequelMovieFrame<float>&);
template SequelMovieFrame<float>& SequelMovieFrame<float>::operator=(const SequelMovieFrame<double>&);
template SequelMovieFrame<double>& SequelMovieFrame<double>::operator=(const SequelMovieFrame<int16_t>&);
template SequelMovieFrame<float>& SequelMovieFrame<float>::operator=(const SequelMovieFrame<int16_t>&);
template SequelMovieFrame<int16_t>& SequelMovieFrame<int16_t>::operator=(const SequelMovieFrame<double>&);

template void SequelMovieFrame<int16_t>::TiledReplicate(const SequelMovieFrame<int16_t>& frame);
template void SequelMovieFrame<int16_t>::Insert(uint32_t rowOffset, uint32_t colOffset, const SequelMovieFrame<int16_t>& frame);

template class SequelMovieFrame<double>;
template class SequelMovieFrame<float>;
template class SequelMovieFrame<int16_t>;
template class SequelMovieFrame<int8_t>;

}}
