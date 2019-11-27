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
/// \brief LiveViewFrame class wrapper

#ifndef PA_ACQUISITION_LIVEVIEWFRAME_H
#define PA_ACQUISITION_LIVEVIEWFRAME_H

#include <stdint.h>

namespace PacBio
{
 namespace Primary
 {
#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4200 )
#pragma pack(push,1)
#endif
struct ZmwPixelPair
{
    uint16_t red;
    uint16_t green;
};
struct LiveViewFrame
{
    uint32_t columnMin; ///< defines left-most column of ROI viewport � default 0 (aka start column)
    uint32_t rowMin;    ///< defines top-most row of ROI viewport � default 0 (aka start row)
    uint32_t columns;   ///< defines total width in pixels of ROI viewport (aka columns)
    uint32_t rows  ;    ///< defines total height in pixels of ROI viewport (aka rows)
    uint32_t Stride;    ///< defines total number of pixels striding between pixel pairs (1 ZMW). a value of 2 means all pixels. a value of 4 means read 2/skip 2. 8 means read2/skip 6.
    uint32_t NumPixels; ///< Number of pixels in the data stream

    union {
        ZmwPixelPair zmw[0]; // variable length array of pixel pairs
        int16_t raw[0];
    } pixels;

private:
    // this silliness is to quash warnings about pixels[0]. making the constructors private makes the errors go away. I never actual construct the struct
    LiveViewFrame( const LiveViewFrame &  );            // undefined
    LiveViewFrame& operator=( const LiveViewFrame &  ); // undefined
};
#ifdef WIN32
#pragma pack(pop)
#pragma warning( pop )
#endif
 }
}


#endif //PA_ACQUISITION_LIVEVIEWFRAME_H

