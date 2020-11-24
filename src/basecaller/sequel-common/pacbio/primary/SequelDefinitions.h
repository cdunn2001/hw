#ifndef _SEQUEL_DEFINITIONS_H_
#define _SEQUEL_DEFINITIONS_H_

// Copyright (c) 2016, Pacific Biosciences of California, Inc.
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
/// \brief  Global constants of the Sequel project. Other global parameters (nonconstant)
///         which used to be in this header may be found in PrimaryConfig.h


#include <stdint.h>
#include <cmath>

#ifndef ATT_UNUSED
#  ifdef __INTEL_COMPILER
// this disables error message #593, "variable not used"
#    define ATT_UNUSED [[gnu::unused]]
#  else
#    define ATT_UNUSED __attribute__((unused))
#  endif
#endif

namespace PacBio {
namespace Primary  {
    const uint32_t numHeaderTilesPerChunk ATT_UNUSED = 1;    ///< Every chunk from the FPGA is prefixed by a header tile.

    // chip parameters that should be refactored out of here
    namespace Sequel // todo. rename Tech 2C2C
    {
        const uint32_t maxPixelRows ATT_UNUSED = 1144; ///< this is the number of pixels in the Sequel "x" dimension
        const uint32_t maxPixelCols ATT_UNUSED = 2048; ///< this is the number of pixels in the Sequel "y" dimension
        const uint32_t numPixelRowsPerZmw ATT_UNUSED = 1;
        const uint32_t numPixelColsPerZmw ATT_UNUSED = 2;
        const uint32_t maxNumDataTilesPerChunk ATT_UNUSED =
                maxPixelRows * maxPixelCols / PacBio::Primary::Tile::NumPixels; // typical ~73K
        const double   defaultFrameRate  ATT_UNUSED      { 100.0};
        const double   defaultLineRate   ATT_UNUSED      { 114784.0 };
    }
    namespace SequelLayout
    {
        const uint32_t maxPixelRows ATT_UNUSED = 1144; ///< this is the number of pixels in the Sequel "x" dimension
        const uint32_t maxPixelCols ATT_UNUSED = 2048; ///< this is the number of pixels in the Sequel "y" dimension
        const uint32_t maxNumDataTilesPerChunk ATT_UNUSED =
                maxPixelRows * maxPixelCols / PacBio::Primary::Tile::NumPixels; // typical ~73K
    }
    namespace Spider // todo. rename Tech 1C4A
    {
        const uint32_t maxPixelRows ATT_UNUSED = 2756; ///< this is the number of pixels in the PacBio "x" dimension
        const uint32_t maxPixelCols ATT_UNUSED = 2912; ///< this is the number of pixels in the PacBio "y" dimension
        const uint32_t numPixelRowsPerZmw ATT_UNUSED = 1;
        const uint32_t numPixelColsPerZmw ATT_UNUSED = 1;
        const uint32_t maxNumDataTilesPerChunk ATT_UNUSED =
                maxPixelRows * maxPixelCols / PacBio::Primary::Tile::NumPixels; // typical ~251K
        const double   defaultFrameRate  ATT_UNUSED      { 100.0};
        const double   defaultLineRate   ATT_UNUSED      { (maxPixelRows+1) * defaultFrameRate };
    }
    namespace SpiderLayout
    {
        const uint32_t maxPixelRows ATT_UNUSED = 2756; ///< this is the number of pixels in the PacBio "x" dimension
        const uint32_t maxPixelCols ATT_UNUSED = 2912; ///< this is the number of pixels in the PacBio "y" dimension
        const uint32_t maxNumDataTilesPerChunk ATT_UNUSED =
                maxPixelRows * maxPixelCols / PacBio::Primary::Tile::NumPixels; // typical ~251K
    }
    // acquisition parameters
    const double   auroraBitsPerSecond  ATT_UNUSED   { 6.25e9 };
    const size_t   bytesPerPixel   ATT_UNUSED        { PIXEL_SIZE };
    const size_t   framesPerTile   ATT_UNUSED        { PacBio::Primary::Tile::NumFrames };
    const size_t   zmwsPerTranche  ATT_UNUSED        { 16 };
    const size_t   pixelsPerSIMD   ATT_UNUSED        { 32 };

    // t2b parameters
    const int      maxNumMicThreads  ATT_UNUSED      { 60*4 }; // cores * threads/core
    const int      trancheQueueSize  ATT_UNUSED      { maxNumMicThreads*16}; //16 is a fudge factor

    // other default values
    const double   defaultExposureSec  ATT_UNUSED    { 1.0/ Sequel::defaultFrameRate };
    const uint32_t defaultExposureUsec ATT_UNUSED    { static_cast<uint32_t>(std::lrint(1000000/ Sequel::defaultFrameRate)) };
    const double   defaultPhotoelectronSensitivity ATT_UNUSED { 1.0 } ;// fixme. Units are photoelectroncs per sensor ADC count.

    const float    defaultRefDwsSnr  ATT_UNUSED      { 11.0f }; ///< SNR of Dye Weighted Sum
    const float    defaultMinSnr     ATT_UNUSED      { 4.0f};   ///< SNR of basecall

// Used internally to PA
SMART_ENUM(CalType    ,none,dark,gain,spectral,loading);

// Used externally between pa-ws and ICS
SMART_ENUM(PrepareMode,none,darkframe,gain,spectral,loading,darkthenacquire,acquire);

#if 0
  const uint32_t chunksPerSuperchunk = 32;
  const size_t   numAllocatedTilesPerSuperchunk = ((numDataTilesPerChunk + numHeaderTilesPerChunk) *
          chunksPerSuperchunk);
  const size_t   framesPerTranche    { framesPerTile * tilesPerTranche };
  const size_t   framesPerBlock      { framesPerTranche / blocksPerTranche };

  extern const size_t&  tilesPerTranche;
  extern const size_t&  framesPerSuperchunk;
#endif

}}

#endif // _SEQUEL_DEFINITIONS_H_
