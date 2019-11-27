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
/// \brief  implementation of Tranche object methods


#include <pacbio/primary/Tranche.h>
#include <pacbio/smrtdata/Basecall.h>

using namespace PacBio::Primary;

using std::ostringstream;

// TODO - FIXME: Restore these asserts.
// Temporarily knocking them out while we get the build machinery set up.
//static_assert(ZmwResultBuffer<uint8_t, uint32_t>::SizePerTrancheInBytes ==
//              sizeof(ZmwResultBuffer<uint8_t, uint32_t>),
//              "ZmwResultBuffer size or offset error");

//static_assert(ZmwResultBuffer<uint32_t, BasicTraceMetrics>::SizePerTrancheInBytes -
//              sizeof(ZmwResultBuffer<uint32_t, BasicTraceMetrics>) < sizeof(uint32_t),
//              "ZmwResultBuffer size or offset error");


namespace PacBio {
namespace Primary {

    std::ostream& operator<<(std::ostream& s, const PacBio::Primary::Tranche& t)
    {
        s << "Tranche type:" << (int)t.Type();
        if(&t.Title() != nullptr)
        {
            s << " FrameIndex:" << t.FrameIndex();
            s << " ZmwLaneIndex:" << t.ZmwLaneIndex();
            s << " ZmwNumber:" << t.ZmwNumber();
            s << " ZmwIndex:" << t.ZmwIndex();
            s << " SuperChunkIndex:" << t.SuperChunkIndex();
            s << " StopStatus:" << t.StopStatus().toString();
            s << " Frames:" << t.FrameCount();
            s << " UnitCells: " << t.ZmwCount() << std::endl;
        }
        else
        {
            s << " Title:nullptr";
        }
        for (auto& call : t.Calls())
        {
            s << " Call: zmw:" << std::hex << call->ZmwIndex() << std::dec << " #bases:" << call->NumSamples() << " #metrics:" << call->NumMetrics() << std::endl ;
        }
        return s;
    }

#if 1
     std::ostream& operator<<(std::ostream& s, const PacBio::Primary::Tranche::Pixelz* t)
     {
         s << (void*) t << ":\n   ";
         for(int i=0;i<16;i++)
         {
             s << "\t" << t->pixels.int16[i];
         }
         s << "..." << std::endl;
         s << "   ";
         for(int i=32;i<48;i++)
         {
             s << "\t" << t->pixels.int16[i];
         }
         s << "..." << std::endl;
         return s;
     }
 TrancheHelper& operator<<(TrancheHelper& s, const PacBio::Primary::Tranche::Pixelz* t)
 {
     *s.s << (void*) t << ":";
     if (s.verbosity_ <= 1)
         *s.s << "first SIMD\n";
     else
     {
         *s.s << "\n";
         *s.s << "  [#] ";
         for (int i = 0; i < 6; i++)
         {
             *s.s << "\tpix:" << i;
         }
         *s.s << "\t... ";
         for (int i = 26; i < 32; i++)
         {
             *s.s << "\tpix:" << i;
         }
         *s.s << "\n";
     }

     int offset = 0;
     for (int iframe = 0; iframe < s.verbosity_; iframe++)
     {
         *s.s << "  [" << iframe << "] ";
         for (int i = 0; i < 6; i++)
         {
             *s.s << "\t" << t->pixels.int16[i + offset];
         }
         *s.s << "\t... ";
         for (int i = 26; i < 32; i++)
         {
             *s.s << "\t" << t->pixels.int16[i + offset];
         }
         offset += Tile::NumPixels;
         *s.s << std::endl;
     }
     return s;
 }
#endif


// static
uint64_t Tranche::relativeFrameIndexOffset_ = 0;


void Tranche::CopyTraceData(void* dst,Pointer& pointer,size_t numFrames) const
{
    if (numFrames == 0) return;

    if (data)
    {
// the data pointer is contiguous, so the copy is easy
        if (tiles.size() > 0) throw PBException("misconfigurated Tranche");
        if (pointer.frame + numFrames > this->FrameCount())
            throw PBException("pointer.frame overflow" +
                              std::to_string(pointer.frame) + " " + std::to_string(this->FrameCount()) + " " +
                              std::to_string(numFrames));

        memcpy(dst, &data->pixels.simd[pointer.frame], numFrames * sizeof(Pixelz::SimdPixel));
        pointer.frame += numFrames;
    }
    else if (tiles.size() > 0)
    {
        uint8_t* dstU8 = static_cast<uint8_t*>(dst);
        int16_t* dstI16 = reinterpret_cast<int16_t*>(dst);
// if tiles are used, then each tile must be accessed individually
        while (numFrames > 0)
        {
            if (pointer.tile >= static_cast<ssize_t>(tiles.size()))
                throw PBException("pointer.tile=" + std::to_string(pointer.tile) +
                                  " bigger than tiles.size():" +
                                  std::to_string(tiles.size()));
            size_t framesThisTime = numFrames;
            if (framesThisTime > (Tile::NumFrames - pointer.frame))
            {
                framesThisTime = (Tile::NumFrames - pointer.frame);
            }

#ifdef  SPIDER_TRANCHES
            if (format_ == PixelFormat::Format1C4A_CA)
            {
                throw PBException("not supported");
            }
            else if (format_ == PixelFormat::Format1C4A_RT)
            {
                int16_t* srcI16 = reinterpret_cast<int16_t*>(tiles[pointer.tile]->data +
                        (sizeof(Pixelz::SimdPixel) * pointer.frame));
                const uint32_t halfTilePixels = Tile::NumPixels/2;
                if (subset_ == 0)
                {
// lower pixels
                    srcI16 += 0; // nop
                }
                else if (subset_ == 1)
                {
// upper pixels
                    srcI16 += halfTilePixels;
                }
                for (size_t j = 0; j < framesThisTime; j++)
                {
                    for (uint32_t i = 0; i < halfTilePixels; i++)
                    {
// duplicate the source pixels in both lower and upper slots of the SIMD word
                        dstI16[i]                  = srcI16[i];
                        dstI16[i + halfTilePixels] = srcI16[i];
                    }
                    dstI16 += Tile::NumPixels;
                    srcI16 += Tile::NumPixels;
                }
                pointer.frame += framesThisTime;
            }
            else if (format_ == PixelFormat::Format2C2A)
            {
#endif
                size_t numBytes = framesThisTime * sizeof(Pixelz::SimdPixel);
                memcpy(dstU8, tiles[pointer.tile]->data + (sizeof(Pixelz::SimdPixel) * pointer.frame), numBytes);
                pointer.frame += framesThisTime;
                dstU8 += numBytes;
#ifdef  SPIDER_TRANCHES
            }
            else
            {
                throw PBException("bad format_ specified");
            }
#endif
            if (pointer.frame > static_cast<int>(Tile::NumFrames))
                throw PBException("pointer.frame can't be bigger than Tile::NumFrames");
            if (pointer.frame == Tile::NumFrames)
            {
                pointer.frame = 0;
                pointer.tile++;
            }
            numFrames -= framesThisTime;
        }
    }
    else
    {
        throw PBException("no data, no tiles");
    }
}


}}  // PacBio::Primary
