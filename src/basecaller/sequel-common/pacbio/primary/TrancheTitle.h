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
/// \brief a class that manages deeding Tranches from one process to another process.

#pragma once

#include <pacbio/primary/Tile.h>
#include <pacbio/ipc/TitleBase.h>
#include <assert.h>
#include <pacbio/primary/ITrancheNew.h>
namespace PacBio
{
 namespace Primary
 {
class TrancheTitle : public PacBio::IPC::Title<PacBio::Primary::Tile>
{
public:

    static const size_t Size_{2*sizeof(uint64_t) + 9*sizeof(uint32_t)};

    TrancheTitle() = default;
     // disallow copying
    TrancheTitle(const TrancheTitle& src) = delete;
    TrancheTitle& operator=(const TrancheTitle& src) = delete;
    ~TrancheTitle() noexcept override = default;

     // experiment. Is this good enough?
     TrancheTitle(TrancheTitle&& src) = default;
     TrancheTitle& operator=(TrancheTitle&& src) = default;

    uint32_t Deserialize(const uint8_t* ptr, uint32_t len) override
    {
        uint32_t x = PacBio::IPC::Title<PacBio::Primary::Tile>::Deserialize(ptr,len);
        ptr += x;
        len -= x;
        if (len < Size_) throw PBException("Too small");
#if 1
        frameIndexStart_ = 0;
        memcpy(&frameIndexStart_,ptr+ 0,sizeof(uint32_t));
#else
        memcpy(&frameIndexStart_,ptr+ 0,sizeof(uint64_t));
#endif
        memcpy(&timeStampStart_ ,ptr+ 8,sizeof(uint64_t));
        memcpy(&configWord_     ,ptr+16,sizeof(uint32_t));
        memcpy(&frameCount_     ,ptr+20,sizeof(uint32_t));
        memcpy(&zmwIndex_       ,ptr+24,sizeof(uint32_t));
        memcpy(&micOffset_      ,ptr+28,sizeof(uint32_t));
        memcpy(&flags_          ,ptr+32,sizeof(uint32_t));
        memcpy(&pixelLane_      ,ptr+36,sizeof(uint32_t));
        memcpy(&superChunkIndex_,ptr+40,sizeof(uint32_t));
        memcpy(&timeStampDelta_ ,ptr+44,sizeof(uint32_t));
        memcpy(&zmwNumber_      ,ptr+48,sizeof(uint32_t));

        return  x + Size_;
    }

    uint32_t Serialize(uint8_t* ptr, uint32_t maxLen) const override
    {
        uint32_t x = PacBio::IPC::Title<PacBio::Primary::Tile>::Serialize(ptr,maxLen);
        ptr += x;
        maxLen -= x;
        if (maxLen < Size_) throw PBException("Too small");
        memcpy(ptr+ 0,&frameIndexStart_,sizeof(uint64_t));
        memcpy(ptr+ 8,&timeStampStart_ ,sizeof(uint64_t));
        memcpy(ptr+16,&configWord_     ,sizeof(uint32_t));
        memcpy(ptr+20,&frameCount_     ,sizeof(uint32_t));
        memcpy(ptr+24,&zmwIndex_       ,sizeof(uint32_t));
        memcpy(ptr+28,&micOffset_      ,sizeof(uint32_t));
        memcpy(ptr+32,&flags_          ,sizeof(uint32_t));
        memcpy(ptr+36,&pixelLane_           ,sizeof(uint32_t));
        memcpy(ptr+40,&superChunkIndex_,sizeof(uint32_t));
        memcpy(ptr+44,&timeStampDelta_ ,sizeof(uint32_t));
        memcpy(ptr+48,&zmwNumber_      ,sizeof(uint32_t));
        return  x + Size_;
    }
    uint64_t FrameIndexStart() const { return frameIndexStart_; }
    void FrameIndexStart(uint64_t x) { frameIndexStart_ = x; }
    uint64_t TimeStampStart() const { return timeStampStart_ ; }
    void TimeStampStart(uint64_t x) { timeStampStart_ = x; }
    uint32_t TimeStampDelta() const { return timeStampDelta_; }
    void TimeStampDelta(uint32_t x) { timeStampDelta_ = x; }
    uint32_t ConfigWord() const { return configWord_; }
    void ConfigWord(uint32_t x) { configWord_ = x; }
    uint32_t FrameCount() const { return frameCount_; }
    void FrameCount(uint32_t x) { frameCount_ = x; }
    uint32_t ZmwIndex() const { return zmwIndex_; }
    void ZmwIndex(uint32_t x)  { zmwIndex_ = x; }
    uint32_t ZmwNumber() const { return zmwNumber_; }
    void ZmwNumber(uint32_t x)  { zmwNumber_ = x; }
    uint32_t MicOffset() const { return micOffset_;}
    void MicOffset(uint32_t x ) { micOffset_ = x;}


    uint32_t PixelLane() const { return pixelLane_;}
    void PixelLane(uint32_t lane) { pixelLane_ = lane;}
    uint32_t SuperChunkIndex() const { return superChunkIndex_;}
    void SuperChunkIndex(uint32_t index) {  superChunkIndex_ = index; }
    ITranche::StopStatusType StopStatus() const { return static_cast<ITranche::StopStatusType::RawEnum>(flags_);}
    void StopStatus(ITranche::StopStatusType status) {
        flags_ = static_cast<uint32_t>(status);
        assert(flags_ == ITranche::StopStatusType::NOT_STOPPED ||
               flags_ == ITranche::StopStatusType::NORMAL ||
               flags_ == ITranche::StopStatusType::BAD_DATA_QUALITY ||
               flags_ == ITranche::StopStatusType::INSUFFICIENT_THROUGHPUT);
    }

     void InitFromTranche(const ITranche& tranche, uint32_t micOffset)
     {
         frameIndexStart_ = tranche.FrameIndex();
         timeStampStart_ = tranche.TimeStampStart();
         timeStampDelta_ = tranche.TimeStampDelta();
         configWord_ = tranche.ConfigWord();
         frameCount_ = tranche.FrameCount();
         StopStatus(tranche.StopStatus());
         zmwIndex_   = tranche.ZmwIndex();
         micOffset_  = micOffset;
         pixelLane_       = tranche.PixelLaneIndex();
         superChunkIndex_ = tranche.SuperChunkIndex();
         zmwNumber_  = tranche.ZmwNumber();
     }

private:
    uint64_t frameIndexStart_; ///< frame index of first frame
    uint64_t timeStampStart_ ; ///< timestamp of first frame (microseconds)
    uint32_t configWord_;      ///< arbitrary 32 bit payload defined by ICS
    uint32_t frameCount_;      ///< number of frames in this Tranche (typically 16384 frames, but could be less for last Chunk)
    uint32_t zmwIndex_;        ///< index of first ZMW in simd channel 0. This Tranche includes 16 ZMWs.
    uint32_t micOffset_;
    uint32_t flags_;
    uint32_t pixelLane_;
    uint32_t superChunkIndex_;
    uint32_t timeStampDelta_;  ///< microseconds between frames
    uint32_t zmwNumber_;       ///< number of first ZMW in simd channel 0. Number is CellUnit location encoded.
 private:
    uint32_t usageCounter=0;
};

  inline std::ostream& operator<<(std::ostream& s, const TrancheTitle& title)
  {
      s << " micOffset:" << title.MicOffset();
      s << " zmwIndex:" << title.ZmwIndex();
      s << " zmwNumber:" << title.ZmwNumber();
      s << " frameCount:" << title.FrameCount();
      s << " frameIndexStart:" << title.FrameIndexStart();
      s << " numOffsets:" << title.GetNumOffsets();
      s << " pixellane:" << title.PixelLane();
      s << " superChunkIndex:" << title.SuperChunkIndex();
      s << " timeDelta:" << title.TimeStampDelta();
      return s;
  }
 }
}
