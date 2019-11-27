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
/// \brief  declaration of Tranche, used by t2b as the data unit of analysis

#ifndef _TRANCHE_H_
#define _TRANCHE_H_

#include <cstdint>
#include <array>
#include <functional>
#include <memory>

#if defined(PB_KNC) || defined(PB_CORE_AVX512)
#include <immintrin.h>
#endif
#ifdef __INTEL_COMPILER
#include <aligned_new>  //cf https://software.intel.com/en-us/node/522531
#endif

#include <pacbio/ipc/TitleBase.h>
#include <pacbio/primary/Tile.h>
#include <pacbio/smrtdata/Basecall.h>
#include <pacbio/primary/ZmwResultBuffer.h>
#include <pacbio/primary/TrancheTitle.h>
#include <pacbio/primary/ITrancheNew.h>
#include <pacbio/primary/LaserPowerChange.h>

#define round_up_4k(x) ((x+4095)/4096*4096)

namespace PacBio {
namespace Primary {

class Tranche : public ITranche
{
public:     // Types
    enum class MessageType
    {
        Data,
        AcquisitionSetup,
        AcquisitionStart,
        AcquisitionStop,
        Abort,
        Init,
        Unknown
    };

    SMART_ENUM(PixelFormat,Format1C4A_RT,Format2C2A,Format1C4A_CA);


class Pixelz
{
    Pixelz() = delete;
public:

    typedef union
    {
        uint8_t u8[64];
        int16_t i16[32];
#if 0 && (defined(PB_KNC) || defined(PB_CORE_AVX512))
        __m512 m512;
#endif
    } SimdPixel;

#ifdef WIN32
#pragma warning(disable:337)
#endif

    static Pixelz* Factory(uint32_t numFrames)
    {
        return reinterpret_cast<Pixelz*>(new SimdPixel[numFrames]);
    }

    union Union
    {
        Union() {}
        SimdPixel simd[1];
        Tile tiles[1];
        int16_t int16[1];
    } pixels;

    static size_t Sizeof(uint32_t framesPerTranche) {
        return framesPerTranche * sizeof(SimdPixel);
    }
#ifdef WIN32
#pragma warning(default:337)
#endif
};


public:     // Data
	MessageType type;

private:
    Pixelz* data = nullptr;
    std::vector<Tile*> tiles;
public:
    std::array<ReadBuffer*,zmwsPerTranche> calls;

    uint64_t order;
#define SPIDER_TRANCHES
#ifdef  SPIDER_TRANCHES

private:    // Static data
    // The difference between relative and absolute frame indexes.
    // Typically, equal to the frame index of tranches from the first superchunk of an acquisition.
    static uint64_t relativeFrameIndexOffset_;

private:
    std::shared_ptr<TrancheTitle> title_;
    int subset_=0; // 0 or 1. 0 for all of sequel, 0 or 1 for spider
    PixelFormat format_;
    int numSubsets_=1;
    std::shared_ptr<const SchunkLaserPowerChangeSet> lpcSet_;

public:
    const TrancheTitle& Title() const { return *title_;}
    TrancheTitle& Title() { return *title_;}
    void CreateTitle(PixelFormat pixelFormat)
    {
        std::shared_ptr<TrancheTitle> title(new TrancheTitle);
        SetSharedTitle(title,0,pixelFormat);
    }
    void Create2C2ATitle()
    {
        CreateTitle(PixelFormat::Format2C2A);
    }
    void Create1C4A_RT_Title()
    {
        CreateTitle(PixelFormat::Format1C4A_RT);
    }
    void Create1C4A_CA_Title()
    {
        CreateTitle(PixelFormat::Format1C4A_CA);
    }

    void SetSharedTitle(std::shared_ptr<TrancheTitle>& title, int subset, PixelFormat format)
    {
        title_ = title;
        subset_ = subset;
        format_ = format;
        switch(format)
        {
        case PixelFormat::Format1C4A_CA:
            numSubsets_ = 1;
            break;
        case PixelFormat::Format1C4A_RT:
            numSubsets_ = 2;
            break;
        case PixelFormat::Format2C2A:
            numSubsets_ = 1;
            break;
        }
        assert(subset_ < numSubsets_);
    }
    void TitleDestroy() { title_.reset(); }
#else
private:
    TrancheTitle title_;
public:
    const TrancheTitle& Title() const { return title_;}
    TrancheTitle& Title() { return title_;}
#endif
public:
    int mark;
    TrancheTitle titleOut_;
    int markOut_;
    std::string payload_;

public:     // Structors
    Tranche()
        : type(MessageType::Unknown)
        , data(nullptr)
		, order(0)
    { }
    Tranche(const Tranche& t) = delete;
    Tranche(Tranche&& t) = delete;
    Tranche& operator=(const Tranche& ) = delete;
    Tranche& operator=(Tranche&& ) = delete;
    ~Tranche() noexcept = default;

public: // ITranche implementation

    size_t FrameCount() const override
    { return Title().FrameCount(); }

    uint64_t FrameIndex() const override
    { return Title().FrameIndexStart(); }

    /// Relative frame index of the first frame represented.
    /// If FrameCount() == 0, returns an arbitrary value.
    /// Uses RelativeFrameIndexOffset(), which is not thread-safe.
    uint64_t RelativeFrameIndex() const override
    { return FrameIndex() - RelativeFrameIndexOffset(); }

    /// Returns a statically defined offset between relative and absolute
    /// frame indexes.
    /// Not thread-safe.
    uint64_t RelativeFrameIndexOffset() const override
    { return relativeFrameIndexOffset_; }

    /// Sets the offset between relative and absolute frame indexes.
    /// Typically defined as the first frame index of the acquisition.
    /// Not thread-safe.
    void RelativeFrameIndexOffset(uint64_t value) override
    { relativeFrameIndexOffset_ = value; }

    /// The number of laser power changes associated with this
    /// tranche's superchunk.
    size_t NumLaserPowerChanges() const override
    { return lpcSet_ ? lpcSet_->size() : 0u; }

    /// The shared pointer to the immutable random-access collection of events
    /// associated with this tranche's superchunk.
    const std::shared_ptr<const SchunkLaserPowerChangeSet>
    LaserPowerChanges() const override
    { return lpcSet_; }

    std::shared_ptr<const SchunkLaserPowerChangeSet>&
    LaserPowerChanges() override
    { return lpcSet_; }

    uint64_t TimeStampStart() const override
    { return Title().TimeStampStart(); }

    uint64_t TimeStampDelta() const override
    { return Title().TimeStampDelta(); }

    uint32_t ConfigWord() const override
    { return Title().ConfigWord(); }

#ifdef  SPIDER_TRANCHES
    uint32_t Subset() const
    { return subset_; }

    uint32_t PixelLaneIndex() const override
    { return Title().PixelLane(); }

    void ZmwLaneIndex(uint32_t lane) override
    { Title().PixelLane(((lane - subset_)/numSubsets_)); }

    uint32_t ZmwLaneIndex() const override
    { return Title().PixelLane()*numSubsets_ + subset_; }

    void ZmwIndex(uint32_t index) override
    { Title().ZmwIndex(index - subset_* zmwsPerTranche); }

    uint32_t ZmwIndex() const override
    { return Title().ZmwIndex() + (subset_ * zmwsPerTranche); }

    void ZmwNumber(uint32_t number) override
    { Title().ZmwNumber(number - subset_ * zmwsPerTranche); }

    uint32_t ZmwNumber() const override
    { return Title().ZmwNumber() + (subset_ * zmwsPerTranche); }

    PixelFormat Format() const
    { return format_; }
#else
    uint32_t ZmwLaneIndex() const override
    { return Title().PixelLane(); }
    
    uint32_t ZmwIndex() const override
    { return Title().ZmwIndex(); }

    uint32_t ZmwNumber() const override
    { return Title().ZmwNumber(); }
#endif

    uint32_t SuperChunkIndex() const override
    { return Title().SuperChunkIndex(); }

    StopStatusType StopStatus() const override
    { return Title().StopStatus(); }
    
    ITranche& StopStatus(StopStatusType value) override
    { Title().StopStatus(value); return *this; }

    const int16_t* TraceData() const override
    {
#ifdef  SPIDER_TRANCHES
        if (subset_ != 0) throw PBException("Can't use TraceData with Spider Interleaved data");
#endif
        return reinterpret_cast<const int16_t*>(data);
    }

    // other members
    Pixelz* GetTraceDataPointer()
    { return data; }

    void AssignTraceDataPointer(int16_t* ptr)
    { data = reinterpret_cast<Pixelz*>(ptr); }

    void AssignTraceDataPointer(Pixelz* ptr)
    { data = ptr; }

    void AssignTileDataPointers(const std::vector<Tile*>& tiles0)
    { tiles = tiles0; }

    void CopyTraceData(void* dst,Pointer& pointer,size_t numFrames) const override;

public:     // Other properties

    MessageType Type() const         { return type; }
    size_t ZmwCount() const          { return zmwsPerTranche; }
    
    uint64_t Checksum() const
    {
        uint64_t sum = 0;
        if (data)
        {
            const uint64_t* ptr = reinterpret_cast<const uint64_t*>(data);
            for (size_t i = 0; i < FrameCount() * 64 / sizeof(uint64_t); i++)
            {
                sum += ptr[i];
            }
        }
        else if (tiles.size() > 0)
        {
            for (auto tile : tiles)
            {
                sum += tile->Checksum();
            }
        }
        else
        {
            throw PBException("no data, no tiles");
        }
        return sum;
    }

 private:
    std::array<ReadBuffer*, zmwsPerTranche>& Calls() { return calls; }
 public:
    const std::array<ReadBuffer*, zmwsPerTranche>& Calls() const { return calls; }
};

 std::ostream& operator<<(std::ostream& s, const PacBio::Primary::Tranche& t);
 std::ostream& operator<<(std::ostream& s, const PacBio::Primary::Tranche::Pixelz* t);

 class TrancheHelper
 {
 public:
     TrancheHelper(int verbosity) : verbosity_(verbosity){}
     std::ostream* s;
     int verbosity_;
 };
 inline TrancheHelper& operator<<(std::ostream& s, TrancheHelper& x) {x.s = &s; return x;}
 TrancheHelper& operator<<(TrancheHelper& s, const PacBio::Primary::Tranche::Pixelz* t);

}} // ::PacBio::Primary

#endif // _TRANCHE_H_
