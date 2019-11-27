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
/// \brief Tranche interface

#ifndef Sequel_Common_Pacbio_Primary_ITracheNew_H_
#define Sequel_Common_Pacbio_Primary_ITracheNew_H_

#include <array>
#include <memory>
#include <stdint.h>

#include <pacbio/utilities/SmartEnum.h>
#include "TrancheSizes.h"
#include "ZmwResultBuffer.h"

#include <pacbio/primary/LaserPowerChange.h>
// TODO: Replace with a forward declaration of SchunkLaserPowerChangeSet.


#ifdef OLD_TRANCHE_API
#error OLD_TRANCHE_API is not supported any more!
#endif
namespace PacBio {
namespace Primary {

/// Consumer oriented interface for tranches, which include a trace data buffer,
/// a results buffer, and access to AcquisitionEvents.
class ITranche
{
public:
    struct Pointer
    {
        int tile  = 0 ;
        int frame = 0;
    };

public:     // Static constants
    static const size_t UnitCellsPerLane = PacBio::Primary::zmwsPerTranche;

public:     // Types
    SMART_ENUM(StopStatusType,
        NOT_STOPPED = 0,
        NORMAL = 1 ,
        BAD_DATA_QUALITY = 2,
        INSUFFICIENT_THROUGHPUT = 3
    );

public:     // Non-modifying functions
    /// Number of frames in the tranche.
    /// FrameCount() == 0 implies StopStatus() != NOT_STOPPED.
    virtual size_t FrameCount() const = 0;

    /// Absolute frame index of first frame represented.
    /// If FrameCount() == 0, returns an arbitrary value.
    virtual uint64_t FrameIndex() const = 0;

    /// Relative frame index of the first frame represented.
    /// If FrameCount() == 0, returns an arbitrary value.
    /// Uses RelativeFrameIndexOffset(), which is not thread-safe.
    virtual uint64_t RelativeFrameIndex() const = 0;

    /// Offset between relative and absolute frame indexes.
    /// Typically defined as the first frame index of the acquisition.
    /// Not thread-safe.
    virtual uint64_t RelativeFrameIndexOffset() const = 0;

    /// Timestamp of the first frame represented.
    /// The timestamp has microsecond precision of time elapsed sinced the Epoch.
    virtual uint64_t TimeStampStart() const = 0;

    /// The different in timestamps between frames.
    /// The difference is expressed in microseconds.
    virtual uint64_t TimeStampDelta() const = 0;

    uint64_t TimeStamp(uint32_t framePosition) const
    {
        return    TimeStampStart() + framePosition *  TimeStampDelta();
    }

    /// The 32-bit user defined payload that is transmitted by the sensor.
    virtual uint32_t ConfigWord() const = 0;

    /// A "pixel lane" is a block of contiguous Pixels, 32 pixels wide, defined by the Tile size coming from the FPGA
    /// It can be the same as the "ZMW lane" if there are 2 pixels per ZMW, or a factor of 2x from the "ZMW lane"
    /// if there are just 1 pixel per ZMW.
    virtual uint32_t PixelLaneIndex() const = 0;

    /// A "lane" is a block of ZmwsPerTranche unit cells.
    /// Each analyzer will receive tranches with 0 <= LaneIndex < number of
    /// lanes to analyze.
    virtual uint32_t ZmwLaneIndex() const = 0;

    /// An identifier for the first unit cell (i.e., ZMW) of the lane. This is a 32-bit value that
    /// encodes X and Y unit cell coordinates
    virtual uint32_t ZmwNumber() const = 0;

    /// A zero based index for the first ZMW of this lane/tranche.
    /// 0 is the first ZMW processed by the sequencing ROI.
    virtual uint32_t ZmwIndex() const = 0;

    /// Defines chronological order of tranches for any particular lane.
    virtual uint32_t SuperChunkIndex() const = 0;

    /// Indicates the status (reason) for terminating the acquisition.
    /// FrameCount() == 0 implies StopStatus() != NOT_STOPPED.
    virtual StopStatusType StopStatus() const = 0;

    /// Indicates that the tranche includes the last chunk of the acquisition.
    bool IsStopped() const
    { return StopStatus() != StopStatusType::NOT_STOPPED; }

public:     // Modifying functions
    /// Set the StopStatus.
    /// \returns *this.
    virtual ITranche& StopStatus(StopStatusType value) = 0;

    virtual void ZmwLaneIndex(uint32_t lane) = 0 ;

    virtual void ZmwIndex(uint32_t index) = 0;

    virtual void ZmwNumber(uint32_t number) = 0;

    /// Sets the offset between relative and absolute frame indexes.
    /// Typically defined as the first frame index of the acquisition.
    /// Not thread-safe.
    virtual void RelativeFrameIndexOffset(uint64_t value) = 0;

public:     // Data access
    /// Trace data
    /// Layout:
    /// uint16_t per pixel = 1.
    /// Pixels per ZMW = 2.
    /// ZMWs per lane-frame (a.k.a. SIMD pixel) = UnitCellsPerLane.
    /// Lane-frames per tile = FrameCount() <= FramesPerTranche.
    virtual const int16_t* TraceData() const = 0;

    virtual void CopyTraceData(void* dst,Pointer& pointer,size_t count) const = 0;

    /// Output buffers
    /// See ReadBuffer for buffer capacities.
    virtual std::array<ReadBuffer*, PacBio::Primary::zmwsPerTranche>& Calls() = 0;

public:     // Events
    /// The number of laser power changes associated with this
    /// tranche's superchunk.
    virtual size_t NumLaserPowerChanges() const = 0;

    /// The shared pointer to the immutable random-access collection of events
    /// associated with this tranche's superchunk.
    virtual const std::shared_ptr<const SchunkLaserPowerChangeSet>
    LaserPowerChanges() const = 0;

    virtual std::shared_ptr<const SchunkLaserPowerChangeSet>&
    LaserPowerChanges() = 0;

protected:  // Disallow polymorphic destruction.
    virtual ~ITranche() noexcept = default;
};


/// Dump tranche metadata to std::ostream.
inline std::ostream& operator<<(std::ostream& s, const ITranche* t)
{
    s << "addr=" << (void *)t;
    if (t != nullptr) s
        << " [Lane " << t->ZmwLaneIndex()
        << ", SChunk " << t->SuperChunkIndex()
        << ", Zmw# " << std::hex << t->ZmwNumber() << std::dec
        << ", ZmwIdx " << t->ZmwIndex()
                << "]"
        << ", FrameCount=" << t->FrameCount()
        << ", StopStatus=" << t->StopStatus();

    return s;
}

}}  // PacBio::Primary

#endif // Sequel_Common_Pacbio_Primary_ITracheNew_H_
