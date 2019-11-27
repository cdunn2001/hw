#ifndef Sequel_Primary_AcquisitionProxy_H_
#define Sequel_Primary_AcquisitionProxy_H_

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
/// \file	AcquisitionProxy.h
/// \brief	Classes used to stand in for an acquisition service, i.e. for
///         filling tranche buffers with pixel data from some source.

#include <memory>
#include <pacbio/primary/SequelTraceFile.h>
#include <pacbio/primary/ChipLayout.h>
#include <pacbio/primary/Acquisition_Setup.h>
#include <pacbio/primary/LaserPowerChange.h>


#include "ChunkFileSource.h"
#include "Tranche.h"

namespace PacBio
{
 namespace Primary
 {

/// A class to act as a stand-in for an acquisition service, i.e., fill tranches.
  class AcquisitionProxy
  {
  public:     // Structors

      AcquisitionProxy(size_t framesPerTranche);

      virtual ~AcquisitionProxy();

  public:     // Methods

      /// Retrieves all available laser power change events.
      virtual std::vector<LaserPowerChange> LaserPowerChanges() const;

      /// Acquisition moves in chunk-major order, filling lanes sequentially for each chunk.
      size_t FillNext(Tranche* tranche);


      /// Set the fraction of a full tranche to fill for the last (terminus) tranche
      AcquisitionProxy& FrameCount(uint64_t frames)
      {
          if (frames == 0)
          {
              throw PBException("frames must be >0 ");
          }
          frames_ = frames;
          return *this;
      }

#if 0
      /// Set the fraction of a full tranche to fill for the last (terminus) tranche
      AcquisitionProxy& TerminusTrancheFraction(float v)
      {
          terminusTrancheFraction_ = std::max(0.0f, std::min(1.0f, v));
          return *this;
      }
#endif

      void EnableCache();

  public:     // Properties
      uint64_t FrameCount() const
      { return frames_; }

#if 0
      float TerminusTrancheFraction() const
      { return terminusTrancheFraction_; }
#endif

      bool Finished() const
      { return chunkIndex_ >= numReplicatedSuperchunks_; }

      size_t NumTranches() const
      { return numReplicatedSuperchunks_ * numReplicatedLanes_; }

      void SetNumReplicatedLanes_(uint32_t numLanes)
      { numReplicatedLanes_ = numLanes; }

      void SetNumReplicatedSuperchunks_(uint32_t numSuperchunks)
      { numReplicatedSuperchunks_ = numSuperchunks; }

      void MicOffsetAndCount(uint32_t micOffset, uint32_t numMics)
      {
          micOffset_ = micOffset;
          numMics_ = numMics;
      }

      void SetROI(const SequelROI& roi, const ChipLayout& layout)
      {
          std::unique_ptr<SequelROI> sourceROI = GetSourceROI();
          SetROIUsingSourceROI(roi, layout, sourceROI.get());
      }

      void SetROIUsingSourceROI(const SequelROI& maskROI,
                                const PacBio::Primary::ChipLayout& layout,
                                SequelROI* sourceROI);

      SequelROI& GetOutputROI() const
      {
          return *outputROI_;
      }

      double FrameRate() const
      {
          return frameRate_;
      }

      virtual void SetRelax() { relax_ = true; }

      virtual std::unique_ptr<SequelROI> GetSourceROI() const;
      virtual uint32_t GetSourceFrames() const;
      virtual uint32_t GetFirstZmwNumber() const;
      virtual PacBio::Primary::ChipLayout& ChipLayout() const;
      //virtual Json::Value UpdateSetup();
      virtual Acquisition::Setup UpdateSetup();
      virtual uint32_t NumChannels() const { return 0; };

  protected:

      // The currently indexed tranche is terminus
      bool IsTerminusChunk()
      { return chunkIndex_ + 1 >= numReplicatedSuperchunks_; }

      // Make standard bookkeeping assignments to the tranche
      void AssignTrancheMetadata(Tranche* tranche, const size_t numFrames);

      void FillNextCached(Tranche* tranche);

      virtual uint32_t LoadTranche(Tranche::Pixelz* tranche, uint32_t lane, uint32_t super);

  protected:    // Data
      size_t numReplicatedLanes_;
      size_t numReplicatedSuperchunks_;
      size_t numSourceLanes_;
      size_t numSourceSuperchunks_;
      size_t framesPerTranche_;

      uint32_t numMics_;
      uint32_t micOffset_;

      mutable size_t laneIndex;
      mutable size_t chunkIndex_;
  private:
#if 0
      float terminusTrancheFraction_;  // [0,1] fraction for truncating frames in the terminus chunk.
#else
      uint64_t frames_;
#endif
      uint32_t timeStampDelta_;
      uint32_t configWord_;
      bool cache_;
      std::vector<std::vector<Tranche::Pixelz*> > pixelCache_;

      friend AcquisitionProxy* CreateAcquisitionProxy(uint32_t numLanes,
                                                      uint32_t numSuperchunks_,
                                                      const std::string& file);

      std::vector<uint32_t> selectedLanes_;
      std::unique_ptr<SequelSparseROI> outputROI_;
      std::vector<uint32_t> zmwNumbers_;
  protected:
      bool relax_ = false;
      mutable std::unique_ptr<PacBio::Primary::ChipLayout> chipLayout_;
      ChipClass chipClass_;
      double frameRate_;
  };

/// A class to act as a stand-in for an acquisition service, i.e., fill tranches,
/// using a ChunkFile as the source of data.
///
  class AcquisitionProxyChunkFile :
          public AcquisitionProxy
  {
  public:     // Structors
      AcquisitionProxyChunkFile(size_t framesPerTranche, const std::string& chunkFile);

      ~AcquisitionProxyChunkFile()
      { }

  public:     // Methods
      std::unique_ptr<SequelROI> GetSourceROI() const override;
      uint32_t GetSourceFrames() const override;
      uint32_t GetFirstZmwNumber() const override;

      /// Chunk files do not support laser power changes.
      /// \returns empty vector.
      std::vector<LaserPowerChange> LaserPowerChanges() const override
      { return std::vector<LaserPowerChange>(); }

      uint32_t LoadTranche(Tranche::Pixelz* tranche, uint32_t lane, uint32_t super) override;

  private:    // Data
      ChunkFileSource src_;
  };

  class AcquisitionProxyTraceFile : public AcquisitionProxy
  {
  public:     // Structors
      AcquisitionProxyTraceFile(size_t framesPerTranche, const std::string& traceFile);

      ~AcquisitionProxyTraceFile()
      { };

  public:     // Methods
      std::unique_ptr<SequelROI> GetSourceROI() const override;
      uint32_t GetSourceFrames() const override;
      uint32_t GetFirstZmwNumber() const override;
      PacBio::Primary::ChipLayout& ChipLayout() const override;

      /// Retrieves all available laser power change events.
      std::vector<LaserPowerChange> LaserPowerChanges() const override;

      uint32_t LoadTranche(Tranche::Pixelz* tranche, uint32_t lane, uint32_t super) override;
      //Json::Value UpdateSetup() override;
      Acquisition::Setup UpdateSetup() override;
      uint32_t NumChannels() const override
      {
          return src_.NUM_CHANNELS;
      }

  private:    // Data
      SequelTraceFileHDF5 src_;
      uint32_t timeStampDelta_;
      uint32_t configWord_;
  };

/// A factory to create an AcquisitionProxy, for filling tranches in the manner of an acquisition server.
  AcquisitionProxy* CreateAcquisitionProxy(size_t framesPerTranche, const std::string& file = "");

 } // Primary
}  // ::PacBio::Primary





#endif // Sequel_Primary_AcquisitionProxy_H_
