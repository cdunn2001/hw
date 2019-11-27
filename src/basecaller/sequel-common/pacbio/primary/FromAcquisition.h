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

#pragma once

#include <pacbio/ipc/ThreadSafeQueue.h>
#include <pacbio/primary/SCIF_Heartbeat.h>
#include <pacbio/ipc/MessageQueue.h>
#include <pacbio/ipc/PoolFactory.h>
#include <pacbio/primary/Tile.h>
#include <pacbio/primary/Tranche.h>
#include <pacbio/process/ProcessBase.h>

namespace PacBio
{
 namespace Primary
 {
  class FromAcquisition
  {
  private:
      /// socket for MIC -> Host communications
      ///
      PacBio::IPC::MessageSocketSender pa_acq_;

      SCIF_Heartbeat heartbeat_;

      uint32_t trancheBufferSize_;

      size_t elementSize_;

      /// A queue of unused tranches. When a deed message arrives from the host,
      /// the next free tranche is popped off this queue, and when the thread is
      /// done analyzing the data, it is pushed to this queue.
      //PacBio::ThreadSafeQueue<ITranche*> trancheFreeQueue_;
      PacBio::ThreadSafeQueue<Tranche*> trancheFreeQueue_;

      PacBio::ThreadSafeQueue<Tranche*> trancheMyQueue_;

      PacBio::ThreadSafeQueue<Tranche*>& trancheReadyQueue_;

  protected:
      /// Contains incoming tranche pixel data. Allocated in one big huge-page allocation.
      ///
      std::unique_ptr<PacBio::IPC::PoolBaseTyped <Tranche::Pixelz>> poolz_;
  private:
      uint32_t trancheBufferUnderflow_{0};
      uint32_t tranchesProcessed_{0};
      uint32_t lastLaneIndex_{0};
      uint32_t numSubsetsPerTile_{0};
      Tranche::PixelFormat format_;
      uint64_t tilesDeedToT2b_{0};
      uint64_t tilesReleasedFromT2b_{0};
      mutable std::mutex tilesReleasedFromT2bMutex_;
    public:
      FromAcquisition(const PacBio::IPC::MessageQueueLabel&a,
                     const PacBio::IPC::MessageQueueLabel& b ,
                     uint32_t trancheBufferSize,
                      PacBio::ThreadSafeQueue<Tranche*>& trancheReadyQueue,
                      size_t elementSize);

      ///
      /// "deed" command handler
      ///
      /// A "deed" message sends titles between acquisition and basecaller. This
      /// handler is called when the acquisition sends a new tranche to the
      /// basecaller.
      ///
      /// returns false if there was a failure in receiving the deed (free queue empty)
      bool ReceiveDeed(const PacBio::IPC::Message& mesg, std::chrono::milliseconds timeout);
      void ReceiveMessage(Tranche* msg);
      void Init(const Json::Value& poolConfig);
      void AddFreeTranche(Tranche* tranche);
      void SetSCIF_Ready();
      void SetPipelineReady();
      bool PipelineReady() const;
      bool EverythingReady() const;
      void MainThread(PacBio::Process::ThreadedProcessBase& parent);
      Tranche::Pixelz* GetPointerByIndex(int i)
      {
          return poolz_->GetPointerByIndex(i);
      }
      size_t NumFreeTranches() const
      {
          return trancheFreeQueue_.Size();
      }
      size_t NumReadyTranches() const
      {
          return trancheReadyQueue_.Size();
      }

      void SendDeed(const PacBio::IPC::Deed& d) { pa_acq_.Send(d); }
      uint64_t TilesDeedToT2b() const {
        std::lock_guard<std::mutex> lock(tilesReleasedFromT2bMutex_);
        return tilesDeedToT2b_;
      }
      uint64_t TilesReleasedFromT2b() const {
        std::lock_guard<std::mutex> lock(tilesReleasedFromT2bMutex_);
        return tilesReleasedFromT2b_;
      }

  };
 }
}


