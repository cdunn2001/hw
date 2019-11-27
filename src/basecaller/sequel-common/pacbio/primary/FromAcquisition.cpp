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

/// This code manages the first few stages of the upstream message queue. Messages are received
/// from pa-acq via IPC, the put into a series of queues.
///  1. trancheMyQueue_
///  2. trancheReadyQueue_
///  when the tranche is done it is pushed to
///  3. traceFreeQueue_
///  which is then recycled back in stage 1.
///
/// The SCIF handshaking is performed as follows
/// 1. t2b send heartbeats via IPC to the peer (pa-acq)
/// 2. peer receives heartbeat and send back "init" message to t2b.
/// 3. t2b opens a SCIF pool connection to the peer
/// 4. t2b sets SCIF_Ready flag, which goes out in the next heartbeat
/// 5.


#include <pacbio/ipc/ThreadSafeQueue.h>
#include <pacbio/ipc/MessageQueue.h>
#include <pacbio/ipc/PoolFactory.h>
#include <pacbio/logging/Logger.h>
#include <pacbio/process/ProcessBase.h>
#include <pacbio/process/RESTServer.h>

#include <pacbio/primary/FromAcquisition.h>
#include <pacbio/primary/ipc_config.h>
#include <pacbio/primary/PrimaryConfig.h>
#include <pacbio/primary/SCIF_Heartbeat.h>
#include <pacbio/primary/Tile.h>
#include <pacbio/primary/Tranche.h>
#include <pacbio/primary/REST_RootInterface.h>

using namespace PacBio::IPC;

static uint64_t numDeeds = 0;
static uint64_t bytesReceived = 0;
static uint64_t freePushed = 0;
static uint64_t gotMarks = 0;
static uint64_t waitedForTranches = 0;

namespace PacBio  {
namespace Primary {

  /// Constructor.
  FromAcquisition::FromAcquisition(const MessageQueueLabel& a,
                                   const MessageQueueLabel& b,
                                   uint32_t trancheBufferSize,
                                   ThreadSafeQueue<Tranche*>& trancheReadyQueue,
                                   size_t elementSize)
          : pa_acq_(b)
          , heartbeat_(a)
          , trancheBufferSize_{trancheBufferSize}
          , elementSize_{elementSize}
          , trancheReadyQueue_(trancheReadyQueue)
  {
      pa_acq_.SetNoLinger();
#if 1
      pa_acq_.SetHighWaterMark(10000);
#endif

      switch(GetPrimaryConfig().cache.numColors)
      {
      case 1:
          numSubsetsPerTile_ = 2;
          format_ = Tranche::PixelFormat::Format1C4A_RT;
          break;
      case 2:
          numSubsetsPerTile_ = 1;
          format_ = Tranche::PixelFormat::Format2C2A;
          break;
      default:
          throw PBException("Not supported");
      }
  }

  ///
  /// "deed" command handler
  ///
  /// A "deed" message sends titles between acquisition and basecaller. This
  /// handler is called when the acquisition sends a new tranche to the
  /// basecaller.
  ///
  bool FromAcquisition::ReceiveDeed(const PacBio::IPC::Message& mesg, std::chrono::milliseconds timeout)
  {
      bool success = false;
      numDeeds++;
      Tranche* tranche;

#ifdef SPIDER_TRANCHES
      std::shared_ptr<TrancheTitle> title(new TrancheTitle, [this](TrancheTitle* p) {
          PacBio::IPC::Deed deed("deed", *p);
          this->SendDeed(deed);
          PBLOG_TRACE << "Deleted title for zmw index:" << p->ZmwIndex();
          {
              std::lock_guard<std::mutex> lock(tilesReleasedFromT2bMutex_);
              tilesReleasedFromT2b_ += p->GetNumOffsets();
          }
          delete p;
      });

      title->DeserializeMessage(&mesg);
#endif

      {
          std::lock_guard<std::mutex> lock(tilesReleasedFromT2bMutex_);
          tilesDeedToT2b_ += title->GetNumOffsets();
      }

#ifdef SPIDER_TRANCHES
      for(uint32_t subset=0;subset<numSubsetsPerTile_;subset++)
      {
#endif
          if (trancheFreeQueue_.Pop(tranche, timeout))
          {
              PBLOG_TRACE << "FromAcquisition::ReceiveDeed _deed! tranche:" << (void*) tranche;

#ifdef OLD_TRANCHE_API
              tranche->type = ITranche::MessageType::Data;
#endif
#ifdef SPIDER_TRANCHES
              tranche->SetSharedTitle(title,subset,format_);
#else
              tranche->Title().DeserializeMessage(&mesg);
#endif

#if 0
              if (tranche->ZmwLaneIndex() < 10)
              {
                  PBLOG_INFO << "FromAcquisition::ReceiveDeed: tranche: " << *tranche << " title*:" << (void*) &tranche->Title();
              }
#endif
              // assign new crop of ZMWs to each ReadBuffer
              uint32_t zmwIndex = tranche->ZmwIndex();
              for (int j = 0; j < 16; j++)
              {
                  tranche->calls[j]->Reset(zmwIndex++);
              }

              if (tranche->Title().GetNumOffsets() > GetPrimaryConfig().cache.tilesPerTranche)
              {
                  throw PBException("can't support more than M elements in a title");
              }
              if (lastLaneIndex_ == 0 || tranche->ZmwLaneIndex() < lastLaneIndex_)
              {
                  // ok
              }
              else if (tranche->ZmwLaneIndex() != lastLaneIndex_ + 1)
              {
                  PBLOG_WARN << "Missing lane(s). lastLaneIndex:" << lastLaneIndex_ << " tranche->title:"
                              << tranche->Title();
              }
              lastLaneIndex_ = tranche->ZmwLaneIndex();

              auto* ipwc = dynamic_cast<IPoolWithCopies*>(poolz_.get());
              if (ipwc)
              {
                  for (uint32_t j = 0; j < tranche->Title().GetNumOffsets(); j++)
                  {
                      Tile* dst = &tranche->GetTraceDataPointer()->pixels.tiles[j];
                      uint64_t remoteOffset = tranche->Title().GetOffset(j);

                      PBLOG_TRACE << "FromAcquisition::ReceiveDeed tile [" << j << "] dst: " << (void*) dst
                                      << " offset:" << remoteOffset
                                      << " size:" << tranche->Title().GetElementSize();

                      //Copies bytes from remoteOffset to localOffset
#ifdef CLIENT_LOCAL_REGISTER
                      uint64_t localOffset = poolz_->GetOffset(dst);
                      // NOTE - COPYING HAPPENS ASYNC
                      ipwc->CopyBytesFromRemotePoolAsync(localOffset, remoteOffset, sizeof(Tile));
                      bytesReceived += sizeof(Tile);
#else
                      pool.CopyBytesFromRemotePoolAsync(dst, remoteOffset, sizeof(Tile) * junk);
                     // pool.WaitForAsyncCopy();
#endif
                  }

                  gotMarks++;
                  tranche->mark = ipwc->GetMark();
                  PBLOG_TRACE << "getMark:" << tranche->mark;
              }
              else
              {
                  std::vector<Tile*> tiles;
                  for (uint32_t j = 0; j < tranche->Title().GetNumOffsets(); j++)
                  {
                      auto offset = tranche->Title().GetOffset(j);
                      Tile* tile = static_cast<Tile*>(poolz_->GetVoidPointerByOffset(offset));
                      tiles.push_back(tile);
#if 0
                      // dump out raw data of the first lane, for low level debugging.
                      if (tranche->ZmwLaneIndex() < 1)
                      {
                          PBLOG_INFO << "lane0: tile:" << j << ", offset:" << offset << " Tile*:" << (void*) tile;
                          for(int iframe=0;iframe<32;iframe++)
                          {
                              std::stringstream ss;
                              for (int pix = 0; pix < 32; pix++)
                              {
                                  ss << tile->GetPixel(pix, 0);
                              }
                              for (int pix = 0; pix < 32; pix++)
                              {
                                  ss << tile->GetPixel(pix, 0);
                              }
                              PBLOG_INFO << "  frame:" << iframe << " " << ss.str();
                          }
                      }
#endif
                  }
                  tranche->AssignTileDataPointers(tiles);
              }

              if (tranche->ZmwLaneIndex() < 1)
              {
                  PBLOG_INFO << "Got a title: lane:" << tranche->ZmwLaneIndex() << " data:"
                              << (void*) tranche->TraceData()
                              << " calls[0]:" << (void*) tranche->calls[0] << "\n" << *tranche;
              }
              trancheMyQueue_.Push(tranche);
              success = true;
          }
          else
          {
              trancheBufferUnderflow_++;
              //   timeout =std::chrono::milliseconds(0);;
              PBLOG_ERROR << "trancheFreeQueue pop timeout";
          }
#ifdef SPIDER_TRANCHES
      }

#ifndef NDEBUG
      // check that my math is correct. We purposesly keep the title instance
      // alive until past the end of this loop, so that all of the other shared instances
      // are created. We don't want to destroy it immediately!
      if (title.use_count() != (1+numSubsetsPerTile_))
      {
        throw PBException("Internal error, mismatched use_count:" + std::to_string(title.use_count()));
      }
#endif
#endif

      PBLOG_TRACE << "FromAcquisition::ReceiveDeed deed callback done";
      return success;
  }

  /// Process an incoming message from the upstream peer (pa-acq).
  /// This simply pushs it on the first incoming queue.
  void FromAcquisition::ReceiveMessage(Tranche* msg)
  {
      trancheMyQueue_.Push(msg);
  }

  /// Initializes the pool. This is only performed after receiving the "init" command from the peer (pa-acq)
  void FromAcquisition::Init(const Json::Value& poolConfigJson)
  {
      PacBio::IPC::PoolFactory::PoolConfig poolConfig;
      poolConfig.Load(poolConfigJson);
      poolConfig.defaultPoolType = GetPrimaryConfig().GetPoolType();

      // The dimensions of the pool must change, because the pa-acq is dimensioned in terms of
      // tiles, and pa-t2b wants to dimension in terms of tranches.
      // For "copyable" pools we can arbitrarily resize the pa-t2b pool, less than or equal to
      // the pa-acq size.
      // For "shared memory" pools, the redimensioned pa-t2b pool must match the
      // pa-acq size (or at least be less than), otherwise the shared memory mapping will fail.
      if (IPoolWithCopies::SupportsCopies(poolConfig.GetBestPoolType()))
      {
          // The pool memory size here does not need to match pa-acq, because we are copying
          // memory via SCIF. The memory of pa-t2b is much less than pa-acq, so the pool on
          // t2b side can be much smaller.  So we can pick new dimensions of the pool that
          // match the buffer requirements.
          // For example, t2b sequel uses the number of MIC threads (i.e. 240) times a fudge factor
          // (i.e. 16) to get the number of elements necessary, one the order of 4096. Times
          // a tranche size of 524288, that is 2GB. In pa-acq, the pool size is about 100GB.
          poolConfig.numElements = trancheBufferSize_;
      }
      else
      {
          if (poolConfig.numElements == 0)
          {
              PBLOG_ERROR << "poolConfig:" << poolConfig.Json();
              throw PBException("numElements field of the poolConfig must not be zero");
          }
          // The t2b pool memory size must be the same total size as the pa-acq size, but
          // we want to change the dimensions of the tranche dimension. To do that,
          // we need to reduce the number of elements. Note that this doesn't actually
          // use any more memory, the memory is only allocated once in pa-acq, and
          // shared to pa-t2b. But it must be consistent.
          uint64_t entirePoolSize = poolConfig.elementSize * poolConfig.numElements;
          uint64_t newNumElements = entirePoolSize / elementSize_;
          poolConfig.numElements = newNumElements;
      }

      // overwrite the elementSize
      poolConfig.elementSize = elementSize_;

      PBLOG_INFO << "poolConfig:" << poolConfig.Json();
      poolz_.reset( PacBio::IPC::PoolFactory::ConstructClient<Tranche::Pixelz>(poolConfig));

      PBLOG_INFO << "Connecting to host pool";
      poolz_->Connect();
      PBLOG_INFO << "Allocated " << poolz_->GetTotalSize() << " bytes on client for SCIF buffering";
  }

  /// Puts a freed tranche back into the free queue.
  void FromAcquisition::AddFreeTranche(Tranche* tranche)
  {
      freePushed++;
      trancheFreeQueue_.Push(tranche);
  }

  /// This means that the SCIF connection is 100% connected.
  void FromAcquisition::SetSCIF_Ready()
  {
      // This means that this M is ready to start processing tranches.
      PBLOG_DEBUG << "FromAcquisition SCIF is ready";
      heartbeat_.SetSCIF_Ready();
  }

  void FromAcquisition::SetPipelineReady()
  {
      // This means that this MIC is ready to start processing tranches.
      PBLOG_DEBUG << "FromAcquisition pipeline is ready";
      heartbeat_.SetPipelineReady();
  }

  bool FromAcquisition::PipelineReady() const
  {
      return heartbeat_.PipelineReady();
  }

  bool FromAcquisition::EverythingReady() const
  {
      return heartbeat_.SCIF_Ready() && heartbeat_.PipelineReady();
  }

  void FromAcquisition::MainThread(PacBio::Process::ThreadedProcessBase& parent)
  {
      PBLOG_INFO << "Started FromAcquisition::MainThread";

      auto t0 = std::chrono::system_clock::now();
      while (!parent.ExitRequested())
      {
          Tranche* tranche;
          if (trancheMyQueue_.Pop(tranche, std::chrono::milliseconds(1000)))
          {
              PBLOG_TRACE << "FromAcquisition::MainThread(Got a tranche:" << (void*) tranche;

              if (tranche == nullptr)
              {
                  PBLOG_WARN << "FromAcquisition::MainThread() null tranche !";
                  return;
              }
              switch (tranche->type)
              {
              case Tranche::MessageType::Data:
              {
                  auto* ipwc = dynamic_cast<IPoolWithCopies*>(poolz_.get());
                  if (ipwc)
                  {
                      PBLOG_TRACE << "FromAcquisition::MainThread DATA MESG , wait for mark:" << tranche->mark;
                      // WaitForMark  - signifies that all incoming data has been copied to tranchequeue.
                      // this is a blocking call, internally it implements pbi_scif_fence_wait
                      ipwc->WaitForMark(tranche->mark);
                      waitedForTranches++;
                      // release deed immediately after data is copied
#ifndef SPIDER_TRANCHES
                      Deed deed("deed", tranche->Title());
                      pa_acq_.Send(deed);
#endif
                  }
                  tranchesProcessed_ += tranche->Title().GetNumOffsets();
                  break;
              }

              case Tranche::MessageType::AcquisitionStart:
                  tranchesProcessed_ = 0;
                  break;
              case Tranche::MessageType::AcquisitionStop:
                  PBLOG_DEBUG << "FromAcquisition::MainThread() Stop called. ";
                  break;
              case Tranche::MessageType::Abort:
                  PBLOG_INFO << "FromAcquisition::MainThread() Abort called. ";
//                  parent.Abort();
#if 0
                  // fixme
                  analyzer->Abort();
#endif
                  break;
              default:
                  PBLOG_DEBUG << "ignored message";
                  break;
              }
              trancheReadyQueue_.Push(tranche);
          }

          if (std::chrono::system_clock::now() - t0 >= std::chrono::milliseconds(1000))
          {
              t0 = std::chrono::system_clock::now();


              PBLOG_DEBUG << "FromAcquisition::MainThread sendinging heartbeat" << heartbeat_.Serialize();
              pa_acq_.Send(Announcement("heartbeat", heartbeat_.Serialize()));

              {
                  auto root = GetREST_Root();

                  root->ModifyNumber("fromAcquisition/trancheMyQueue", trancheMyQueue_.Size());
                  root->ModifyNumber("fromAcquisition/trancheFreeQueue", trancheFreeQueue_.Size());
                  root->ModifyNumber("fromAcquisition/trancheDataQueue",trancheReadyQueue_.Size());
                  root->ModifyNumber("fromAcquisition/trancheBufferUnderflow_", trancheBufferUnderflow_);
                  root->ModifyNumber("fromAcquisition/numDeeds", numDeeds);
                  root->ModifyNumber("fromAcquisition/bytesReceived", bytesReceived);
                  root->ModifyNumber("fromAcquisition/freePushed", freePushed);
                  root->ModifyNumber("fromAcquisition/gotMarks", gotMarks);
                  root->ModifyNumber("fromAcquisition/waitedForTranches", waitedForTranches);
                  root->ModifyNumber("fromAcquisition/tranchesProcessed_", tranchesProcessed_);
                  root->ModifyNumber("fromAcquisition/tilesDeedToT2b_", TilesDeedToT2b());
                  root->ModifyNumber("fromAcquisition/tilesReleasedFromT2b_", TilesReleasedFromT2b());
              }
          }
      }
  }


}} // namespace


