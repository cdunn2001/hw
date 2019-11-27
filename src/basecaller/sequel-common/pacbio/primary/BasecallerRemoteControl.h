
// Copyright (c) 2014, Pacific Biosciences of California, Inc.
// Copyright (c) 2014, Pacific Biosciences of California, Inc.
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
// File Description:
/// \brief  a class used to control the pa-t2b service remotely (using IPC messages)
//
// Programmer: Mark Lakata

#ifndef SEQUELACQUISITION_BASECALLERREMOTECONTROL_H
#define SEQUELACQUISITION_BASECALLERREMOTECONTROL_H

#pragma once

#include <chrono>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <atomic>

#include <pacbio/ipc/Message.h>
#include <pacbio/ipc/MessageQueue.h>
#include <pacbio/primary/ipc_config.h>
#include <pacbio/primary/SequelROI.h>
#include <pacbio/primary/TrancheTitle.h>
#include <pacbio/logging/Logger.h>
#include <pacbio/process/ProcessBaseRemoteControl.h>

#include <json/json.h>
#include <pacbio/utilities/SmartEnum.h>

#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_DEFAULT "\033[39m"

namespace PacBio
{
 using namespace IPC;

 namespace Primary
 {
  using namespace std;

  class BasecallerStatus : public PacBio::Process::ConfigurationObject
  {
  public:
      SMART_ENUM(GlobalState, undefined, boot, idle, armed, capture, error, offline);

      ADD_PARAMETER(double,time,0.0);
      ADD_ENUM(GlobalState,state,GlobalState::undefined);
      ADD_PARAMETER(std::string,token,"");
  };

  class BasecallerRemoteControl : public PacBio::Process::ProcessBaseRemoteControl
  {
      bool abort = false;
      bool color_ = false;

  private:
      const PacBio::IPC::MessageQueueLabel& CtrlQueue(int micOffset)
      {
          switch(micOffset) {
          case 0: return pa_t2b0CommandQueue;
          case 1: return pa_t2b1CommandQueue;
          case 2: return pa_t2b2CommandQueue;
          }
          throw PBException("not supported");
      }
      const PacBio::IPC::MessageQueueLabel& StatQueue(int micOffset)
      {
          switch(micOffset) {
          case 0: return pa_t2b0StatusQueue;
          case 1: return pa_t2b1StatusQueue;
          case 2: return pa_t2b2StatusQueue;
          }
          throw PBException("not supported");
      }

      static constexpr int CtrlPort(int micOffset)
      {
          switch (micOffset)
          {
          case 0:
              return PORT_CTRL_BASECALLER_MIC0;
          case 1:
              return PORT_CTRL_BASECALLER_MIC1;
          case 2:
              return PORT_CTRL_BASECALLER_MIC2;
          default:
              throw PBException("not supported");
          }
      }

      static constexpr int StatPort(int micOffset)
      {
          switch (micOffset)
          {
          case 0:
              return PORT_STAT_BASECALLER_MIC0;
          case 1:
              return PORT_STAT_BASECALLER_MIC1;
          case 2:
              return PORT_STAT_BASECALLER_MIC2;
          default:
              throw PBException("not supported");
          }
      }

  public:
      BasecallerRemoteControl(const BasecallerRemoteControl& rc ) : BasecallerRemoteControl(rc.micOffset_) {}

      BasecallerRemoteControl(int micOffset, std::string host = "localhost" )
              : ProcessBaseRemoteControl(host, CtrlPort(micOffset), StatPort(micOffset))
              , micOffset_(micOffset)
      {
          commandSocket_.SetNoLinger(); // commands will only block for up to 100 milliseconds.
          statusSocket_.SetTimeout(100);
      }

      virtual ~BasecallerRemoteControl()
      {
      }

      void EmitHeartbeat(const std::string& heartBeatData)
      {
          PBLOG_DEBUG << "heartbeat to basecaller" << heartBeatData;
          Announcement msg("heartbeat", heartBeatData);
          commandSocket_.Send(msg);
      }

      void SendMessage(const PacBio::IPC::Message& mesg)
      {
          commandSocket_.Send(mesg);
      }

      void EnableColor(bool flag = true)
      {
          color_ = flag;
      }

      template<typename R,typename P>
      BasecallerStatus::GlobalState GetCurrentState(std::chrono::duration<R,P> timeout )
      {
          auto start = std::chrono::high_resolution_clock::now();
          Json::Reader r;

          LargeHeapMessage mesg;
          while (std::chrono::high_resolution_clock::now() - start < timeout )
          {
              if (abort) throw PBException("External Abort requested");
              statusSocket_.Receive(*mesg);
              if (mesg->IsAvailable())
              {
                  if (mesg->GetName() == "basecaller/status")
                  {
                      BasecallerStatus s;
                      try
                      {
                          s.Load(mesg->GetData());
                      }
                      catch (const std::exception& err)
                      {
                          PBLOG_WARN << "Bad XML or json:" << err.what() << "\n" << mesg->GetData() ;
                      }

                      return s.state;
                  }
                  else
                  {
                      PBLOG_INFO << "Message: " << *mesg ;
                  }
              }
          }
          throw TimeoutException("timeout waiting for status");
      }

      BasecallerStatus::GlobalState GetCurrentState()
      {
          return GetCurrentState(std::chrono::seconds(5));
      }
      virtual void StatusCallback(const BasecallerStatus& s, BasecallerStatus::GlobalState state)
      {
          PBLOG_INFO
          <<  (color_ ? ANSI_COLOR_YELLOW : "")
          << "pa-t2b" << micOffset_ << " "
          << "State: " << s.state().toString() << " (waiting for " << state.toString() << ")"
          << " Time: " << std::setprecision(2) << std::fixed << s.time
          << " tok: " << s.token
          << (color_ ? ANSI_COLOR_DEFAULT : "");
      }
      /// returns seconds spent waiting or throws a TimeoutException
      template<typename R,typename P>
      double WaitFor(BasecallerStatus::GlobalState state, std::chrono::duration<R,P> timeout)
      {
          auto start = std::chrono::high_resolution_clock::now();
          Json::Reader r;

          LargeHeapMessage mesg;
          while (std::chrono::high_resolution_clock::now() - start < timeout )
          {
              if (abort) throw PBException("External Abort requested");
              statusSocket_.Receive(*mesg);
              if (mesg->IsAvailable())
              {
                  if (mesg->GetName() == "basecaller/status")
                  {
                      BasecallerStatus s;
                      try
                      {
                          s.Load(mesg->GetData());
                      }
                      catch (const std::exception& err)
                      {
                          PBLOG_WARN << "Bad XML or json:" << err.what() << "\n" << mesg->GetData() ;
                      }

                      StatusCallback(s,state);

                      if (s.state() == state)
                      {
                          auto secs = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start);
                          return static_cast<double>(secs.count());
                      }
                      if (s.state() == BasecallerStatus::GlobalState::error)
                      {
                          throw PBException("error state entered");
                      }
                  }
                  else
                  {
                      PBLOG_INFO << "Message: " << *mesg ;
                  }
              }
          }
          throw TimeoutException("timeout waiting for " + state.toString());
      }

      double WaitFor(BasecallerStatus::GlobalState state)
      {
          return WaitFor(state,std::chrono::milliseconds(10000));
      }

      void Start()
      {
          Send("start");
      }

      void Stop()
      {
          Send("stop");
      }

      void FeedWatchdog()
      {
          Send("feedWatchdog", "123");
      }

      void GetNextStatusMessage(Message& mesg, const char* name,  std::chrono::milliseconds timeout)
      {
          statusSocket_.WaitForNextMessage(mesg, name,timeout);
      }

      void Abort()
      {
          Send("abort");
          //Send("quit");
      }
#if 0
      void ReportRate(double rate)
      {
          Send("config/statusReportRate", std::to_string(rate));
      }
#endif
      double Timestamp()
      {
          return PacBio::Utilities::Time::GetTimeOfDay();
      }

  public:
      void SendJson(const char* name, Json::Value& value)
      {
          Json::FastWriter fastWriter;
          std::string text = fastWriter.write(value);
          Send(name, text);
      }

      void DismissError()
      {
          Send("stop");
      }
      void KillAll()
      {
          abort = true;
      }
 private:
     int micOffset_;
  };
 }
}

#endif //SEQUELACQUISITION_BASECALLERREMOTECONTROL_H
