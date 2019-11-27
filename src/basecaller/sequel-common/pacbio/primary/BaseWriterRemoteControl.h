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
/// \brief  a class used to control the pa-bw service remotely (using IPC messages)
//
// Programmer: Mark Lakata

#ifndef SEQUEL_BASEWRITERREMOTECONTROL_H
#define SEQUEL_BASEWRITERREMOTECONTROL_H

#endif //SEQUEL_BASEWRITERREMOTECONTROL_H

//
// Created by mlakata on 5/26/15.
//

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

namespace PacBio {
using namespace IPC;

namespace Primary {
using namespace std;

class BasewriterStatus : public PacBio::Process::ConfigurationObject
{
public:
    SMART_ENUM(GlobalState,unknown, boot, idle, armed, capture, complete, aborted, checked, error, busy, analyzed, offline);
    ADD_PARAMETER(double,time,0.0);
    ADD_ENUM(GlobalState,state,GlobalState::unknown);
    ADD_PARAMETER(std::string, token, "");
};

  class BaseWriterRemoteControl  : public PacBio::Process::ProcessBaseRemoteControl
  {
      bool abort = false;
      bool color_ = false;

  public:

      class TimeoutException : public std::runtime_error
      {
      public: TimeoutException(const string& msg) : std::runtime_error(msg) { }
      };

  public:
      BaseWriterRemoteControl(string host, bool publish = false)
              : ProcessBaseRemoteControl(MessageQueueLabel("cmd",host,PORT_CTRL_BASEWRITER),
                                         MessageQueueLabel("stat",host,PORT_STAT_BASEWRITER))
      {
          (void)publish;
          commandSocket_.SetNoLinger(); // commands will only block for up to 100 milliseconds.
          statusSocket_.SetTimeout(100);
      }

      virtual ~BaseWriterRemoteControl()
      {
      }

      void EmitHeartbeat(const std::string& heartBeatData)
      {
          PBLOG_DEBUG << "heartbeat to basewriter" << heartBeatData;
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
      BasewriterStatus::GlobalState GetCurrentState(std::chrono::duration<R,P> timeout )
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
                  HandleAnyMessage(*mesg);
                  if (mesg->GetName() == "basewriter/status")
                  {
                      BasewriterStatus s;
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

      BasewriterStatus::GlobalState GetCurrentState()
      {
          return GetCurrentState(std::chrono::seconds(5));
      }
      virtual void StatusCallback(const BasewriterStatus& s, BasewriterStatus::GlobalState state)
      {
          PBLOG_INFO
              <<  (color_ ? ANSI_COLOR_YELLOW : "")
              << "pa-bw "
              << "State: " << s.state().toString() << " (waiting for " << state.toString() << ")"
              << " Time: " << std::setprecision(2) << std::fixed << s.time
              << " tok: "  << s.token()
              << (color_ ? ANSI_COLOR_DEFAULT : "");
      }
      /// returns seconds spent waiting or throws a TimeoutException
      template<typename R,typename P>
      double WaitFor(BasewriterStatus::GlobalState state, std::chrono::duration<R,P> timeout)
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
                  if (mesg->GetName() == "basewriter/status")
                  {
                      BasewriterStatus s;
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
                      if (s.state() == BasewriterStatus::GlobalState::error)
                      {
                          throw PBException("error state entered");
                      }
                  }
                  else
                  {
                      PBLOG_INFO << "Message: " << *mesg ;
                  }
                  HandleAnyMessage(*mesg);
              }
          }
          throw TimeoutException("timeout waiting for " + state.toString());
      }

      double WaitFor(BasewriterStatus::GlobalState state)
      {
          return WaitFor(state,std::chrono::milliseconds(10000));
      }

      int cameraConfigKey = 0;
      string instrumentName = "unknown";
      string chipId = "??";
      string xml = "<this>{ \" : } } }</this>"; // xml attempting to break JSON
#if 0
      unique_ptr<SequelROI> roi;

      void SetPhysicalOffset(uint32_t row0, uint32_t col0)
      {
          SequelSensorROI sensorROI(row0,col0,roi->Sensor().PhysicalRows(), roi->Sensor().PhysicalCols());
          unique_ptr<SequelROI> roix(new SequelROI(*(roi.get()),sensorROI));
          roi = std::move(roix);
      }
      void SetROI(uint32_t rowMin0, uint32_t colMin0, uint32_t rowSize0, uint32_t colSize0)
      {
          unique_ptr<SequelROI> roix(new SequelROI(rowMin0, colMin0, rowSize0, colSize0, roi->Sensor()));
          roi = std::move(roix);
      }
#endif
      static PacBio::IPC::Announcement SetupAnnouncement(const std::string& bazfilename,
                                                         uint32_t numZmws,
                                                         const std::string& chipLayoutName // ugh, I don't like this design
      )
      {
          Json::Value value;
          value["bazfile"] = bazfilename;
          value["numzmws"] = numZmws;
          value["basecallerVersion"] = "1.2.3.4"; //fixme
          value["framerate"] = 99.9; //fixme
          value["basecaller"] = "{}";
          value["chipLayoutName"] = chipLayoutName;
          value["chipClass"] = GetPrimaryConfig().chipClass().toString();
//          value["cameraConfigKey"] = cameraConfigKey;
//          value["instrumentName"] = instrumentName;
//          value["chipId"] = chipId;
//          value["exposure"] = 0.01;
//          value["acquisitionXML"] = xml;
          Json::FastWriter fastWriter;
          Announcement mesg("setup",fastWriter.write(value));
          return mesg;
      }

      void Setup(const std::string& filename, uint32_t numZmws, const std::string& chipLayoutName)
      {
          commandSocket_.Send(SetupAnnouncement(filename,numZmws, chipLayoutName));
      }

      void Start()
      {
          Send("start");
      }

      void Stop()
      {
          Send("stop");
      }

      void SendReadBufferDeed(const TrancheTitle& title)
      {
          PBLOG_TRACE << "readbuffer deed " << title.ZmwIndex();
          Deed deed("readbufferdeed", title);
          commandSocket_.Send(deed);
      }

      void GetNextStatusMessage(Message& mesg, const char* name,  std::chrono::milliseconds timeout)
      {
          statusSocket_.WaitForNextMessage(mesg, name,timeout);
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

      void DismissError()
      {
          Send("dismiss");
      }

      void KillAll()
      {
          abort = true;
      }
  };

}} //namespace
