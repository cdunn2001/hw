// Copyright (c) 2010, Pacific Biosciences of California, Inc.
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
/// \brief  implementation of a Remote Control, to control functionality of the pa-acq process.
//
// Programmer: Mark Lakata
#ifndef _PA_COMMON_ACQUISITION_REMOTE_CONTROL_H_
#define _PA_COMMON_ACQUISITION_REMOTE_CONTROL_H_

#include <chrono>
#include <iostream>
#include <iomanip>
#include <sstream>

#include <json/json.h>

#include <pacbio/ipc/Message.h>
#include <pacbio/ipc/MessageQueue.h>
#include <pacbio/ipc/JSON.h>
#include <pacbio/process/ConfigurationBase.h>
#include <pacbio/logging/Logger.h>
#include <pacbio/process/ProcessBaseRemoteControl.h>
#include <pacbio/text/String.h>
#include <pacbio/smrtdata/Readout.h>
#include <pacbio/smrtdata/MetricsVerbosity.h>
#include <pacbio/utilities/SmartEnum.h>

#include <pacbio/primary/ChipClass.h>
#include <pacbio/primary/EventObject.h>
#include <pacbio/primary/ipc_config.h>
#include <pacbio/primary/SequelROI.h>
#include <pacbio/primary/Acquisition_Setup.h>
#include <pacbio/primary/LiveViewFrame.h>
#include <pacbio/primary/AcquisitionTransmitConfig.h>
#include <pacbio/primary/PrimaryConfig.h>

#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_DEFAULT "\033[39m"

namespace PacBio {
using namespace IPC;

namespace Primary {

 ///
 /// A class who's sole existence is to unprotect the protected Receive() method
 /// to be able to read LiveView messages, which are not part of the IPC message framework.
 ///
 class SuperSubscriber :
         public PacBio::IPC::MessageSocketSubscriber
 {
 public:
     /// constructor that simply forwards its args to the baseclass
     SuperSubscriber(const PacBio::IPC::MessageQueueLabel& name)
             : PacBio::IPC::MessageSocketSubscriber(name)
     {
     }

     /// unprotect the Receive methods
     using MessageSocketSubscriber::Receive;
 };

 class AuroraWarning : public PacBio::Process::ConfigurationObject
 {
     ADD_PARAMETER(double,time,0.0);
     ADD_PARAMETER(uint64_t,rxGthCrcErrCount,0);
     ADD_PARAMETER(uint64_t,rxGthSoftErrCount,0);
     ADD_PARAMETER(uint64_t,rxGthFramingErrCount,0);
     ADD_PARAMETER(uint64_t,rxIncompleteFrameCount,0);
     ADD_PARAMETER(uint64_t,rxOversizedFrameCount,0);
     ADD_PARAMETER(uint64_t,rxOverflowPduCount,0);
     ADD_PARAMETER(uint64_t,rxOverflowChunkCount,0);
     ADD_PARAMETER(uint64_t,rxFrameErrorCount,0);
     ADD_PARAMETER(uint64_t,rxFrameSevereErrorCount,0);
     ADD_PARAMETER(uint64_t,rxMagicDetErrCount,0);
     ADD_PARAMETER(uint64_t,rxChunkFrameErrorCount,0);
     ADD_PARAMETER(uint64_t,rxTileStarvedCount,0);
     ADD_PARAMETER(std::string,token,"");
 };




 /// member of PaAcqStatus pipeline array.
/// Defines one stage of the pipeline, and gives the socket URL(s) for that stage.
/// example: stage:"pa-acq", statusPorts:["pac-host:46602"]
 class PaAcqPipelineStage : public PacBio::Process::ConfigurationObject
 {
     ADD_PARAMETER(std::string, stage, "");
     ADD_ARRAY(std::string, statusPorts);
 };

/// member of PaAcqStatus storageCapacity array
/// example: partiion:"/data/pa", bytesRemaining:1234, hoursRemaining:12.34
 class PaAcqStorageCapacity : public PacBio::Process::ConfigurationObject
 {
     ADD_PARAMETER(std::string, partition, "");
     ADD_PARAMETER(double, bytesRemaining, 0);
     ADD_PARAMETER(double, hoursRemaining, 0);
 };

 class AcquisitionStartConfig : public PacBio::Process::ConfigurationObject
 {
     ADD_PARAMETER(std::string, token, "");
     ADD_PARAMETER(double, timestamp, 0.0);
     ADD_PARAMETER(uint64_t, startFrame, 0);
 };

 /// types of calibration supported by the API. Not all calibrations are supported in software.

 /// configuration for the calibration/setup comment
 class CalibrationSetup : public PacBio::Process::ConfigurationObject
 {
     CONF_OBJ_SUPPORT_COPY(CalibrationSetup);

     ADD_PARAMETER(std::string, token, "");
     ADD_PARAMETER(double, timestamp, 0.0);
     ADD_PARAMETER(uint32_t, cameraConfigKey, 0);
     ADD_PARAMETER(std::string, chipId, "");
     ADD_ENUM(CalType, type, CalType::dark);
     ADD_PARAMETER(double, exposure, 0);
     ADD_PARAMETER(double, photoelectronSensitivity, 0);
     ADD_PARAMETER(double, frameRate, 0);
     ADD_PARAMETER(std::string, calFileName, "");
     ADD_PARAMETER(std::string, datestamp, "");
     ADD_PARAMETER(std::string, movieDirName, "");
     ADD_PARAMETER(std::string, darkFrameCalFile, "n/a this must be set for loading calibration");

   public:
     std::vector<Kernel> Psfs() const
     {
       std::vector<Kernel> psfs;

       const Json::Value& jsonPsfs = Json()["psfs"];
       if (jsonPsfs.size() == 0)
       {
           PBLOG_WARN << "acquisition/setup JSON does not contain PSFs. Unity PSFs will be used.";
           double g[5][5] = {
             {0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0},
             {0, 0, 1, 0, 0},
             {0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0}};

           double r[5][5] = {
             {0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0},
             {0, 0, 1, 0, 0},
             {0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0}};

           boost::multi_array_ref<double, 2> mg((double*) g, boost::extents[5][5]);
           Kernel kg(mg);
           psfs.push_back(kg);

           boost::multi_array_ref<double, 2> mr((double*) r, boost::extents[5][5]);
           Kernel kr(mr);
           psfs.push_back(kr);
        }
        else
        {
           for (const auto& jsonPsf : jsonPsfs)
           {
               Kernel k(jsonPsf);
               psfs.push_back(k);
           }
        }
        return psfs;
     }

 };

 class CalibrationStart : public PacBio::Process::ConfigurationObject
 {
     CONF_OBJ_SUPPORT_COPY(CalibrationStart);

     ADD_PARAMETER(std::string, token, "");
     ADD_PARAMETER(double, timestamp, 0.0);
     ADD_PARAMETER(uint64_t, numFrames, 0);
     ADD_PARAMETER(uint64_t, startFrame, 0);
     ADD_PARAMETER(double, exposure, 0);
     ADD_PARAMETER(uint32_t, exposureNum, 0);
     ADD_PARAMETER(std::string, moviePathName, "");
 };


namespace AcquisitionStatus {
SMART_ENUM(GlobalState, unknown, boot, idle, armed, capture, complete, aborted, checked, error, busy, analyzed,
            offline);
SMART_ENUM(AuroraStatus, up, down, unknown);
}

/// broadcast from pa-acq as payload of the acquisition/status messages
class PaAcqStatus :  public PacBio::Process::ConfigurationObject
{
    ADD_PARAMETER(double,time,0); // is it double?
    ADD_ENUM(AcquisitionStatus::GlobalState,state,AcquisitionStatus::GlobalState::unknown);
    ADD_PARAMETER(uint64_t,currentFrame,0);
    ADD_PARAMETER(uint64_t,currentFrameIndex,0);
    ADD_PARAMETER(uint64_t,movieFrame,0);
    ADD_PARAMETER(double,frameRate,0);
    ADD_PARAMETER(double,lineRate,0);
    ADD_PARAMETER(std::string,token,"");
    ADD_PARAMETER(uint32_t,cameraConfigKey,0);
    ADD_ENUM(AcquisitionStatus::AuroraStatus,auroraStatus,AcquisitionStatus::AuroraStatus::unknown);
    ADD_PARAMETER(uint64_t,transmitFrame,0);
    ADD_PARAMETER(int32_t,phiCount,0);
    ADD_PARAMETER(double,freeTilePercent,0.0);
    ADD_ARRAY(PaAcqPipelineStage,pipeline);
    ADD_ARRAY(PaAcqStorageCapacity,storageCapacity);
    ADD_ENUM(PacBio::SmrtData::Readout,readout,PacBio::SmrtData::Readout::BASES);
    ADD_ENUM(PacBio::SmrtData::MetricsVerbosity, metricsVerbosity, PacBio::SmrtData::MetricsVerbosity::MINIMAL);
};

  /// the object that controls AcqProcess in pa-acq, via remote IPC calls.
  class AcquisitionRemoteControl : public PacBio::Process::ProcessBaseRemoteControl
  {
  public:

  public:
      /// constructor. The host argument is the name or IP address of the remote AcqProcess process.
      AcquisitionRemoteControl(std::string host = "localhost")
         : ProcessBaseRemoteControl(host,PORT_CTRL_ACQUISITION,PORT_STAT_ACQUISITION)
         , token("")
         , liveViewQueue_("liveview",host,PORT_LIVEVIEW_ACQUISITION)
           , abort_(false)
      , color_(false)

      {
          commandSocket_.SetNoLinger(); // commands will only block for up to 100 milliseconds.
          statusSocket_.SetTimeout(100);
      }

      ~AcquisitionRemoteControl()
      {
      }

      /// overrides the base implementation to send the command to one of two possible sockets.
      virtual void SendBase(const PacBio::IPC::Message& message)
      {
          PBLOG_DEBUG << "Send(" << message << ")" ;
          commandSocket_.Send(message);
      }

      /// sends a rendered string of the JSON value, along with the given name, to the AcqProcess.
      void Send(const char* name, const Json::Value& value)
      {
          Json::StreamWriterBuilder b;
          std::string text = Json::writeString(b, value);
          Send(name, text);
      }

      /// import the other Send methods too.
      using ProcessBaseRemoteControl::Send;

      /// enable coloring the text output to make it easier to interpret was is in-band and out-of-band
      /// messages.
      void EnableColor(bool flag = true)
      {
        color_ = flag;
      }

      /// create a simple LiveView client that simply prints out a summary for each live view frame that is received
      void LiveViewClient()
      {
          SuperSubscriber liveview(liveViewQueue_);
          while (!abort_)
          {
              std::vector <uint8_t> buffer(0x10001c);
              PBLOG_TRACE << "waiting for frame" ;
              size_t size = liveview.Receive(&buffer[0], buffer.size());
              if (size > 0)
              {
                  Message* msg = reinterpret_cast<Message*>(&buffer[0]);
                  auto lvf = msg->GetDataPointer<LiveViewFrame>();
                  PBLOG_INFO << "Got liveview frame: "<< *msg ;
#if 1
                  std::stringstream s;
                  for(uint32_t i=0;i<size;i++)
                  {
                      if (i%16 == 0) s << " raw ";
                      s << std::hex << std::setw(2) << (int)buffer[i] << std::dec << " ";
                      if (i % 16 == 15) s << std::endl;
                  }
                  PBLOG_DEBUG << s.str();
#endif
                  PBLOG_INFO << "offset: " << lvf->rowMin <<"," << lvf->columnMin
                  << " size:" <<lvf->rows << "," << lvf->columns
                  << " stride:" << lvf->Stride
                  << " pixels:" << lvf->NumPixels ;
                  std::stringstream t;
                  for(uint32_t i=0;i<lvf->NumPixels;i++)
                  {
                      if (i%16 == 0) t << " lvf ";
                      t << std::hex << std::setw(4) << (int)lvf->pixels.raw[i] << std::dec << " ";
                      if (i % 16 == 15) t<< std::endl;
                  }
                  PBLOG_DEBUG << t.str();
              }
          }
      }

  private:
      Json::Value latestErrorStatistics_;

      void HandleStatusMessage(const Message& mesg)
      {
          if (mesg.GetName() == "acquisition/warning")
          {
              latestErrorStatistics_ = ParseJSON(mesg.GetData());
          }
          HandleAnyMessage(mesg);
      }

/// get the current state of PaqAcq, waiting for the next published status message
  public:
      template<typename R,typename P>
      AcquisitionStatus::GlobalState GetCurrentState(std::chrono::duration<R,P> timeout )
      {
          auto start = std::chrono::high_resolution_clock::now();
          Json::Reader r;
          auto elapsed = std::chrono::duration_cast<std::chrono::duration<R,P> >(start - start);

          //std::unique_ptr<LargeMessage> mesg(new LargeMessage);
          LargeHeapMessage mesg;
          while ((elapsed = std::chrono::duration_cast<std::chrono::duration<R,P> >(std::chrono::high_resolution_clock::now() - start)) < timeout )
          {
              if (abort_) throw PBException("External Abort requested");
              statusSocket_.Receive(*mesg);
              if (mesg->IsAvailable())
              {
                  HandleStatusMessage(*mesg);
                  if (mesg->GetName() == "acquisition/status")
                  {
                      PaAcqStatus s;
                      try
                      {
                          s.Load(mesg->GetData());
                      }
                      catch (const std::exception& err)
                      {
                          PBLOG_WARN << "Bad XML or json:" << err.what() << "\n" << mesg->GetData() ;
                      }
                      return s.state();

                  }
                  else
                  {
                      PBLOG_INFO << "Message: " << *mesg ;
                  }
              }
          }
          throw TimeoutException("timeout waiting for status, " + std::to_string(timeout.count()) + " elapsed:" + std::to_string(elapsed.count()));
      }

      /// get current state, using default timeout of 5 seconds
      AcquisitionStatus::GlobalState GetCurrentState()
      {
          return GetCurrentState(std::chrono::seconds(50));
      }

      /// this method is called on every received status message. this implementation
      /// prints out a human readable log entry on each status message.
      virtual void StatusCallback(const PaAcqStatus& s, AcquisitionStatus::GlobalState waitingForState)
      {
          PBLOG_INFO
              <<  (color_ ? ANSI_COLOR_YELLOW : "")
                      << "pa-acq "
                      << "State: " << s.state().toString()
                      << std::string((waitingForState != AcquisitionStatus::GlobalState::unknown) ? (" (waiting for " + waitingForState.toString() + ")") : "")
                      << " Time: " << std::setprecision(2) << std::fixed << s.time()
                      << " CurrentFrame:" << s.currentFrame()
                      << " MovieFrame:" << s.movieFrame()
                      << " XmitFrame:" << s.transmitFrame()
                      << " " << s.frameRate() << "fps"
                      << " aurora:" << s.auroraStatus().toString()
                      << (color_ ? ANSI_COLOR_DEFAULT : "")
                      << " tok:" << s.token()
                      ;
      }

      /// waits for the requested state. Returns seconds spent waiting or throws a TimeoutException
      template<typename R,typename P>
      double WaitFor(AcquisitionStatus::GlobalState state, std::chrono::duration<R,P> timeout)
      {
          auto start = std::chrono::high_resolution_clock::now();

          bool error =false;
          LargeHeapMessage mesg;
          while (std::chrono::high_resolution_clock::now() - start < timeout )
          {
              if (abort_) throw PBException("External Abort requested");
              PBLOG_DEBUG << "statusSocket_.Receive(*mesg);";
              statusSocket_.Receive(*mesg);
              if (mesg->IsAvailable())
              {
                  HandleStatusMessage(*mesg);
                  if (mesg->GetName() == "acquisition/status")
                  {
                      PBLOG_DEBUG << mesg->GetData();
                      PaAcqStatus s;
                      try
                      {
                          s.Load(mesg->GetData());
                      }
                      catch (const std::exception& err)
                      {
                          PBLOG_WARN << "Bad XML or json:" << err.what() << "\n" << mesg->GetData();
                      }

                      StatusCallback(s, state);

                      if (s.state() == state)
                      {
                          auto secs = std::chrono::duration_cast<std::chrono::seconds>(
                                  std::chrono::high_resolution_clock::now() - start);
                          return static_cast<double>(secs.count());
                      }
                      if (s.state() == AcquisitionStatus::GlobalState::error)
                      {
                          start = std::chrono::high_resolution_clock::now();
                          timeout = std::chrono::duration_cast<std::chrono::duration<R, P> >(
                                  std::chrono::milliseconds(200));
                          error = true;
                      }
                  }
                  else
                  {
                      PBLOG_INFO << *mesg;
                  }
              }
          }
          if (error)
          {
              throw PBException("error state entered during WaitFor " + state.toString());
          }

          auto secs = std::chrono::duration_cast<std::chrono::seconds>(
                  std::chrono::high_resolution_clock::now() - start);
          PBLOG_ERROR << "Elapsed time for waiting for state " << state.toString() << " was "
              << static_cast<double>(secs.count()) << " secs";
          throw TimeoutException("timeout waiting for " + state.toString());
      }

      /// wait for the requested state, for no more than 10 seconds.
      /// The time to attain that state is returned, or a TimeoutException is thrown.
      double WaitFor(AcquisitionStatus::GlobalState state)
      {
          return WaitFor(state,std::chrono::milliseconds(10000));
      }

      bool roiOldFormat = false;

      /// sets up an acquisition. Note that several fields must be set as member variables before
      /// calling this function (see below for a list). The acquisition can acquire an HDF5 file and/or a BAZ file.

      AcquisitionRemoteControl& SetExpectedFrameRate(double rate)
      {
          expectedFrameRate_ = rate;
          return *this;
      }

      AcquisitionRemoteControl& SetPhotoelectronSensitivity(double sensitivity)
      {
          photoelectronSensitivity_ = sensitivity;
          return *this;
      }

      Json::Value GetCurrentErrorStatisticsJSON() const
      {
          return latestErrorStatistics_;
      }
      uint32_t TotalRxErrors() const
      {
          uint32_t sum = 0;
          for( Json::ValueIterator itr = latestErrorStatistics_.begin() ; itr != latestErrorStatistics_.end() ; itr++ )
          {
              if (PacBio::Text::String::EndsWith(itr.key().asString(),"Count"))
              {
                  sum += itr->asUInt();
              }
          }
          return sum;
      }
      void AcquisitionSetup( const PacBio::Primary::Acquisition::Setup& setupObject)
      {
          PacBio::Primary::Acquisition::Setup mutableSetup(setupObject);
          mutableSetup.Sanitize();
          Send("acquisition/setup", mutableSetup.Json());
      }

#if 0
  private:
      void AcquisitionSetup(const std::string& hdf5fileOutput, const std::string& bazfilename, uint64_t frames, bool dryRun)
      {
          Json::Value setup = AcquisitionSetupRender(hdf5fileOutput, bazfilename, frames, dryRun);
          Send("acquisition/setup", setup);
      }
  public:
      Json::Value AcquisitionSetupRender(const std::string& hdf5fileOutput, const std::string& bazfilename, uint64_t frames, bool dryRun)
      {
          if (expectedFrameRate_ == 0)
          {
              throw PBException("SetExpectedFrameRate is now required to be called before  AcquisitionSetupRender");
          }
          Json::Value value;
          value["hd5output"] = hdf5fileOutput;
          value["bazfile"] = bazfilename;
          value["cameraConfigKey"] = cameraConfigKey;
          value["instrumentName"] = instrumentName;
          value["chipId"] = chipId;
//        value["pane"] = 1;
//        value["paneSetId"] = 0;
          value["numFrames"] = frames;
          value["darkFrameCalFile"] = darkFrameCalFile;
          value["token"] = token;
          value["dryrun"] = dryRun;
          value["expectedFrameRate"] = expectedFrameRate_;
          value["exposure"] = 1/expectedFrameRate_;
          value["photoelectronSensitivity"] = photoelectronSensitivity_;

          value["metricsVerbosity"] = PacBio::Text::String::ToLower(metricsVerbosity.toString());
          value["readout"] = PacBio::Text::String::ToLower(readout.toString());
          value["chipLayoutName"] = chipLayoutName_;

          std::unique_ptr<ChipLayout> layout(ChipLayout::Factory(chipLayoutName_));
          chipClass_ = layout->GetChipClass();

          if (!refDwsSnr.isNull()) value["refDwsSnr"] = refDwsSnr;
          if (!refSpectrum.isNull()) value["refSpectrum"] = refSpectrum;

          expectedFrameRate_ = 0; // one shot setup
          FormatXML();

          if (!roiOldFormat)
          {
              if ( bazROI && hdf5ROI && (bazROI->SensorROI() != hdf5ROI->SensorROI()) )
              {
                  throw PBException("BazROI and SensorROI have different SensorROIs");
              }

              Json::Value& roiMetaData = value["roiMetaData"];

              if (hdf5ROI) roiMetaData["traceFilePixelROI"] = hdf5ROI->GetJson();
              if (bazROI)  roiMetaData["sequencingPixelROI"] = bazROI->GetJson();

              roiMetaData["remapVroiOption"] = remapVroiOption;
          }

          value["acquisitionXML"] = xml;

          if (analogs.isArray())
          {
              value["analogs"] = analogs;
          }

          if (psfs.isArray())
          {
              value["psfs"] = psfs;
          }
          else
          {
              value["crosstalkFilter"] = crosstalkFilter;
          }

          PBLOG_DEBUG << "acquisition/setup:" << PacBio::IPC::RenderJSON(value);

          return value;
      }
#endif
  public:
      /// starts acquiring frames
      /// \param acqStart - a mutuable reference
      void AcquisitionStart(AcquisitionStartConfig& acqStart)
      {
          if (acqStart.token() == "" &&
              token != "") acqStart.token = token;
          acqStart.timestamp = Timestamp();
          Send("acquisition/start", acqStart.Json());
      }

      /// stops an acquisition prematurely.
      void AcquisitionStop()
      {
          Send("acquisition/stop", "");
      }

      /// stops transmitting a movie (from sensor simulator)
      void TransmissionStop()
      {
          Send("transmitMovieStop", "");
      }

      /// changes the rate of status reports. The default is 1 status report/second.
      void ReportRate(double rate)
      {
          Send("config/statusReportRate", std::to_string(rate));
      }

      /// returns Unix epoch timestamp
      static double Timestamp()
      {
          return PacBio::Utilities::Time::GetTimeOfDay();
      }

  public:

      /// send the event object to pa-acq
      void ForwardEvent(const EventObject& eo)
      {
          Send("acquisition/event", eo.Json());
      }

      void ForwardEvent(const Json::Value& event)
      {
          Send("acquisition/event", event);
      }

      /// configure live view on pa-acq. The 2 parameters are the frame rate
      /// and the ROI (given in absolute pixel coordinates).
      void LiveViewConfig(double rate, const SequelRectangularROI& roi)
      {
          Json::Value value;
          // value["token"] = token;
          value["viewportOffset"]["row"] = roi.RelativeRowPixelMin();
          value["viewportOffset"]["col"] = roi.RelativeColPixelMin();
          value["viewportSize"]["rows"] = roi.NumPixelRows();
          value["viewportSize"]["cols"] = roi.NumPixelCols();
          value["rate"] = rate;
          Send("liveview/setup", value);
      }

      /// start live view on pa-acq
      void LiveViewStart()
      {
          Send("liveview/start");
      }

      /// stop live view on pa-acq
      void LiveViewStop()
      {
          Send("liveview/stop");
      }

      /// start a calibration at a particular exposure and exposure number. The calibration
      /// will start on or after `startFrame` is received.
      void CalStart(CalibrationStart& calStart)
      {
          calStart.timestamp = Timestamp();
          if (calStart.token() == "" && token != "") calStart.token = token;
          Send("calibration/start", calStart.Json());
      }

      /// configure a calibration type. This will change the state from "idle" to "armed"
      /// Side effect is that the calSetup object is modified with a timestamp
      void CalConfig(CalibrationSetup& calSetup)
      {
          calSetup.timestamp = Timestamp();
          if (calSetup.token() == "" && token != "") calSetup.token = token;
          Send("calibration/setup", calSetup.Json());
      }

      /// stop calibration
      void CalStop(const std::string& token0)
      {
          Json::Value value;
          value["token"] = token0;
          Send("calibration/stop", value);
      }

      /// Start a calibration analyze step
      void CalAnalyze(const std::string& token0)
      {
          Json::Value value;
          value["token"] = token0;
          Send("calibration/analyze", value);
      }

      /// Reset the calibration to unity, ie no dark frame subtraction, no scaling
      void CalReset()
      {
          Send("calibration/reset");
      }

      /// load calibration file
      void CalLoad(const std::string& filename)
      {
          Json::Value value;
          value["filename"] = filename;
          Send("calibration/load", value);
      }

      //// Dismisses an error on pa-acq. On success, the pa-acq status will go to "idle"
      void DismissError()
      {
          PBLOG_DEBUG << "DismissError() called";
          LargeHeapMessage mesg;
          while(true)
          {
              statusSocket_.Receive(*mesg);
              if (mesg->IsAvailable())
              {
                  PBLOG_DEBUG << "DismissError() mesg" << *mesg;
                  HandleStatusMessage(*mesg);
              }
              else
              {
                  PBLOG_DEBUG << "DismissError() no more messages";
                  break;
              }
          }
          PBLOG_DEBUG << "DismissError() Send(\"config/dismiss\")";
          Send("config/dismiss");
          int attempts = 0;
          while(attempts++ < 5)
          {
              auto state = GetCurrentState();
              PBLOG_DEBUG << "DismissError() state is " << state.toString() << " attempt:" << attempts;
              if (state == AcquisitionStatus::GlobalState::idle) return;
          }
          throw PBException("Can't clear error mode");
      }

      /// Stops the remote control (asynchronously) by break out of any waiting loops.
      void KillAll()
      {
          abort_ = true;
      }

      /// Starts the transmission of a file from pa-acq (in sensor simulator mode) for the given number
      /// of frames and at the frame rate. The filename must be local to the pa-acq process, it may not be local
      /// machine where the RemoteControl is used (unless the remote control and pa-acq are on the same machine).
      void TransmitMovie(const TransmitConfig& config)
      {
          if (config.startFrame == TransmitConfig::READ_FRAME_INDEX_FROM_FILE)
          {
              // to difficult to do the right thing here. Just fake it and assume the file starts with 0.
              currentTransmitFrameIndex_ = 0;
          }
          else if (config.startFrame == TransmitConfig::CONTINUE_FRAME_INDICES)
          {
              // do nothing
          }
          else
          {
              currentTransmitFrameIndex_ = config.startFrame;
          }

          Send("transmitMovieSim", config.Json());
          currentTransmitFrameIndex_ += config.frames;
      }


      /// Waits for pa-acq to start a simulated transmission. This event can only happen once in the lifetime of a pa-acq
      /// instance, when the very first frame is transmitted.   This method is only intended to be used in
      /// functional tests where pa-acq is relaunched for each simulated transmission.
      /// To reiterate, this only waits for the simulated transmission to start, it does NOT wait for received
      /// data (either simulated or not). This is not intended for use outside of functional tests.
      /// \throw std::exception if there is a timeout (30 seconds)

     template<typename R,typename P>
     void WaitForTransmissionToStart(std::chrono::duration<R,P> timeout = std::chrono::milliseconds(30000))
      {
          LargeHeapMessage msg;
          statusSocket_.WaitForNextMessage(*msg,"transmission/start",timeout);
      }

      /// Set the trace injection file for the movie.
      void SetInjectionFile(const std::string& filename, const std::string& roiString)
      {
          Json::Value value;
          value["filename"] = filename;
          value["roi"] = roiString;
          Send("config/injectionfile", value);
      }

      /// Performs a pipelinetest on pa-acq
      void PipelineTest()
      {
          Send("pipelinetest");
      }

      /// Read the crosstalk coefficients from a ASCII test file, and send them to the Fpga.
      /// The file ASCII format is
      ///   numRows numCols coefficient1 coefficient2 .... coefficientN
      /// where N = numRows*numCols and the coefficients are major order is row, then minor order is column

      void SetFilterCoefficientsFromFile(const std::string& filename)
      {
          Json::Value crosstalkFilter;
          std::ifstream input(filename);
          int rowSize;
          int colSize;
          input >> rowSize >> colSize;
          if (rowSize >=1 && rowSize <=7 && colSize >=1 && colSize <= 7)
          {
              crosstalkFilter.resize(rowSize);
              for(int row=0;row<rowSize;row++)
              {
                  crosstalkFilter[row].resize(colSize);
                  for(int col =0;col<colSize;col++)
                  {
                      double f;
                      input >> f;
                      crosstalkFilter[row][col] = f;
                  }
              }
          }
          PBLOG_DEBUG << crosstalkFilter ;
          // the pa-acq does not support the "filterCoefficients" IPC command, but if it did
          // this is where it would be sent.
          //  Send("filterCoefficients",crosstalkFilter);
      }

      void PulseAuroraPolarity(double width, double intvl, uint32_t count)
      {
          Json::Value v;
          v["pulseWidth"] = width;
          v["pulseIntvl"] = intvl;
          v["pulseCount"] = count;
          Send("config/pulseAuroraPolarity", v );
      }

      void SensorFpgaWrite(uint16_t addr, uint16_t data)
      {
          Json::Value v;
          v["addr"] = addr;
          v["data"] = data;
          Send("config/sensorFpgaWrite", v );
      }

      void PurgeChunkFIFO_NonBlocking()
      {
          Send("config/purgeChunkFifo","");
      }

      void PurgeChunkFIFO()
      {
          PBLOG_NOTICE << "AcquisitionRemoteControl::PurgeChunkFIFO started";
          // guarding against calling purge when a chunk flush is, or could become, pending, by enforcing idle at start
          {
              auto t = WaitFor(AcquisitionStatus::GlobalState::idle, std::chrono::seconds(3));
              PBLOG_NOTICE << "PurgeChunkFIFO entry idle took " << t << " seconds";
          }
          PurgeChunkFIFO_NonBlocking();
          {
              auto t = WaitFor(AcquisitionStatus::GlobalState::busy, std::chrono::seconds(3));
              PBLOG_NOTICE << "PurgeChunkFIFO busy took " << t << " seconds";
          }
          PBLOG_NOTICE << "PurgeChunkFIFO flush sent" << std::endl;
      }

      int64_t GetLastFrameIndex() const
      {
          return currentTransmitFrameIndex_;
      }
  public:
      /// to do. All these public variables should be removed and replaced by
      /// just a private AcquisitionSetup object.
      std::string token = "";

  private:
      MessageQueueLabel liveViewQueue_;
      bool abort_ = false;
      bool color_ = false;
      bool setTraceMetadata_ = false;
      double expectedFrameRate_ = 0;
      double photoelectronSensitivity_ = 1.0;
      ChipClass chipClass_ = ChipClass::DONT_CARE;
      int64_t currentTransmitFrameIndex_;
  };

  class PowerUserAcquisitionRemoteControl : public AcquisitionRemoteControl
  {
  public:
      using AcquisitionRemoteControl::Send;
  };
 }
}

#endif //_PA_ACQ_CLI_ACQUISITIONPROXY_H_
