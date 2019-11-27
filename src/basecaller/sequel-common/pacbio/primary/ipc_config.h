// Copyright (c) 2017, Pacific Biosciences of California, Inc.
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
// Description:
/// \brief global definitions that define the IPC port numbers for all processes in Primary Analysis

#ifndef _PRIMARY_IPC_CONFIG_H_
#define _PRIMARY_IPC_CONFIG_H_

#include <string>
#include <pacbio/ipc/MessageQueue.h>

namespace PacBio
{
 namespace Primary
 {
  #ifndef PORT_BASE_ACQUISITION_DEFAULT
  #define PORT_BASE_ACQUISITION_DEFAULT 46600
//  #define PORT_BASE_ACQUISITION_DEFAULT 23600

  #endif

  const std::string pac_host{"pac-host"};

  const uint16_t PORT_BASE_ACQUISITION = PORT_BASE_ACQUISITION_DEFAULT;
  const uint16_t PORT_CTRL_ACQUISITION = PORT_BASE_ACQUISITION + 0;
  const uint16_t PORT_STAT_ACQUISITION = PORT_BASE_ACQUISITION + 1;
  const uint16_t PORT_HTTP_ACQUISITION = PORT_BASE_ACQUISITION + 2;
  const uint16_t PORT_LIVEVIEW_ACQUISITION = PORT_BASE_ACQUISITION + 4;
  const uint16_t PORT_SCIF_ACQUISITION = PORT_BASE_ACQUISITION + 6;

  const uint16_t PORT_BASE_BASECALLER_MIC0  = PORT_BASE_ACQUISITION_DEFAULT + 10;
  const uint16_t PORT_CTRL_BASECALLER_MIC0  = PORT_BASE_BASECALLER_MIC0 + 0;
  const uint16_t PORT_STAT_BASECALLER_MIC0  = PORT_BASE_BASECALLER_MIC0 + 1;
  const uint16_t PORT_HTTP_BASECALLER_MIC0  = PORT_BASE_BASECALLER_MIC0 + 2;
  const uint16_t PORT_CTRL7_BASECALLER_MIC0 = PORT_BASE_BASECALLER_MIC0 + 7;

  const uint16_t PORT_BASE_BASEWRITER  = PORT_BASE_ACQUISITION_DEFAULT + 20;
  const uint16_t PORT_CTRL_BASEWRITER  = PORT_BASE_BASEWRITER + 0;
  const uint16_t PORT_STAT_BASEWRITER  = PORT_BASE_BASEWRITER + 1;
  const uint16_t PORT_HTTP_BASEWRITER  = PORT_BASE_BASEWRITER + 2;
  const uint16_t PORT_SCIF_BASEWRITER  = PORT_BASE_BASEWRITER + 6;

  const uint16_t PORT_BASE_WEBSERVICE  = PORT_BASE_ACQUISITION_DEFAULT + 30;
  const uint16_t PORT_CTRL_WEBSERVICE  = PORT_BASE_WEBSERVICE + 0;
  const uint16_t PORT_STAT_WEBSERVICE  = PORT_BASE_WEBSERVICE + 1;
  const uint16_t PORT_HTTP_WEBSERVICE  = PORT_BASE_WEBSERVICE + 2;

  const uint16_t PORT_BASE_BASECALLER_MIC1  = PORT_BASE_ACQUISITION_DEFAULT + 40;
  const uint16_t PORT_CTRL_BASECALLER_MIC1  = PORT_BASE_BASECALLER_MIC1 + 0;
  const uint16_t PORT_STAT_BASECALLER_MIC1  = PORT_BASE_BASECALLER_MIC1 + 1;
  const uint16_t PORT_HTTP_BASECALLER_MIC1  = PORT_BASE_BASECALLER_MIC1 + 2;
  const uint16_t PORT_CTRL7_BASECALLER_MIC1 = PORT_BASE_BASECALLER_MIC1 + 7;

  const uint16_t PORT_BASE_BASECALLER_MIC2  = PORT_BASE_ACQUISITION_DEFAULT + 50;
  const uint16_t PORT_CTRL_BASECALLER_MIC2  = PORT_BASE_BASECALLER_MIC2 + 0;
  const uint16_t PORT_STAT_BASECALLER_MIC2  = PORT_BASE_BASECALLER_MIC2 + 1;
  const uint16_t PORT_HTTP_BASECALLER_MIC2  = PORT_BASE_BASECALLER_MIC2 + 2;
  const uint16_t PORT_CTRL7_BASECALLER_MIC2 = PORT_BASE_BASECALLER_MIC2 + 7;

  const uint16_t PORT_BASE_POSTPRIMARY      = PORT_BASE_ACQUISITION_DEFAULT + 60;
  const uint16_t PORT_CTRL_POSTPRIMARY      = PORT_BASE_POSTPRIMARY + 0; // placeholder
  const uint16_t PORT_STAT_POSTPRIMARY      = PORT_BASE_POSTPRIMARY + 1;
  const uint16_t PORT_HTTP_POSTPRIMARY      = PORT_BASE_POSTPRIMARY + 2;

  const uint16_t subPostPrimaryPort = 5601;
  const uint16_t postPrimPushPort = 5600;
  const uint16_t xferPort = 8090;
  const uint16_t pawsPort = 8091;

  const PacBio::IPC::MessageQueueLabel pa_wsCommandQueue("pa-ws-command",pac_host, PORT_CTRL_WEBSERVICE);
  const PacBio::IPC::MessageQueueLabel pa_wsStatusQueue ("pa-ws-status" ,pac_host, PORT_STAT_WEBSERVICE);

  const PacBio::IPC::MessageQueueLabel pa_acqCommandQueue("pa-acq-command", pac_host, PORT_CTRL_ACQUISITION);
  const PacBio::IPC::MessageQueueLabel pa_acqStatusQueue ("pa-acq-status" , pac_host, PORT_STAT_ACQUISITION);

  const PacBio::IPC::MessageQueueLabel pa_bwCommandQueue("pa-bw-command", pac_host, PORT_CTRL_BASEWRITER);
  const PacBio::IPC::MessageQueueLabel pa_bwStatusQueue ("pa-bw-status" , pac_host, PORT_STAT_BASEWRITER);

  const PacBio::IPC::MessageQueueLabel pa_t2b0CommandQueue{"t2b0Command", "mic0", PORT_CTRL_BASECALLER_MIC0};
  const PacBio::IPC::MessageQueueLabel pa_t2b0StatusQueue {"t2b0Status" , "mic0", PORT_STAT_BASECALLER_MIC0};

  const PacBio::IPC::MessageQueueLabel pa_t2b1CommandQueue{"t2b1Command", "mic1", PORT_CTRL_BASECALLER_MIC1};
  const PacBio::IPC::MessageQueueLabel pa_t2b1StatusQueue {"t2b1Status" , "mic1", PORT_STAT_BASECALLER_MIC1};

  const PacBio::IPC::MessageQueueLabel pa_t2b2CommandQueue{"t2b2Command", "mic2", PORT_CTRL_BASECALLER_MIC2};
  const PacBio::IPC::MessageQueueLabel pa_t2b2StatusQueue {"t2b2Status" , "mic2", PORT_STAT_BASECALLER_MIC2};

  const PacBio::IPC::MessageQueueLabel ppadCommandQueue("incomingMessage",postPrimPushPort); ///< The incoming command queue label (pull model)
  const PacBio::IPC::MessageQueueLabel ppadStatusQueue("outgoingMessage", subPostPrimaryPort);   ///< the outgoing status queue label (publisher model)

 }
}






#endif // _PRIMARY_IPC_H_
