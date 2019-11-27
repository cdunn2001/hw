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
/// \brief Intel MIC SCIF Heartbeat class.

#include <json/json.h>

#include <pacbio/ipc/pbi_scif.h>
#include <pacbio/ipc/MessageQueue.h>
#include <pacbio/logging/Logger.h>
#include <pacbio/primary/SCIF_Heartbeat.h>
#include <pacbio/primary/PrimaryConfig.h>


using namespace PacBio::IPC;


namespace PacBio
{
 namespace Primary
 {

  SCIF_Heartbeat::SCIF_Heartbeat()
          :
          name_(""),
          pid_(0),
          scifReady_(false),
          pipelineReady_(false)
  {
  }


  SCIF_Heartbeat::SCIF_Heartbeat(const MessageQueueLabel& queue)
          :
          pid_(getpid()),
          scifReady_(false),
          pipelineReady_(false)
  {
      uint16_t selfNode;
      // the stub version of scif_get_nodeIDs returns 0
      // the real version returns the number of nodes (usually 4)
      int status;
      if (pbi_scif_supported())
      {
          status = pbi_scif_get_nodeIDs(nullptr, 0, &selfNode);
      }
      else
      {
          status = 1; // one node, self
          selfNode = 0; // self
      }

      if (status > 0)
      {
          if (selfNode > 3)
          {
              throw PBException("pbi_scif_get_nodeIDs reported a node ID of " + std::to_string(selfNode) + " which is unexpected");
          }
          if (selfNode == 0)
          {
              name_="pac-host";
          }
          else
          {
              name_ = "mic" + std::to_string(selfNode - 1);
          }
      }
      else
      {
          PBLOG_WARN << "pbi_scif_get_nodeIDs, status:" << status << " selfNode:" << selfNode;
          name_ = "pac-host";
      }
      name_ += ":" + std::to_string(queue.GetPort());
      PBLOG_INFO << "Configured SCIF_Heartbeat heartbeat. name:" << name_;
  }

  std::string SCIF_Heartbeat::Serialize() const
  {
      Json::Value v;
      v["name"] = name_;
      v["pid"] = pid_;
      v["scifready"] = scifReady_;
      v["pipelineready"] = pipelineReady_;
      Json::StreamWriterBuilder b;
      std::string text = Json::writeString(b, v);
      return text;
  }

  void SCIF_Heartbeat::Deserialize(const std::string& json)
  {
      PBLOG_DEBUG << "Got heartbeat: " << json;
      Json::Reader r;
      Json::Value v;
      if (!r.parse(json, v, false))
      {
          throw PBException("JSONCPP Parse error for " + json);
      }
      name_ = v["name"].asString();
      pid_ = v["pid"].asInt();
      scifReady_ = v["scifready"].asBool();
      pipelineReady_ = v["pipelineready"].asBool();
  }
 }
}
