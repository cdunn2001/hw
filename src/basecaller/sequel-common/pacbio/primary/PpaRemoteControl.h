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
//  Description:
/// \brief  Methods for controling and monitoring PPA remotely from a C++ program
///
#ifndef SEQUELACQUISITION_PPAREMOTECONTROL_H
#define SEQUELACQUISITION_PPAREMOTECONTROL_H

#include <pacbio/process/ProcessBaseRemoteControl.h>
#include <pacbio/process/ConfigurationBase.h>

namespace PacBio {
namespace Primary {

SMART_ENUM(PPAState,
        complete = 0,
        start,
        idle,
        busy, // aka "started"
        progress,
        error, // aka "ERROR"
        warning
);



struct PpaStatus : public PacBio::Process::ConfigurationObject
{
    ADD_PARAMETER(std::string,acqId,"");
    ADD_PARAMETER(std::string,message,"");
    ADD_PARAMETER(uint32_t,progress,0);  // todo consider changing this to double
    ADD_ENUM(PPAState,state,PPAState::idle);
};

class PpaStartMessage : public PacBio::Process::ConfigurationObject
{
    ADD_PARAMETER(std::string, acqId, "");
    ADD_PARAMETER(std::string, outputPrefix, "");
    ADD_PARAMETER(std::string, bazFile, "");
    ADD_PARAMETER(std::string, rmdFile, "");
    ADD_PARAMETER(std::string, outputFolder, "/data/pa");
    ADD_PARAMETER(std::string, barcodesFile, "");
    ADD_PARAMETER(std::string, barcodeScoreMode, "");
    ADD_PARAMETER(int, minSubLength, 50);
    ADD_PARAMETER(float, minSnr, 4);
};


class PpaRemoteControl : public PacBio::Process::ProcessBaseRemoteControl
{
public:
    PpaRemoteControl(std::string host = "localhost") : ProcessBaseRemoteControl(host,postPrimPushPort,subPostPrimaryPort)
    {

    }

    void Start(const PpaStartMessage& message)
    {
        this->Send("ppaStart",PacBio::IPC::RenderJSON(message.Json()));
    }

    void Stop(const std::string& id)
    {
        Json::Value jsonMessage;
        jsonMessage["acqId"] = id;
        Send("ppaStop",PacBio::IPC::RenderJSON(jsonMessage));
    }
};

}}


#endif //SEQUELACQUISITION_PPAREMOTECONTROL_H
