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
/// \brief C++ methods to control and monitor transfer services



#ifndef SEQUELACQUISITION_XFERREMOTECONTROL_H
#define SEQUELACQUISITION_XFERREMOTECONTROL_H

#include <pacbio/process/ConfigurationBase.h>

namespace PacBio {
namespace Primary {

SMART_ENUM(XferState,
    unknown,
    CREATED,    ///< aka idle
    RUNNING,    ///< aka busy
    FAILED,
    SUCCESSFUL
);

class XferStatus : public PacBio::Process::ConfigurationObject
{
    ADD_PARAMETER(std::string, uuid, "");
    ADD_PARAMETER(std::string, message,"");
    ADD_PARAMETER(double, runTimeSec, 0.0);
    ADD_ENUM(XferState, state, XferState::unknown);
public:
    double startTime_ = 0.0;
};


}}

#endif //SEQUELACQUISITION_XFERREMOTECONTROL_H
