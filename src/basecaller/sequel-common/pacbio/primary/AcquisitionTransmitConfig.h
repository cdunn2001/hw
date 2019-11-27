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
/// \brief  Configuration object definition used for transmitting simulated data with pa-acq
///

#ifndef SEQUELACQUISITION_ACQUISITIONTRANSMITCONFIG_H
#define SEQUELACQUISITION_ACQUISITIONTRANSMITCONFIG_H

#include <pacbio/process/ConfigurationBase.h>
#include <pacbio/primary/SequelDefinitions.h>
#include <pacbio/primary/ChipClass.h>

namespace PacBio {
namespace Primary {

class TransmitConfig : public PacBio::Process::ConfigurationObject
{
public:
    enum _startframe_e : int64_t
    {
        READ_FRAME_INDEX_FROM_FILE = -1,
        CONTINUE_FRAME_INDICES     = -2
    };
    SMART_ENUM(Mode,NOP,File,Generated);

    ADD_PARAMETER(uint64_t,frames,512);
    ADD_PARAMETER(bool,enablePadding,false);
    ADD_PARAMETER(uint64_t,limitFileFrames,0x7FFFFFFFFFFFFFFFULL);
    ADD_PARAMETER(std::string,hdf5input,"");
    ADD_PARAMETER(double,rate,Sequel::defaultFrameRate);
    ADD_PARAMETER(double,linerate,0);
    ADD_PARAMETER(bool,condensed,false);
    ADD_PARAMETER(int64_t,startFrame,READ_FRAME_INDEX_FROM_FILE);
    ADD_PARAMETER(std::string,paacqCommandSocket,""); // transmit simulated events to this IPC endpoint. could be "localhost:46600"
    ADD_ENUM(Mode, mode , Mode::NOP);

public:
    TransmitConfig() = delete;
    TransmitConfig(ChipClass chipClass)
    {
        if (chipClass != ChipClass::DONT_CARE)
        {
            rate     = DefaultFrameRate(chipClass);
            linerate = DefaultLineRate(chipClass);
        }
    }

};

}}

#endif //SEQUELACQUISITION_ACQUISITIONTRANSMITCONFIG_H
