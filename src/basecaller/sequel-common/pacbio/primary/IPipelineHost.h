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
/// \brief Interface declaration for managing the pipeline from the host side


#ifndef PA_BW_INTERFACES_H
#define PA_BW_INTERFACES_H

#include <cstdint>
#include <memory>
#include <pacbio/primary/SequelMovie.h>

namespace PacBio
{
 namespace IPC
 {
  class PoolBase;
  class TitleBase;
  class Message;
 }
 namespace Primary
 {
  class Tile;
  class TrancheTitle;
 }
}


class IPipelineHost
{
public:
    virtual void SendDeed(const std::string& name,PacBio::IPC::TitleBase& title, uint32_t micOffset) = 0;
    virtual void SendMessageDownStream(const PacBio::IPC::Message& message) = 0;
    virtual PacBio::IPC::PoolBase& GetPool() = 0;
    virtual bool IsEnabled() const = 0;
    virtual uint32_t MicTotalCount() const = 0;
};


#endif //PA_BW_INTERFACES_H
