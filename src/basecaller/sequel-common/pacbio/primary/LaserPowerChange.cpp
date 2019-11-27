
// Copyright (c) 2018, Pacific Biosciences of California, Inc.
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
//  Defines some members of class LaserPowerChange.

#include <pacbio/primary/LaserPowerChange.h>
#include <pacbio/primary/EventObject.h>

namespace PacBio {
namespace Primary {

LaserPowerChange::LaserPowerChange(const EventObject& eo) :
    timeStamp_{ eo.timestamp_epoch() },
    frameInterval_{ eo.startFrame(), eo.stopFrame()}
{
    for(const auto& laser : eo.lasers)
    {
        if (laser.name() == LaserPowerObject::LaserName::topLaser)
        {
            topPower_[0] = static_cast<float>(laser.startPower_mW());
            topPower_[1] = static_cast<float>(laser.stopPower_mW());
        }
        else if (laser.name() == LaserPowerObject::LaserName::bottomLaser)
        {
            bottomPower_[0] = static_cast<float>(laser.startPower_mW());
            bottomPower_[1] = static_cast<float>(laser.stopPower_mW());
        }
        else
        {
            throw PBException("Unknown laser name " + laser.name().toString() + ", can't import into LaserPowerChange");
        }
    }
}

std::string LaserPowerChange::ToString() const
{
    std::ostringstream ss;

    ss << "[ TimeStamp:" << TimeStamp()
       <<  " StartFrame:" << StartFrame()
       <<  " StopFrame:" << StopFrame()
       <<  " LaserPowers top:" << topPower_[0] << " to " << topPower_[1]
       <<  " bottom:" << bottomPower_[0] << " to " << bottomPower_[1] << " ]";
    return ss.str();
}

}}  // PacBio::Primary
