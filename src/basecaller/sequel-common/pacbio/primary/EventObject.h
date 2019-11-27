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
/// \brief  JSON based description of "events", such as LaserPowerChange, etc

#ifndef SEQUELACQUISITION_EVENTOBJECT_H
#define SEQUELACQUISITION_EVENTOBJECT_H

#include <string>
#include <pacbio/process/ConfigurationBase.h>
#include <pacbio/utilities/SmartEnum.h>

namespace PacBio {
namespace Primary {

class LaserPowerObject :
        public PacBio::Process::ConfigurationObject
{
public:
    SMART_ENUM(LaserName, unknown=-1, topLaser = 0, bottomLaser = 1 );
public:
    CONF_OBJ_SUPPORT_COPY(LaserPowerObject);
    ADD_ENUM(LaserName, name, LaserName::unknown);
    ADD_PARAMETER(double, startPower_mW, 0); // laser power at startFrame
    ADD_PARAMETER(double, stopPower_mW, 0); // laser at stopFrame
};

class EventObject :
        public PacBio::Process::ConfigurationObject
{
public:
    SMART_ENUM(EventType, unknown, laserpower, hotstart);
    EventObject() {
        Register();
    }
    EventObject(const EventObject& a ) : ConfigurationObject() {
        Register();
        Copy(a);
        PostImportAll();
    }
    EventObject& operator=(const EventObject& a) {
        Register();
        Copy(a);
        PostImportAll();
        MarkChanged();
        return *this;
    }

public:

    ADD_PARAMETER(std::string, timeStamp,
                  ""); // a time stamp to associate with this event in ISO8601 format, millisecond precision
    ADD_PARAMETER(uint64_t, startFrame, 0); // inclusive frame index
    ADD_PARAMETER(uint64_t, stopFrame, 0); // exclusive frame index
    ADD_ARRAY(LaserPowerObject, lasers);
    ADD_ENUM(EventType, eventType, EventType::unknown);
    ADD_PARAMETER(std::string, token, ""); // aka acquisition UIID
    ADD_PARAMETER(double, timestamp_epoch,
                  0.0); // double Epoch time, converted from ISO8601 format. Do not send this field!

    void OnChanged() override;
private:
    void Register();
};

}} // end namespace

#endif //SEQUELACQUISITION_EVENTOBJECT_H
