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
/// \brief  Alarm functionality for creating and sending Alarms to ICS
///

#include <mutex>

#include <boost/regex.hpp>

#include <json/json.h>

#include <pacbio/ipc/JSON.h>
#include <pacbio/primary/Alarm.h>
#include <pacbio/utilities/ISO8601.h>
#include <pacbio/text/String.h>
#include <pacbio/logging/Logger.h>
#include <pacbio/Utilities.h>
#include <pacbio/process/CivetServer.h>

namespace PacBio {
namespace Primary {

// this will cause the unit tests to fail on purpose. I
// It simulates the condition of a fresh boot, in which all alarms are asserted until the Aging time has elapsed.
// const static double A_LONG_TIME_AGO = 0;

static const double A_LONG_TIME_AGO = -3e7; // one year ago ish :)

Alarm::Alarm(const std::string& name0,
             const std::string& englishName0,
             const std::string& format0,
             const std::string& doc0) :
        name(name0),
        englishName(englishName0),
        format(format0),
        doc(doc0),
        severity(Alarm::Severity_e::CLEAR),
        whenChanged(PacBio::Utilities::ISO8601::TimeString())
{
    lastTimestamp_.fill(A_LONG_TIME_AGO);
    boost::regex r("^[A-Za-z]([A-Za-z0-9_]+\\.){0,99}[A-Za-z0-9_]*[A-Za-z0-9]$");
    if (!boost::regex_match(name, r))
        throw PBException("Invalid Alarm name: " + name);
    Clear();
}

Alarm::Alarm(const Json::Value& value) :
        name(value["name"].asString()),
        englishName(value["englishName"].asString()),
        format(value["format"].asString()),
        doc(value["doc"].asString()),
        severity(value["severity"].asString()),
        message(value["message"].asString()),
        whenChanged(value["whenChanged"].asString()),
        stacktrace(value["stacktrace"].asString())
{
    lastTimestamp_.fill(A_LONG_TIME_AGO);
    metrics.clear();
    for (const auto& node : value["metrics"])
    {
        metrics.push_back(node.asString());
    }
#if 0
    value["info"] = Json::objectValue;
    for(const auto& node : info)
    {
        value["info"][node.first] = node.second;
    }
#endif
}

// Raising to a non CLEAR state always happens immediately.
// Clearing can be delayed by an aging mechanism. For this to work, the alarm must be continually
// cleared. When age_ has elapsed from the first Clear() to the most recent Clear(), then the
// severity of the alarm is actually changed.
void Alarm::Raise(Severity_e severity0, std::string arg0, std::string arg1, std::string arg2)
{
    PBLOG_TRACE << "Raise:" << severity0;
    Severity_e newSeverity = Severity_e::CLEAR;

    double t = GetMonotonicTime();
    if((int)severity0 < 0 || (int)severity0 >= NumSeverityLevels)
    {
        throw PBException("Invalid severity");
    }
    lastTimestamp_[severity0] = t;
    for(int enumVal = Severity_e::FATAL; enumVal >= 0; enumVal--)
    {
        newSeverity = Severity_e::RawEnum(enumVal);
        double currentAge = t - lastTimestamp_[newSeverity];
        PBLOG_TRACE << this->name << " " << newSeverity << " is age " << currentAge << " compared to " << age_;
        if (currentAge <= age_) break;
    }
    if((int)newSeverity < 0 )
    {
        throw PBException("Logic bug");
    }

    if (newSeverity != severity)
    {
        PBLOG_TRACE << this->name << " going to " << newSeverity;
        severity = newSeverity;
        if (newSeverity == Severity_e::CLEAR)
        {
            message = "";
            metrics.clear();
            info.clear();
        }
        else
        {
            // immediately update alarm
            message = PacBio::Text::String::Format(format, arg0.c_str(), arg1.c_str(), arg2.c_str());
            metrics.resize(3);
            metrics[0] = arg0;
            metrics[1] = arg1;
            metrics[2] = arg2;
        }

        whenChanged = PacBio::Utilities::ISO8601::TimeString();
        stacktrace = "tbd";
        if (manager_) manager_->Update(*this);
    }

}

void Alarm::Clear()
{
    Raise(Severity_e::CLEAR);
}

void Alarm::RaiseOrClear(bool flag, Severity_e severity0, std::string arg0, std::string arg1, std::string arg2)
{
    Raise((flag ? severity0.native() : Severity_e::CLEAR), arg0, arg1, arg2);
}

void Alarm::ClearImmediately()
{
    // pretend the alarm has been cleared since forever
    lastTimestamp_.fill(A_LONG_TIME_AGO);
    Clear();
}

void Alarm::SetInfo(const std::string& field, const std::string& value)
{
    info[field] = value;
}

Json::Value Alarm::ToInternalJSON() const
{
    Json::Value value;
    value["englishName"] = englishName;
    value["name"] = name;
    value["format"] = format;
    value["doc"] = doc;
    value["severity"] = severity.toString();
    value["message"] = message;
    value["stacktrace"] = stacktrace;
    value["whenChanged"] = whenChanged;
    value["metrics"] = Json::arrayValue;
    for (const auto& node : metrics)
    {
        value["metrics"].append(node);
    }
    value["info"] = Json::objectValue;
    for (const auto& node : info)
    {
        value["info"][node.first] = node.second;
    }
    return value;
}

Json::Value Alarm::ToExternalJSON() const
{
    Json::Value value;
    value["id"] = 0;
    value["name"] = name;
    value["severity"] = severity.toString();
    value["message"] = message;
    if (manager_)
    {
        value["source"] = manager_->Source();
    }
    else
    {
        value["source"] = "svc://unknown";
    }
    return value;
}

std::string Alarm::RenderJSON() const
{
    std::stringstream ss;
    ss << ToInternalJSON();
    return ss.str();
}

void Alarm::Update(const Alarm& src)
{
    severity = src.severity;
    metrics = src.metrics;
    message = src.message;
    whenChanged = src.whenChanged;
    stacktrace = src.stacktrace;
    info = src.info;
    PBLOG_DEBUG << "Alarm::Update src:" << src.RenderJSON() << " this:" << RenderJSON();
}

bool Alarm::operator==(const Alarm& b)
{
    return severity == b.severity &&
           metrics == b.metrics &&
           message == b.message &&
           whenChanged == b.whenChanged &&
           stacktrace == b.stacktrace;
    //info = src.info;
}

void Alarm::Register(AlarmManager& manager)
{
    manager_ = &manager;
    PBLOG_DEBUG << "Alarm::Register " <<  (void*) this << " " << name <<
       " -> " << (void*) manager_;
}

void Alarm::Unregister()
{
    PBLOG_DEBUG << "Alarm::Unregister "<< (void*) this << " " << name;
    manager_ = nullptr;
    // pretend this alarm is brand new
    lastTimestamp_.fill(A_LONG_TIME_AGO);
    Clear();
}

bool Alarm::Registered() const
{
    PBLOG_DEBUG << "Alarm::Registered: this:" << (void*) this << " registered? manager_:" << (void*) manager_ <<
            " bool:" << (manager_ != nullptr);
    return manager_ != nullptr;
}

void Alarm::SetAgingPolicy(double age)
{
    age_ = age;
}

std::ostream& operator<<(std::ostream& os, const Alarm& a)
{
    os << " current severity:" << a.severity << "\n";
    double t = a.GetMonotonicTime();
    for(int i=Alarm::Severity_e::CLEAR; i < Alarm::NumSeverityLevels; i++)
    {
        Alarm::Severity_e severity = Alarm::Severity_e::RawEnum(i);
        double currentAge = t - a.LastTimestamp(severity);
        os << severity << " age:" << currentAge << "\n";
    }
    return os;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////


AlarmManager::AlarmManager(PacBio::IPC::MessageSocketPublisher& label,
                           const std::string& source) :
        sender_(label),
        source_(source),
        enabled_(false)
{
    // sender_.SetNoLinger(); not supported for publisher (only command) queue
}

AlarmManager::~AlarmManager()
{
    std::lock_guard<std::recursive_mutex> lock(mutex_);

    // unregister all alarms. They are global and have a longer life than this manager.
    for (auto& mm : alarms_)
    {
        PBLOG_DEBUG << "Unregistering alarm " << mm.first;
        mm.second.Unregister();
    }
    for (auto& mm : externalAlarms_)
    {
        PBLOG_DEBUG << "Unregistering alarm " << mm->name;
        mm->Unregister();
    }
}

void AlarmManager::Register(Alarm& alarm)
{
    std::lock_guard<std::recursive_mutex> lock(mutex_);

    // this is the external alarm object
    alarm.Register(*this);
    externalAlarms_.insert(&alarm);

    // this is the internal alarm object, which follows the external object.
    alarms_.emplace(alarm.name, Alarm(alarm));
    alarms_[alarm.name].Register(*this);
}

const Alarm& AlarmManager::Find(const std::string& name0) const
{
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    return alarms_.at(name0);
}


void AlarmManager::Update(const Alarm& alarm)
{
    PBLOG_DEBUG << "AlarmManager::Update for " << alarm.RenderJSON();

    // merge alarm and/or check if there was a change

    std::lock_guard<std::recursive_mutex> lock(mutex_);

    Alarm& oldAlarm = alarms_[alarm.name];

    if (oldAlarm.name == "")
    {
        PBLOG_ERROR << "Could not find alarm name " << alarm.name << " in AlarmManager, adding now. Please Fix this.";
        oldAlarm.Update(alarm);
        dirty_ = true;
    }
    else
    {
        if (!( oldAlarm == alarm))
        {
            if (enabled_)
            {
                PBLOG_INFO << "Alarm " << alarm.name << " changed, now dirty.";
                PBLOG_INFO << "oldAlarm:" << oldAlarm.RenderJSON();
                PBLOG_INFO << "newAlarm:" << alarm.RenderJSON();
            }
            dirty_ = true;
        }
        oldAlarm.Update(alarm);
    }
}

Json::Value AlarmManager::ToInternalJSON() const
{
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    Json::Value v;
    for (const auto& x : alarms_)
    {
        v.append(x.second.ToInternalJSON());
    }
    return v;
}

Json::Value AlarmManager::ToExternalJSON() const
{
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    Json::Value v;
    for (const auto& x : alarms_)
    {
        v.append(x.second.ToExternalJSON());
    }
    return v;
}

void AlarmManager::SendJsonUpdate()
{
    std::lock_guard<std::recursive_mutex> lock(mutex_);

    if (!enabled_) return;

    Json::Value v = ToExternalJSON();
    PBLOG_DEBUG << " AlarmManager::SendJsonUpdate: sending alarms/" << IPC::RenderJSON(v);
    IPC::Announcement msg("alarms", IPC::RenderJSON(v));
    sender_.Send(msg);
}

void AlarmManager::Service()
{
    double t = PacBio::Utilities::Time::GetMonotonicTime();

    std::lock_guard<std::recursive_mutex> lock(mutex_);

    if (dirty_ || t - tLastAlarmBroadcast >= ALARM_UPDATE_INTERVAL)
    {
        PBLOG_DEBUG << "AlarmManager::Service Dirty:" << dirty_ << " time since last alarm update:"
                        << (t - tLastAlarmBroadcast) << " >= " << ALARM_UPDATE_INTERVAL;
        tLastAlarmBroadcast = t;
        SendJsonUpdate();
        dirty_ = false;
    }

}

void AlarmManager::RegisterAllGlobalAlarms()
{
    PBLOG_DEBUG << "Registering " << AlarmManager::Globals::List().size() << " Alarms ";
    for (auto& a : AlarmManager::Globals::List())
    {
        Register(*a);
    }
}


class AlarmCivetHandler :
        public CivetHandler
{
public:
    AlarmCivetHandler(AlarmManager& alarmManager) :
            alarmManager_(alarmManager)
    {

    }

    bool handleGet(CivetServer* /*server*/, struct mg_connection* conn) override
    {
        mg_printf(conn, "HTTP/1.1 200 OK\r\n"
                "Content-Type: text/json\r\n"
                "Access-Control-Allow-Methods: POST, GET, OPTIONS\r\n"
                "Access-Control-Allow-Headers: Origin, X-Requested-With, Content-Type, Accept\r\n"
                "Access-Control-Allow-Origin: *\r\n\r\n");

        Json::Value v = alarmManager_.ToExternalJSON();
        std::string s = IPC::RenderJSON(v);

        mg_write(conn, s.c_str(), s.size());
        return true;
    }

private:
    AlarmManager& alarmManager_;
};

CivetHandler* AlarmManager::GetCivetHandler()
{
    return new AlarmCivetHandler(*this);
}


}
}

