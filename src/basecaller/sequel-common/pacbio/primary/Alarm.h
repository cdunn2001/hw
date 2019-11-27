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
/// \brief  Alarm object declarations
///

#ifndef SEQUELACQUISITION_ALARM_H
#define SEQUELACQUISITION_ALARM_H

#include <unordered_set>
#include <boost/unordered_map.hpp>

#include <pacbio/utilities/SmartEnum.h>
#include <pacbio/ipc/MessageQueue.h>

namespace Json {
class Value;
}

class CivetHandler;

namespace PacBio {
namespace Primary {

  class AlarmManager;

  class Alarm
  {
  public:
      SMART_ENUM(Severity_e,CLEAR=0,WARNING=1,ERROR=2,CRITICAL=3,FATAL=4);
      static const int NumSeverityLevels = 5;
      static_assert(Severity_e::FATAL +1 == 5,"misconfiguration here");

      /// The default constructor
      Alarm() {}

      /// The normal constructor for creating new alarms
      /// \param id0     A hierarchical path. Example: "primary.acq.alarms.temperature.wolverine.high"
      /// \param name0   A human readable name for this alarm.
      /// \param format0 A sprintf format string that uses %s for each alarm metric (up to 3)
      /// \param doc0    Documentation on this alarm, including what causes it to be raised and trouble shooting info.
      Alarm(const std::string& id0,
            const std::string& name0,
            const std::string& format0,
            const std::string& doc0);

      /// Constructor from JSON object
      Alarm(const Json::Value& json);

     // Alarm(const Alarm& copy) = delete;

  public:
      /// Raises the alarm and notifies the parent alarm manager. The optional metric argument are used to
      /// fill in the %s placeholders of the format memeber.
      /// \param severity The new severity. Can be CLEAR.
      /// \param metric0 The first metric.
      /// \param metric1 The second metric
      /// \param metric2 The third metric
      void Raise(Severity_e severity, std::string metric0 = "", std::string metric1 = "", std::string metric2 = "");

      /// Clears the alarm. Equivalent to calling the Raise with Severity_e::CLEAR, but easier to read.
      void Clear();

      /// Raises or Clears the alarm, based on the boolean flag argument.
      /// \param flag If true, the the alarm is raised to the severity. If false, the alarm is cleared.
      /// \param metric0 The first metric.
      /// \param metric1 The second metric
      /// \param metric2 The third metric
      void RaiseOrClear(bool flag, Severity_e severity0,  std::string metric0 = "", std::string metric1 = "", std::string metric2 = "");

      /// Sets optional key/value pairs for further describing the context of the alarm. This may
      /// be useful for setting things like robot position, etc. The output JSON object will have an info member
      /// that consists of "key":"value" members, i.e. {"info": { "key1" :"value1", "key2": "value2", ...} }
      /// \param key The key string
      /// \param value The value string
      void SetInfo(const std::string& key, const std::string& value);

      /// Copying an alarm is not allowed. This is because some members are immutable (such as name).
      /// If you wish to copy the state of an alarm to another alarm, use the Update() member.
      Alarm& operator=(const Alarm& src) = delete;

      /// Compares two alarms
      /// \param b The external alarm to compare with "this" alarm
      bool operator==(const Alarm& b);

      /// Renders the alarm into a JSON formatted string
      std::string RenderJSON() const;

      /// Converts the alarm to a JSON object that is used internally.
      Json::Value ToInternalJSON() const;

      /// Converts the alarm to a JSON object that is sent externally, i.e. to PAWS
      Json::Value ToExternalJSON() const;

      /// Registers the alarm with the alarm manager. The alarm manager is notified when the alarm changes severity.
      /// \ param manager The AlarmManager instant to which this alarm will be connected.
      void Register(AlarmManager& manager);

      /// Alarms can be connected to at most one alarm manager. Usually there is just one alarm manager in a process,
      /// but in test code, the alarm manager can come and go. In some test code, there is only the global alarm
      /// object without a manager. So we have to unregister alarms when the alarm manager is destroyed.
      void Unregister();

      /// \returns true if the alarm has been registered by an AlarmManager
      bool Registered() const;

      /// Updates the mutable fields of the alarm with a second alarm
      /// \param src The source Alarm object to copy mutable state from (severity, etc)
      void Update(const Alarm& src);

      void SetAgingPolicy(double ageInSeconds);

      void ClearImmediately();

      double LastTimestamp(Severity_e s) const { return lastTimestamp_[ s];}

      virtual double GetMonotonicTime() const { return PacBio::Utilities::Time::GetMonotonicTime(); }

      Severity_e GetSeverity() const { return severity; }

  public:
      // "immutable", set at construction
      const std::string name;
      const std::string englishName;
      const std::string format;
      const std::string doc;

      // "mutable", changed during raising/clearing
      Severity_e  severity;             ///< argument to Raise
      std::vector<std::string> metrics; ///< arguments to Raise()
      std::string message;              ///< The output of sprintf(format,metrics)
      std::string whenChanged;          ///< Contains an ISO8601 formatted date/time stamp
      std::string stacktrace;           ///< optional stack trace to the line where the alarm was Raise()
      std::map<std::string,std::string> info; ///< optional information filled in before Raise()

  protected:
      AlarmManager* manager_ = nullptr;
      std::array<double,NumSeverityLevels> lastTimestamp_= {{0,0,0,0,0}};
      double age_ = 0;
  };

  std::ostream& operator<<(std::ostream& os, const Alarm& a);

  /// The AlarmManager acts as the gateway between the Alarm objects and the outside world.
  /// It notifies foreign IPC end points with JSON formatted Alarm objects.
  /// It manages a database of Alarms. When an alarm is Raised or Cleared, the new state of the
  /// alarm is compared to the database of Alarms. If there is a change, the manager is marked "dirty"
  /// and the next call to the Service() member will broadcast the changes to the IPC end-point.
  /// It will also send periodic Alarm updates to the IPC end-point.
  /// It is thread safe.
  class AlarmManager
  {
  public:
      /// Constructor.
      /// \param label The IPC message socket end-point to which send Alarm updates
      /// The socket must exist already.
      AlarmManager(PacBio::IPC::MessageSocketPublisher& socket, const std::string& source);
      AlarmManager(const AlarmManager&) = delete;
      AlarmManager(AlarmManager&&) = delete;

      /// dtor
      ~AlarmManager();

      /// Registers the alarm with the alarm manager. An alarm can be registered with just
      /// one alarm manager.
      /// \param the alarm to register
      void Register(Alarm& alarm);

      /// Updates the internal Alarm database with the indicates Alarm
      /// \param the alarm to update
      void Update(const Alarm& alarm);

      /// Finds the alarm in the internal Alarm data based on the name key.
      /// If it is not found, an exception is thrown.
      /// \param name The alarm name to find.
      const Alarm& Find(const std::string& name) const;

      /// Outputs a JSON array of JSON objects of Alarms, representing the current internal database.
      Json::Value ToInternalJSON() const;

      /// Outputs a JSON array of JSON objects of Alarms, to be sent to PAWS
      Json::Value ToExternalJSON() const;

      /// Sends a JSON message to the IPC end-point
      void SendJsonUpdate();

      /// A member that should be called frequently to service the state machine of the AlarmManager.
      /// If placed in a thread, this should be called with a period that is as large as possible
      /// without affecting latency requirements.  A 1 second interval is recommended if an Alarm latency of
      /// 1 second is acceptable.
      void Service();

      std::string Source() const { return source_;}

      /// enable the alarm manager, which basically means allow it to emit alarm messages.
      /// Thread-safe
      void Enable()
      {
          std::lock_guard<std::recursive_mutex> l(mutex_);
          dirty_ = true;
          enabled_ = true;
      }

      /// disable the alarm manager, prohibit it to emit alarm messages.
      /// Thread-safe
      void Disable()
      {
          std::lock_guard<std::recursive_mutex> l(mutex_);
          enabled_ = false;
      }

      /// The maximum time between Alarm updates. If no alarm has changed within this time period,
      /// the an alarm update will be broadcast.
      static constexpr double ALARM_UPDATE_INTERVAL = 600.0;

      /// Used for aging alarms
      static constexpr double ALARM_SERVICE_INTERVAL = 1.0;

      /// Registers all the global alarms with this alarm manager.
      void RegisterAllGlobalAlarms();

      /// Globals::List() is a list of all alarms that were constructed at ctor_init time.
      /// The constructor is used to simply append to the static vector.
      class Globals
      {
      public:
          Globals(Alarm& a)
          {
                List().push_back(&a);
          }
          /// Meyer's singleton for global
          static std::vector<Alarm*>& List()
          {
              static std::vector<Alarm*> alarms;
              return alarms;
          }
      };

      /// Returns a CivetHandler instance that handles the "/alarms" HTTP get request.
      CivetHandler* GetCivetHandler();

private:
      boost::unordered_map<std::string,Alarm> alarms_;
      PacBio::IPC::MessageSocketPublisher& sender_;
      const std::string source_;
      double tLastAlarmBroadcast = 0;
      mutable std::recursive_mutex mutex_;
      bool dirty_ = false;
      bool enabled_ = false;
      std::unordered_set<Alarm*> externalAlarms_;
  };

/// The ALARM() macro is used in 3 places
/// 1. in C++ declaration mode, i.e. in a header
/// 2. in C++ implementation mode, i.e. in a cpp file
/// 3. in documentation mode, using cpp but ultimately getting converted to a HTML documentation file
///
/// The five fields are
/// a) C++ instance name
/// b) the textual ID
/// c) the textual name (for display in a UI, for example)
/// d) the format string to prepare a message. Optional arguments are all strings and should be notated with %s
/// e) instructions to the end user on how to deal with the alarm, either explaining what causes it and what can be done
///    to correct it.
///
/// ALARM1() is an extension to ALARM() an extra field.
/// f) the criteria used to raise the alarm to various severities. By convention, the severity names should be given
///    in ALL CAPS.
///
#ifndef ALARM
#ifdef PACBIO_ALARM_IMPLEMENTATION
 #define ALARM(a,b,c,d,e)    PacBio::Primary::Alarm a (b,c,d,e); PacBio::Primary::AlarmManager::Globals a ## _initer(a);
#else
/// This macro declares a new alarm object.
 #define ALARM(a,b,c,d,e)    extern PacBio::Primary::Alarm a;
#endif
#endif
#ifndef ALARM1
// wrapper for documentation. 'f' field is not used in code, but used by documentation automation.
#define ALARM1(a,b,c,d,e,f) ALARM(a,b,c,d,e)
#endif

}}
#endif //SEQUELACQUISITION_ALARM_H
