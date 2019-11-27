    //
// Created by mlakata on 6/3/16.
//

#include <future>

#include <gtest/gtest.h>

#include <pacbio/dev/AutoTimer.h>
#include <pacbio/dev/gtest-extras.h>
#include <pacbio/dev/MockSocket.h>

#include <pacbio/process/CivetServer.h>
#include <pacbio/ipc/JSON.h>
#include <pacbio/ipc/MessageQueue.h>
#include <pacbio/PBException.h>
#include <pacbio/logging/Logger.h>
#include <pacbio/utilities/ISO8601.h>

#include <pacbio/primary/Alarm.h>
#include <pacbio/POSIX.h>

using namespace PacBio::Primary;
using namespace PacBio::Logging;
using namespace PacBio;

TEST(Alarm,UnitTestConstructor)
{
    /// Tests alarm construction and members
// .. requirement:: Alarm object properties

    Alarm alarm("primary.test.test", "Testing", "I need %s, %s and %s", "Try Ctrl-Alt-Delete to reboot");

    EXPECT_EQ("primary.test.test", alarm.name);
    EXPECT_EQ("Testing", alarm.englishName);
    EXPECT_EQ("I need %s, %s and %s", alarm.format);
    EXPECT_EQ("Try Ctrl-Alt-Delete to reboot", alarm.doc);

}

TEST(Alarm,UnitTestNamespace)
{
    /// Tests that namespace convention is asserted (with exceptions if badly formed)
// .. requirement:: Alarm id namespace
    EXPECT_NO_THROW(Alarm alarm1("primary.test.test","Testing","I need %s, %s and %s","Try Ctrl-Alt-Delete to reboot"));

    EXPECT_THROW(Alarm alarm2("","","",""), std::runtime_error) << "too short";
    EXPECT_THROW(Alarm alarm3("pri-mary.test.group","","",""), std::runtime_error) << "bad character";
}

TEST(Alarm,UnitTestRaiseAndClear)
{
    LogSeverityContext l(LogLevel::WARN);

    /// Tests that the Raise (and Clear) methods set the severity correctly and
    /// transmit the IPC message to the IPC receiver socket.
    /// Also indirectly tests that the JSON message is deserialized correctly.

//    PacBio::Logging::LogSeverityContext xxx(PacBio::Logging::LogLevel::DEBUG);


    // .. requirement:: Alarm Manager
    PacBio::IPC::MessageQueueLabel x("foo",46609);


    // This is the subscriber thread, to empty out the ZMQ queue
    std::promise<bool> ready;
    auto readyFuture = ready.get_future();

    int iterations = 2;
    std::vector<PacBio::IPC::SmallMessage> receivedAlarms;

    std::future<int> rx = std::async(std::launch::async,[x,&ready,iterations,&receivedAlarms](){
        PacBio::IPC::MessageSocketSubscriber receiver(x);
        receiver.SetTimeout(1000);
        int count=0;
        ready.set_value(true);
        while (true)
        {
            //TEST_COUT << "waiting.." << std::endl;
            PacBio::IPC::SmallMessage m1 = receiver.Receive();
            if (m1.IsAvailable())
            {
                receivedAlarms.push_back(m1);
                count++;
                //TEST_COUT << "got " << count << std::endl;
                if (m1.GetData() == "" || count >= iterations) break;
            }
            else
            {
                TEST_COUT << "TIMEOUT! count:" << count << " expected:" << iterations << std::endl;
                break;
            }
        }
        return count;
    }) ;

    PacBio::IPC::MessageSocketPublisher xx(x);
    Alarm a1("primary.test.test","Testing","I need %s, %s and %s","Try Ctrl-Alt-Delete to reboot");
    Alarm a2("primary.test.test.1","Testing1","I need %s, %s and %s","Try Ctrl-Alt-Delete to reboot");
    Alarm a3("primary.test.test.2","Testing2","I need %s, %s and %s","Try Ctrl-Alt-Delete to reboot");
    Alarm a4("primary.test.test.3","Testing3","I need %s, %s and %s","Try Ctrl-Alt-Delete to reboot");
    // alarms must have longer life than manager.
    EXPECT_FALSE(a1.Registered());
    {
        AlarmManager manager(xx,"svc://foo");
        // .. requirement:: Updateable Message Configuration
        // .. requirement:: Troubleshooting Guide and Support Per Alarm
        manager.Register(a1);
        manager.Enable();

        EXPECT_TRUE(a1.Registered());

        readyFuture.get();
        usleep(20000);

        // .. requirement:: C++ API to Raise and Clear Alarms
        EXPECT_EQ(Alarm::Severity_e::CLEAR, a1.severity);
        a1.Raise(Alarm::Severity_e::WARNING, "x","y","z");
        EXPECT_EQ("I need x, y and z", a1.message);
        EXPECT_EQ("I need x, y and z", manager.Find("primary.test.test").message);
        EXPECT_EQ(Alarm::Severity_e::WARNING, a1.severity);
        EXPECT_EQ(Alarm::Severity_e::WARNING, manager.Find("primary.test.test").severity);


        manager.Service();

        a1.Clear();
        EXPECT_EQ("", a1.message);
        EXPECT_EQ("", manager.Find("primary.test.test").message);
        EXPECT_EQ(Alarm::Severity_e::CLEAR, a1.severity);
        EXPECT_EQ(Alarm::Severity_e::CLEAR, manager.Find("primary.test.test").severity);

        manager.Service();


        rx.wait();
        EXPECT_EQ(2,rx.get());
        ASSERT_EQ(2,receivedAlarms.size());

        PacBio::IPC::SmallMessage m1 = receivedAlarms[0];
        ASSERT_TRUE(m1.IsAvailable());
        Json::Value alarms = PacBio::IPC::ParseJSON(m1);
        for(const auto& alarmJson : alarms)
        {
            Alarm a(alarmJson);
            EXPECT_EQ("0",alarmJson["id"].asString());
            EXPECT_EQ("primary.test.test",a.name) << alarmJson;
            EXPECT_EQ(Alarm::Severity_e::WARNING,a.severity) << alarmJson;
        }


        m1 = receivedAlarms[1];
        ASSERT_TRUE(m1.IsAvailable());
        alarms = PacBio::IPC::ParseJSON(m1);
        for(const auto& alarmJson : alarms)
        {
            Alarm a(alarmJson);
            EXPECT_EQ("primary.test.test",a.name);
            EXPECT_EQ(Alarm::Severity_e::CLEAR,a.severity);
        }

        EXPECT_TRUE(a1.Registered());

        // manager gets destroyed here, should unregister all alarms.
    }
    EXPECT_FALSE(a1.Registered());
}

TEST(Alarm,UnitTestJson)
{
    /// Tests that the alarm can be serialized to JSON
    // .. requirement:: Alarm object properties

    Alarm alarm("primary.test.test","Testing","I need %s, %s and %s","Try Ctrl-Alt-Delete to reboot");

    auto now = PacBio::Utilities::ISO8601::TimeString();

    alarm.SetInfo("outlook","good");
    alarm.Raise(Alarm::Severity_e::WARNING, "x","y","z");
    std::string s = alarm.RenderJSON();

    Json::Value v;
    std::stringstream ss;
    ss << s;
    ss >> v;

    EXPECT_EQ("primary.test.test",v["name"].asString()) << s;
    EXPECT_EQ("Testing",v["englishName"].asString()) << s;
    EXPECT_EQ("I need x, y and z",v["message"].asString()) << s;
    EXPECT_EQ("I need %s, %s and %s",v["format"].asString()) << s;
    EXPECT_EQ("Try Ctrl-Alt-Delete to reboot",v["doc"].asString()) << s;
    EXPECT_EQ("WARNING",v["severity"].asString()) << s;
    // EXPECT_EQ("",v["stacktrace"].asString()) << s;

    ASSERT_TRUE(v["metrics"].isArray()) << s;
    EXPECT_EQ("x",v["metrics"][0].asString());
    EXPECT_EQ("y",v["metrics"][1].asString());
    EXPECT_EQ("z",v["metrics"][2].asString());

    ASSERT_TRUE(v["info"].isObject()) << s;
    EXPECT_EQ("good",v["info"]["outlook"].asString());

    double delta = PacBio::Utilities::ISO8601::EpochTime(now) -
                   PacBio::Utilities::ISO8601::EpochTime(v["whenChanged"].asString());
    EXPECT_LT(std::abs(delta),1) << s;
}

TEST(Alarm,DISABLED_UnitTestMasking)
{
    /// Tests that the Alarms can be masked
    // .. requirement:: Maskable Alarms

    EXPECT_TRUE(false);
}

TEST(Alarm,Aging1)
{
    Alarm a1("primary.test.test", "Testing", "I need %s, %s and %s", "Try Ctrl-Alt-Delete to reboot");
    a1.SetAgingPolicy(0.5);

    EXPECT_EQ(Alarm::Severity_e::CLEAR, a1.severity);
    POSIX::usleep(1000000);
    EXPECT_EQ(Alarm::Severity_e::CLEAR, a1.severity);

    // test raising and clearing with aging
    a1.Raise(Alarm::Severity_e::WARNING);
    EXPECT_EQ(Alarm::Severity_e::WARNING, a1.severity);
    a1.Clear();
    EXPECT_EQ(Alarm::Severity_e::WARNING, a1.severity);
    a1.Clear();
    EXPECT_EQ(Alarm::Severity_e::WARNING, a1.severity);
    POSIX::usleep(300000);
    a1.Clear();
    EXPECT_EQ(Alarm::Severity_e::WARNING, a1.severity);
    POSIX::usleep(200000);
    a1.Clear();
    EXPECT_EQ(Alarm::Severity_e::CLEAR, a1.severity);
}

TEST(Alarm,AgingImmediateClear)
{
    Alarm a1("primary.test.test", "Testing", "I need %s, %s and %s", "Try Ctrl-Alt-Delete to reboot");
    a1.SetAgingPolicy(0.5);
    // test clear immediately
    a1.Raise(Alarm::Severity_e::WARNING);
    EXPECT_EQ(Alarm::Severity_e::WARNING, a1.severity);
    a1.ClearImmediately();
    EXPECT_EQ(Alarm::Severity_e::CLEAR, a1.severity);
}

TEST(Alarm,Aging2)
{
    Alarm a1("primary.test.test", "Testing", "I need %s, %s and %s", "Try Ctrl-Alt-Delete to reboot");
    a1.SetAgingPolicy(0.5);
    // test raising to ERROR, then WARN
    a1.Raise(Alarm::Severity_e::ERROR);
    EXPECT_EQ(Alarm::Severity_e::ERROR, a1.severity);
    a1.Raise(Alarm::Severity_e::WARNING);
    EXPECT_EQ(Alarm::Severity_e::ERROR, a1.severity);
    POSIX::usleep(300000);
    a1.Raise(Alarm::Severity_e::WARNING);
    EXPECT_EQ(Alarm::Severity_e::ERROR, a1.severity);
    POSIX::usleep(300000);
    a1.Raise(Alarm::Severity_e::WARNING);
    EXPECT_EQ(Alarm::Severity_e::WARNING, a1.severity);
}

TEST(Alarm, Aging3)
{
    Alarm a1("primary.test.test", "Testing", "I need %s, %s and %s", "Try Ctrl-Alt-Delete to reboot");
    a1.SetAgingPolicy(0.5);
    // test raising to ERROR, then WARN, then ERROR within the age
    a1.Raise(Alarm::Severity_e::ERROR);
    EXPECT_EQ(Alarm::Severity_e::ERROR, a1.severity);
    a1.Raise(Alarm::Severity_e::WARNING);
    EXPECT_EQ(Alarm::Severity_e::ERROR, a1.severity);
    POSIX::usleep(300000);
    a1.Raise(Alarm::Severity_e::ERROR);
    EXPECT_EQ(Alarm::Severity_e::ERROR, a1.severity);
    a1.Raise(Alarm::Severity_e::WARNING);
    EXPECT_EQ(Alarm::Severity_e::ERROR, a1.severity);
    POSIX::usleep(300000);
    a1.Raise(Alarm::Severity_e::WARNING);
    EXPECT_EQ(Alarm::Severity_e::ERROR, a1.severity) << a1;
    POSIX::usleep(300000);
    a1.Clear();
    EXPECT_EQ(Alarm::Severity_e::WARNING, a1.severity);
    POSIX::usleep(300000);
    a1.Clear();
    EXPECT_EQ(Alarm::Severity_e::CLEAR, a1.severity);
}

class AlarmEx : public Alarm
{
public:
    using Alarm::Alarm;
    double GetMonotonicTime() const override { return PacBio::Utilities::Time::GetMonotonicTime() - offset_; }
    void SetTZero() { offset_ = PacBio::Utilities::Time::GetMonotonicTime();}
private:
    double offset_;
};

TEST(Alarm, AgingAtTzero)
{
    AlarmEx a1("primary.test.test", "Testing", "I need %s, %s and %s", "Try Ctrl-Alt-Delete to reboot");
    a1.SetTZero();
    a1.SetAgingPolicy(0.5);
    a1.Clear();
    EXPECT_EQ(Alarm::Severity_e::CLEAR, a1.severity);

    a1.SetAgingPolicy(0.0);
    a1.Clear();
    EXPECT_EQ(Alarm::Severity_e::CLEAR, a1.severity);
}


TEST(Alarm,Performance1)
{
    LogSeverityContext l(LogLevel::WARN);
    /// Tests that the Raise and Clear methods have less than 1 ms penalty
    /// or 1000/second throughput.

    // .. requirement:: Low Alarm Raise and Clear Overhead
    PacBio::IPC::MessageQueueLabel x("foo",46609);
    PacBio::IPC::MessageSocketPublisher xx(x);
    AlarmManager manager(xx,"svc://foo");

    Alarm a1("primary.test.test","Testing","I need %s, %s and %s","Try Ctrl-Alt-Delete to reboot");
    manager.Register(a1);

    const uint32_t iterations = 10000;
    manager.Enable();

    // This is the subscriber thread, to empty out the ZMQ queue
    std::promise<bool> ready;
    auto readyFuture = ready.get_future();
    std::future<uint32_t> rx = std::async(std::launch::async,[x,&ready,iterations](){
        PacBio::IPC::MessageSocketSubscriber receiver(x);
        receiver.SetTimeout(1000);
        uint32_t count=0;
        ready.set_value(true);
        while (true)
        {
            // TEST_COUT << "waiting.." << std::endl;
            PacBio::IPC::SmallMessage m1 = receiver.Receive();
            if (m1.IsAvailable())
            {
                count++;
                //TEST_COUT << "got " << count << std::endl;
                if (m1.GetData() == "" || count >= iterations) break;
            }
            else
            {
                TEST_COUT << "TIMEOUT! count:" << count << " expected:" << iterations << std::endl;
                break;
            }
        }
        return count;
    }) ;

    // wait for subscriber to be ready, so we don't miss anything
    readyFuture.get();
    //TEST_COUT << "ready!: " << std::endl;

    usleep(1000);
    PacBio::Dev::QuietAutoTimer timer(iterations );
    // now run the speed test
    for(uint32_t i = 0; i < iterations; ++i)
    {
       // TEST_COUT << "iter:" << i << std::endl;
        a1.Raise(Alarm::Severity_e::WARNING, "x", "y", "z");
        a1.Clear();
        manager.Service();
    }

    rx.wait();
    EXPECT_EQ(iterations,rx.get());

    TEST_COUT << "Alarm Rate: " << timer.GetRate() << std::endl;
#ifdef PB_MIC_COPROCESSOR
    EXPECT_GT(timer.GetRate() , 1000); // per second
#elif defined(__OPTIMIZE__)
    EXPECT_GT(timer.GetRate() , 10000); // per second
#else
    EXPECT_GT(timer.GetRate() ,  1000); // per second
#endif
}



TEST(Alarm,REST)
{
    PacBio::IPC::MessageQueueLabel x("foo",46609);
    PacBio::IPC::MessageSocketPublisher xx(x);
    AlarmManager manager(xx,"svc://pa/foo");
    Alarm a1("test.test","Testing","I need %s, %s and %s","Try Ctrl-Alt-Delete to reboot");
    manager.Register(a1);

    std::unique_ptr<CivetHandler> handler( manager.GetCivetHandler() );
    ASSERT_NE(nullptr, handler.get());

    PacBio::Dev::MockSocket ms;

    struct mg_connection* conn = mg_get_fake_connection(ms.GetSocket());
    handler->handleGet(nullptr,conn);

    std::string contents = ms.ReadAll();
    //TEST_COUT << contents << std::endl;

    int payload = contents.find("\r\n\r\n");

    ASSERT_GT(payload , 0);
    std::string ss = contents.substr(payload+4);

    Json::Value v = PacBio::IPC::ParseJSON(ss);
    ASSERT_EQ(v.size(),1);
    v = v[0];

    EXPECT_EQ("HTTP/1.1 200 OK\r\nContent-Type: text/json\r\nAccess-Control-Allow-Methods: POST, GET, OPTIONS\r\nAccess-Control-Allow-Headers: Origin, X-Requested-With, Content-Type, Accept\r\nAccess-Control-Allow-Origin: *",contents.substr(0,payload));
    EXPECT_EQ("0",v["id"].asString());
    EXPECT_EQ("",v["message"].asString());
    EXPECT_EQ("test.test",v["name"].asString());
    EXPECT_EQ("svc://pa/foo",v["source"].asString());
    EXPECT_EQ("CLEAR",v["severity"].asString());
}
