import logging
import tempfile
import time
import unittest

from datetime import datetime
from subprocess import check_call, check_output
from time import sleep
from typing import List

from junit_xml import TestSuite, TestCase  # type: ignore
from lxml import etree  # type: ignore

import HttpHelper
import ProgressHelper
from Helpers import *

class SensorSim(HttpHelper.SafeHttpClient):
#    """Simulates a sensor, using loopback from wx-daemon. Communication is with wx-daemon via REST calls"""
    def __init__(self, url, progresser ):
        super(SensorSim, self).__init__()
        self.url = url
        self.progresser = progresser
        self.progresser.SetScope("SensorSim", self.StatusUpdate)
        self.startFrame = 0
        self.platform = self.GetPlatform()
        self.WaitForState("online")
        self.frameLatency = int(self.checkedGet(self.url + "/status/wolverine/frameLatency"))

    def StatusUpdate(self):
        return "aok"

    def StartFrame(self):
        return self.startFrame

    def SendFrames(self, numFrames, filename, frameRate=100, pixelRowDuplication=False, movieNumber =0):
        if self.GetBusy():
            raise Exception("Can't SendFrames, sensor is still busy")

        numFrames = numFrames + self.frameLatency

        tconfig = {
            "condensed": False,
            "enablePadding": False,
            "mode": "GENERATED",
            "hdf5input": filename, # "CONSTANT/123",
            "limitFileFrames":512,
            "linerate":0,
            "movieNumber": movieNumber,
            "rate":frameRate,
            "frames":numFrames,
            "startFrame":self.startFrame  #-1
        }
#        if pixelRowDuplication :
#            command.append('--condensed')
#            command.append('--pixelrowduplication')
        logging.info('SensorSim.SendFrames: %s', tconfig)
        self.checkedPost(self.url + "/sras/0/transmit", tconfig)

        self.progresser.SetProgress("StartingTransmit")

        self.progresser.SetProgress("Busy")
        logging.info("SensorSim.SendFrames: busy")
        # Sensor frame index will continue from last frame after first movie transmission.
        self.startFrame = -2

    def GetTransmitStatus(self):
        """returns one of IDLE, PRELOADING, TRANSMITTING"""
        return self.checkedGet(self.url + "/sras/0/status/transmitter/state")

    def GetBusy(self):
        """returns True if IDLE"""
        ts = self.GetTransmitStatus()
        logging.debug("GetBusy: transmit status:%s" % ts)
        return ts != "IDLE"

    def WaitUntilPreloaded(self):
        self.progresser.SetProgress("WaitUntilPreloaded")
        logging.info("SensorSim.WaitUntilPreloaded: ")

        timeout = 600 # 10 minutes
        t0 = time.monotonic()
        while self.GetTransmitStatus() == "PRELOADING":
            self.progresser.SetProgress("preloading" )
            t = time.monotonic()
            if t - t0 > timeout :
                raise RealtimeException("waiting for preloading state to change timed out, waited for %f seconds" %
                                        timeout)
            sleep(.05)
        if self.GetTransmitStatus() == "IDLE":
            raise Exception("Transmission is idle already")

        logging.info("SensorSim.WaitUntilPreloaded: busy but not preloading anymore. Transmission should start now ")

    def GetCurrentFrameIndex(self):
        index = self.checkedGet(self.url + "/sras/0/status/transmitter/frameIndex")
        if index == 18446744073709551615:
            return None
        else:
            return index


    def WaitUntilNotBusy(self):
        logging.info("SensorSim.WaitUntilNotBusy: Waiting on frame %s", self.GetCurrentFrameIndex())
        while self.GetBusy():
            ts = self.GetTransmitStatus()
            logging.info("transmit status frame:%d transmitstate:%s" %
                         (self.GetCurrentFrameIndex(), ts))
            self.progresser.SetProgress("still waiting in SensorSim.WaitUntilNotBusy, frame(%d)" %
                        (self.GetCurrentFrameIndex()))
            sleep(1)

    def WaitForState(self,state):
        self.progresser.SetProgress("WaitForState:%s" % (state))
        logging.info('Pac.WaitForState: %s' %  state)
        while True:
            r = self.checkedGet(self.url + "/sras/0/status/state")
            logging.debug("current state:%s" % r)
            if r == state:
                return
            sleep(1)    

    def GetPlatform(self):
        s = self.checkedGet(self.url+'/status') 
        return s['platform']


class TestSensorSim(unittest.TestCase):
    def test_one(self):    
        import WxDaemonSim

        td = tempfile.TemporaryDirectory()
        fn = td.name + "/progress.txt"
        progressor = ProgressHelper.ProgressManager(fn)

        wxdaemon = WxDaemonSim.WxDaemonSim()
        wxdaemon.Run()
        ss = SensorSim(wxdaemon.GetUrl(), progressor)
        wxdaemon.Shutdown()
        td.cleanup()

if __name__ == '__main__':
    unittest.main()
