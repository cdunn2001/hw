import logging
import datetime
import os
import sys
import tempfile

from datetime import datetime

from Helpers import *

class ProgressManager:
    def __init__(self, filename):
        self.progressfile = filename
        self.currentState = [ ]
        # currentState: List[str] = [ ]
        self.callbacks = []

    def SetScope(self, scope, callback):
        self.callbacks.append(callback)

    def SetProgress(self, state):
        """Writes the state of the script to a temporary file, which can be read by another 
        program such as `watch`."""

        logging.debug("SetProgress:" + state)
        temp_file = "/tmp/progress_" + datetime.now().isoformat()
        with open(temp_file, "w+") as f:
            f.write("kpa-ws-test.py\n")
            f.write("state: %s\n --> %s\n" % ("\n ... ".join(self.currentState), state))
            f.write(" time: %s\n" % datetime.now())

            for callback in self.callbacks:
                f.write(callback())

            # url = "http://" + args.pachost + ':46602/.json'
            # response = requests.get(url)
            # r = response.json()
            # s = r["acquisition"]["status"]
            # f.write("-----pa-acq------ state: %s  time:%d\n" % (s["state"],s["time"]))
            # f.write(" currentFrame: %d currentFrameIndex: %d\n" % (s["currentFrame"],s["currentFrameIndex"]))
            # f.write(" frameRate: %f    movieFrame: %d\n" % (s["frameRate"],s["movieFrame"]))
            # f.write(" transmitStatus: %s transmitFrame: %d\n" % (s["transmitStatus"],s["transmitFrame"]))

            # url = "http://" + args.pachost + ':46612/.json'
            # response = requests.get(url)
            # r = response.json()
            # s = r["fromAcquisition"]
            # f.write("-----pa-t2b ------ state: %s\n" % r["state"])
            # f.write(" t2b tranchesProcessed: %d\n" % s["tranchesProcessed_"])

            # url = "http://" + args.pachost + ':46622/.json'
            # response = requests.get(url)
            # r = response.json()
            # f.write("-----pa-twb ------ state: %s\n" % r["state"])
            # f.write(" bw totalBytes: %d\n" % r["totalBytes"])

            # f.write("----pa-ws------\n")
            # global registeredUUIDs
            # for uuid in registeredUUIDs :
            #     url = "http://" + args.pawshost + ':8090/acquisitions/' + uuid + '/status'
            #     response = requests.get(url)
            #     r = response.json()
            #     if response.status_code / 100 != 2 :
            #         state = "ERROR - HTTP status %d " % response.status_code
            #     else :
            #         state = r['state']
            #     f.write(" %s -> %s\n" % (uuid, state))
        os.rename(temp_file, self.progressfile)


    def PushProgress(self, substate):
        self.currentState.append(substate)
        self.SetProgress("PUSH")


    def PopProgress(self):
        self.currentState.pop(-1)
        self.SetProgress("POP")



class ProgressScope(object):
    def __init__(self, progressManager, name):
        self.progressManager = progressManager
        self.name = name

    def __enter__(self):
        self.progressManager.PushProgress(self.name)

    def __exit__(self, exceptionType, exceptionValue, tb):
        self.progressManager.PopProgress()


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARN)

    td = tempfile.TemporaryDirectory()
    fn = td.name + "/progress.txt"
    try:
        pm = ProgressManager(fn)
        pm.SetProgress("starting")
        s = slurp(fn)
        logging.debug(s)
        if not "starting" in s:
            raise Exception("bad progress file state")
        if not "time:" in s:
            raise Exception("bad progress file state")


        with ProgressScope(pm, "constructor") as pp:
            s = slurp(fn)
            logging.debug(s)
            if not "constructor" in s:
                raise Exception("bad progress file state")
                
            if not "PUSH" in s:
                raise Exception("bad progress file state")
            pm.SetProgress("running")
            s = slurp(fn)
            logging.debug(s)
            if not "running" in s:
                raise Exception("bad progress file state")
            if not "time:" in s:
                raise Exception("bad progress file state")

        s = slurp(fn)
        logging.debug(s)
        if "constructor" in s:
            raise Exception("bad progress file state")
        if "running" in s:
            raise Exception("bad progress file state")
        if not "POP" in s:
            raise Exception("bad progress file state")
        if not "time:" in s:
            raise Exception("bad progress file state")

        pm.SetProgress("done")
        s = slurp(fn)
        logging.debug(s)
        print("ProgressHelper:PASS")
        sys.exit(0)
    except Exception as ex:
        print("ProgressHelper:FAIL")
        logging.error("Exception :%s",format(ex))
        sys.exit(1)
