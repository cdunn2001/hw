#!/bin/bash
# A little hack to start in bash, source the virtual env for python and continue in python.
# Putting double quotes around words are ignored in python, but evaluated in bash.

"echo" "switching from default python to VE python ..."
"export" "SCRIPT=`realpath $0`"
"export" "SCRIPTPATH=`dirname $SCRIPT`"
"echo" "running from $SCRIPTPATH"
"which" "python"
"python" "--version"
"." "$SCRIPTPATH/install_python3.sh"
"which" "python"
"python" "--version"
"exec" "python3" "$SCRIPT" "$@"

# the rest is python

import argparse
from datetime import datetime
import getpass
import glob
import json
import h5py  # type: ignore
from junit_xml import TestSuite, TestCase  # type: ignore
import logging
from lxml import etree  # type: ignore
import numpy  # type: ignore
import os
import requests
import signal
import string
from subprocess import check_call, check_output
import sys
import tempfile
from time import sleep
import traceback
from typing import List

sys.path.append(sys.path[0] + "/PaWsTest")
import Acquisition
from Helpers import RealtimeException, TerminateNamedProcess
# from HttpHelper import SafeHttpClient
from KestrelRT import RT
from ProgressHelper import ProgressManager, ProgressScope

# Globals
testSuite = TestSuite('kpa-ws test suite', [])
args = argparse.Namespace()



class Statistics(dict):
    def __missing__(self,key):
        if 'count' in str(key): # this allows key to be numeric or stringy
            value = self[key] = 0
        else:
            value = self[key] = type(self)()
        return value


stats = Statistics()


def signalExitHandler(sig, frame):
    global stats
    print('Caught signal %s at frame %s' % (sig, frame))
    logging.critical('Caught signal %s at frame %s' , sig, frame)
    print("End Stats (Exception): %s" % stats)
    logging.info("End Stats (Exception): %s",stats)
    #    signal.raise_signal(sig)
    sys.exit(143)


def signalHupHandler(sig, frame):
    global stats
    print('Caught signal %s at frame %s' % (sig, frame))
    print("End Stats (Exception): %s" % stats)


signal.signal(signal.SIGINT, signalExitHandler)
signal.signal(signal.SIGTERM, signalExitHandler)
signal.signal(signal.SIGHUP, signalHupHandler)

scriptDir = os.path.dirname(os.path.abspath(__file__))

def KillWxDaemon():
    logging.warning("Terminating wx-daemon!")
    TerminateNamedProcess("pa-acq/main")
    logging.warning("Terminating wx-daemon!")


# edit this function to add a user breakpoint to help in debug
def UserBreakPoint():
    pass
    # input("Press Enter to continue... dark frame done")

def SetupLogging():
    global args
    root = logging.getLogger()
    root.setLevel(getattr(logging, args.loglevel.upper(), None))

    formatter = logging.Formatter('%(asctime)s %(levelname)s - %(message)s')
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, args.loglevel.upper(), None))
    ch.setFormatter(formatter)
    root.addHandler(ch)

    if args.logfile != "":
        logch = logging.FileHandler(args.logfile)
        logch.setLevel(getattr(logging, args.loglevel.upper(), None))
        logch.setFormatter(formatter)
        root.addHandler(logch)


class EndToEnd:
    def __init__(self, progressMonitor):
        self.progressMonitor = progressMonitor
        self.sensor = None # simulates the sensor. Controlled via pa-acq-cli. Could be the same host as self.rt
        self.rt = None    # simulates the PAC. Controlled via pa-acq-cli. Could be the same host as self.sensor
        self.verifier = None  # PPA based verification
        self.smrtlinkimport = None  # tests that the final data can be imported successfully to SMRTLink
        self.overallresult = 'FAIL'  # until the end of Setup()
        self.darkframecalfiles = {}
        self.movieNumber = 100 # this number is arbitrary. It will increment by one for each app

    def Setup(self):
        global args
        # Check hosts running tests.
        if args.loopback:
            logging.info("EndToEnd.Setup: Running test in loopback mode - setting --sensorhost and --pachost to localhost")
#            args.sensorhost = 'localhost'
#            args.pachost = 'localhost'
#            args.pawshost = 'localhost'

        quickTest : List(int) = [ args.wxurl, args.pawsurl, args.sensorurl ]
        for url in quickTest:
            try:
                response = requests.get(url +  "/status")
            except Exception :
                logging.error("%s not responding", url)
                sys.exit(1)

        self.progressMonitor.SetProgress("setup")

        if args.expedite:
            args.numdarkcalframestx = 2048
            args.darkcalstartframe = 200
            args.numdarkcalframes = 100

            args.numloadcals = 1
            args.numloadcalframestx = 2048
            args.numloadcalframes = "100"   # csv format
            args.loadcalstartframe= "2048"  # csv format
            args.loadcalexpcount  = "200.0" # csv format

        # Check loading calibration options are sane.
        args.loadcalstartframe = [int(s) for s in args.loadcalstartframe.split(',')]
        args.loadcalexpcount = [float(s) for s in args.loadcalexpcount.split(',')]
        args.numloadcalframes = [int(s) for s in args.numloadcalframes.split(',')]

        assert(len(args.loadcalstartframe) == len(args.loadcalexpcount))
        assert(len(args.loadcalexpcount) == len(args.numloadcalframes))
        assert(len(args.numloadcalframes) == args.numloadcals)
        self.overallresult = 'PASS'

    def Run(self):
        """ returns exit code for process"""
        global args
        try:
            self.sensor = SensorSim(url=args.sensorurl, progresser=self.progressMonitor)
            self.rt = RT(self.progressMonitor)
            self.rt.wxdaemon = args.wxurl
            self.rt.paws = args.pawsurl
#            self.verifier = Verification()

            self.progressMonitor.SetProgress("VerifyPipelineIsUp")
            socket = "1"
            self.rt.VerifyPipelineIsUp() # raises exception if not up
            self.rt.Reset(socket)
#            self.rt.DeleteFiles()

            acquisitions = []
            acquisitions.append( Acquisition.Acquisition("m1234"))

            logging.info("%d acquisitions are all configured", len(acquisitions))

            for acq in acquisitions:
                logging.info("RunOne %s" % acq.mid)
                self.progressMonitor.SetProgress("RunOne %s" % acq.mid)
                try:
                    self.RunOne(acq)
                except Exception as ex:
                    if args.keepgoing:
                        logging.warning("continuing after exception \"%s\"", ex)
                        self.overallresult= 'FAIL'
                    else:
                        raise
            logging.info("Runs have all RunOne")

            logging.info("FinishingEverything")
            self.progressMonitor.SetProgress("FinishEverything")
            return self.FinishEverything()

        except Exception as ex:
            logging.error('Exception caught during EndToEnd::Run ' + str(ex))
            self.progressMonitor.SetProgress("EXCEPTION: " + str(ex))
            tc = TestCase(str(ex))
            tc.add_failure_info(output=traceback.format_exc())
            testSuite.test_cases.append(tc)
            self.overallresult = 'FAIL'
            return 1

    def RunOne(self,acq):
        global args
        try:
            with ProgressScope(self.progressMonitor, "%s" % acq) as p0:
                logging.info("RunOne: Starting %s" % acq)
                self.progressMonitor.SetProgress("allocatespace")

                with ProgressScope(self.progressMonitor,"DoCalibrations") as p1:
                    self.DoCalibrations(acq)

                with ProgressScope(self.progressMonitor, "DoAcquisition") as p2:
                    self.DoAcquisition(acq)

        except RealtimeException as ex:
            logging.error("Caught Realtime Exception during %s: %s." % acq, str(ex))
            raise Exception("RunOne failed" + str(ex))
            # logging.warn("Caught Realtime Exception, %s. Continuing with next acquisition." % str(ex))

    def FinishRun(self,run):
        """ Check all acquisitions for this run completed successfully."""
        global args
        for i, id in enumerate(run.ids):
            if run.exception:
                requests.delete(self.rt.paws + '/acquisitions/' + id)
            else:
                # Boost the timeout to account for data transfer. We should
                # maybe figure out what the data transfer payload is and
                # account for it.
                dataTransferAddedTimeMin = 60
                self.rt.WaitFor(id, 'Complete', (run.movieLengths[i] + dataTransferAddedTimeMin) * 60)
                self.rt.CheckCompletionStatus(id)
                self.rt.CheckRtMetricsStatus(id)

                errors = self.rt.GetErrors(id)
                if len(errors) != 0:
                    raise Exception("errors caught " + str(errors))

                subreadsetXml = self.rt.GetResults(id)
                # one of these replacements should do something.
                resultsPath = subreadsetXml.replace('.subreadset.xml', '')
                resultsPath = resultsPath.replace('.consensusreadset.xml', '')
                canary = resultsPath + '.transferdone'
                if not os.path.exists(canary):
                    raise Exception("{} not found".format(canary))
                else:
                    logging.info("FinishRun: Found file {}".format(canary))

                self.verifier.Run(resultsPath, self.darkframecalfiles[id])

                if args.smrtlinkimport:
                    self.smrtlinkimport.Run(subreadsetXml)
                else:
                    self.verifier.RunPbValidate(subreadsetXml)

                collectionDir = os.path.dirname(resultsPath)
                command = ['ls',
                           '-la',
                           collectionDir]
                logging.info("FinishRun: %s", command)
                check_call(command)

    def FinishEverything(self):
        """ returns suggested exit code for process"""

        global args

        tc = TestCase('end-to-end-overall')
        if self.overallresult != 'PASS':
            tc.add_failure_info(output='exception caused test to fail')
        testSuite.test_cases.append(tc)

        logging.info('EndToEnd.Run: overallresults = %s', self.overallresult)

        if args.junitxml:
            logging.info('EndToEnd.Run: Writing junitxml results to %s', args.junitxml)
            with open(args.junitxml, 'w') as f:
                TestSuite.to_file(f, [testSuite])

        exitCode = 0 if self.overallresult == 'PASS' else 1
        return exitCode

    def _checkCalibrationFile(self, filename, dname, expval, expstartframe):
        global args
        if os.path.isfile(filename):
            logging.info("EndToEnd._checkCalibration: filename = %s, dname = %s", filename, dname)
            f = h5py.File(filename, 'r')
            dset = f[dname]
            # There should be a REST endpoint that returns the configured ROI, instead of this sill
            # guessing game.
            # FIXME
            platform = self.rt.GetPlatform()
            if 'Sequel1' in platform:
                val = numpy.average(dset[32:32+1080, 64:64+1920])
            elif 'Benchy' in platform:
                val = numpy.average(dset[0:1360, 0:1440])
            elif 'Sequel2' in platform:
                    val = numpy.average(dset[0:2756, 0:2912])
            else:
                raise Exception('platform ' + platform + ' not supported in _checkCalibrationFile')

            startFrame = dset.attrs['Frame_StartIndex']
            expIsClose = (abs(val - expval) / (expval + 0.0000001)) <= 0.05
            frameMatches = startFrame == expstartframe
            if expIsClose:
                logging.info("EndToEnd._checkCalibration: val= %f, expval= %f", val, expval)
            else:
                logging.error("EndToEnd._checkCalibration: val= %f, expval %f", val, expval)
            if frameMatches:
                logging.info("EndToEnd._checkCalibration: startFrame= %s, expstartframe= %s", startFrame, expstartframe)
            else:
                logging.error("EndToEnd._checkCalibration: startFrame= %s, expstartframe= %s", startFrame, expstartframe)
            reason = "pattern(%s vs %s) startFrame(%s vs %s)" % (val, expval, startFrame, expstartframe)

            if args.waiver >= 1 and not expIsClose :
                expIsClose = True
                logging.info("waiving expIsClose match!")
            if args.waiver >= 2 and not frameMatches :
                frameMatches = True
                logging.info("waiving exp frame match!")
            return expIsClose and frameMatches, reason

        else :
            logging.info("EndToEnd._checkCalibration: filename = %s does not exist, " + 
                         "might be a different machine, skipping test", filename)
            return True, "skipping test because calibration file not found"

    def DoCalibrations(self, acq):
        global args
        platform = self.rt.GetPlatform()

        # localPaName is the name that the local components refer to each other through the local network

        logging.info('EndToEnd.DoCalibrations: ao = %s', str(acq)[0:100]+"...")
        logging.debug('EndToEnd.DoCalibrations: ao = %s', str(acq))
        cellId = acq.cellId

        self.progressMonitor.SetProgress("WaitUntilNotBusy")
        self.sensor.WaitUntilNotBusy()
        currentframe = self.sensor.GetCurrentFrameIndex()
        if currentframe == None or self.sensor.StartFrame() == 0:
            currentframe = 0
        logging.info("EndToEnd._doCalibrations: SendFrames ... darkcal, startFrame = %d, currentframe = %d",
                     self.sensor.StartFrame(),
                     currentframe)

        with ProgressScope(self.progressMonitor,"preloadingDynamicCalFrames") as p30:
            self.sensor.SendFrames(args.numdarkcalframestx, args.darkcalfile, frameRate=args.framerate, movieNumber=self.movieNumber)
            logging.info("EndToEnd.DoCalibrations: before preload, currentframe %d",self.sensor.GetCurrentFrameIndex())
            self.sensor.WaitUntilPreloaded()
            logging.info("EndToEnd.DoCalibrations: after preload, currentframe %d",self.sensor.GetCurrentFrameIndex())

        with ProgressScope(self.progressMonitor,"DoDarkCalibration") as p3:
            calFileUrl = self.DoDarkCalibration(acq)
            #Test.assertIn('file://' + localPaName + '/data/pa/cal/' + acq.mid + '_dark_', co['url'])

            currentframe = self.sensor.GetCurrentFrameIndex()
            logging.info("Pac._doCalibrations: loadcal, currentframe = %s", currentframe)

        self.sensor.WaitUntilNotBusy()

        self.movieNumber +=1
        offsetFrame = self.sensor.GetCurrentFrameIndex()
        if offsetFrame == None or self.sensor.StartFrame() == 0:
            offsetFrame = 0

        with ProgressScope(self.progressMonitor,"preloadingDynamicCalFrames") as p50:
            self.sensor.SendFrames(args.numloadcalframestx, args.loadcalfile, frameRate=args.framerate, movieNumber=self.movieNumber)
            self.sensor.WaitUntilPreloaded()

        loadinfo = []
        for loadcaliter in range(0, args.numloadcals):
            logging.info('EndToEnd._doCalibrations: Doing dynamic calibration # %s', loadcaliter)
            with ProgressScope (self.progressMonitor,"DoDynamicLoadingCalibration %d" % loadcaliter) as p5:
                li = self.DoDynamicLoadingCalibration(acq, loadcaliter, offsetFrame)
                loadinfo.append(li)
                UserBreakPoint()

                self.sensor.WaitUntilNotBusy()

                self.movieNumber +=1

        all_load_checks = True
        for loadcaliter, li in enumerate(loadinfo):
            logging.info("EndToEnd._doCalibrations: loadcal,  %s", li)
            # co = self.rt.GetCalObject(li[0])
            # Test.assertIn('file://' + localPaName + '/data/pa/cal/' + cellId + '_loading', co['url'])
            # p = re.compile('^file://[^/]+') # remove something like "file://pac" or "file://rt"
            # filename = p.sub('',co['url'])
            # loadCheck, reason = self._checkCalibrationFile(filename,
            #                                        '/Loading/LoadingMean',
            #                                        args.loadcalexpcount[loadcaliter], li[1])

            # if args.fastcheck == "loadstartframe":
            #     # This case is for fast debugging of the startFrame for loading calibrations.
            #     # While in the for loop, We just stash the results of the loading stage, and then when all the
            #     # loading steps are done, we exit the acquisition entirely with a soft exception so we
            #     # can skip the actual data capture and go straight to the next run acquisition.
            #     if not loadCheck:
            #         all_load_checks = False # this means at least one load calibration failed
            #     logging.info("loadstartframe loadCheck:%s reason:%s loadcaliter:%d filename:%s", loadCheck, reason,
            #                  loadcaliter, filename)
            #     if loadCheck:
            #         stats[args.fastcheck][loadcaliter]['successes_count'] += 1
            #     else:
            #         stats[args.fastcheck][loadcaliter]['failures_count'] += 1
            #         key = "%s_count" % reason
            #         stats[args.fastcheck][loadcaliter][key] += 1
            # else:
            #     # this statement will raise an exception and terminate this function
            #     Test.assertTrue(loadCheck ,  "_checkCalibrationFile for " + filename)

    def DoDarkCalibration(self, acq):
        global args
        logging.info('EndToEnd.DoDarkCalibration: %s', acq)
        self.rt.StartDarkcal("1", acq, self.movieNumber)
        self.rt.WaitForState("1","darkcal","RUNNING")
        self.rt.WaitForState("1","darkcal","COMPLETE")

        logging.info('EndToEnd.DoDarkCalibration: done')
        return self.rt.GetApp("1","darkcal")


    def DoDynamicLoadingCalibration(self, acq, loadcaliter, frameOffset):
        global args
        logging.info('EndToEnd.DoDynamicLoadingCalibration: %s', acq)
        self.rt.StartLoadingcal("1", acq, self.movieNumber)
        self.rt.WaitForState("1","loadingcal","RUNNING")
        self.rt.WaitForState("1","loadingcal","COMPLETE")
        logging.info('EndToEnd.DoDynamicLoadingCalibration: done')
        return self.rt.GetApp("1","loadingcal")


    def DoAcquisition(self, acq):
        global args
        global args
        logging.info('EndToEnd.DoAcquisition: %s', acq)
        self.rt.StartBasecaller("1", acq, self.movieNumber)
        self.rt.WaitForState("1","basecaller","RUNNING")
        self.rt.WaitForState("1","basecaller","COMPLETE")
        logging.info('EndToEnd.DoAcquisition: done')
        return self.rt.GetApp("1","basecaller")

def SecureCopySimFileAndGetNewName(filename):
    """ Copies a file name from the local machine to the remote sensorhost using rsync.  This is useful
    if the sensorhost does not have NFS mounted disks. The files are placed in args.simfiledir.
    The path to the copied file is returned."""
    global args
    destination_file = args.simfiledir + "/" + os.path.basename(filename)

    command=[ "ssh",
              "-i", os.path.normpath(args.simkeyfile),
              args.simuser + "@"+ args.sensorhost,
              " mkdir -p " + args.simfiledir]
    check_call(command)
    command=["rsync",
             "--progress", "-e",
             'ssh -oBatchMode=yes -v -i ' +args.simkeyfile+'',
             filename, args.simuser +"@"+args.sensorhost+":"+destination_file]
    check_call(command)

    return destination_file

def SecureCopySimFiles():
    """Copies all local simulation files to the remote sensorhost, and rewrites the command line arguments.
    The args.datafile is a special case. It is parsed locally to get the run metadata, so the local path
    remains as args.datafile and the remote file is put into args.datafile_on_sensorhost."""
    global args
    args.datafile_on_sensorhost = SecureCopySimFileAndGetNewName(args.datafile)
    args.darkcalfile = SecureCopySimFileAndGetNewName(args.darkcalfile)
    args.loadcalfile = SecureCopySimFileAndGetNewName(args.loadcalfile)

    print("Files have been copied to %s and will be used locally on the sensor host" % args.sensorhost)
    print("args.datafile_on_sensorhost:%s" % args.datafile_on_sensorhost)
    print("args.darkcalfile:%s" % args.datafile)
    print("args.loadcalfile:%s" % args.loadcalfile)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='End-to-end integration test simulating ICS')

    #
    # Test setup options
    #
    group = parser.add_argument_group('Test Setup')
    group.add_argument('--loglevel',
                       help="Log level of end-to-end test script log messages",
                       default='INFO',
                       choices=['WARNING', 'INFO', 'DEBUG'])
    group.add_argument('--logfile',
                       help="Log file name",
                       default='')
    group.add_argument('--sensorurl',
                       help="URL to wx-daemon hosting the transmitting. Default = %(default)s",
                       default='http://localhost:23602')
    group.add_argument('--wxurl',
                       help="URL to wx-daemon hosting acquisition. Default = %(default)s",
                       default='http://localhost:23602')
    group.add_argument('--pawsurl',
                       help="URL to pa-ws. Default = %(default)s",
                       default='http://localhost:23632')
    group.add_argument('--loopback',
                       help="Set --sensorhost and --pachost to localhost",
                       action='store_true',
                       default=False)
    group.add_argument('--junitxml', help="Filename to write JUnit XML results",
                       default=None)
    group.add_argument('--outputprefix',
                       help="Local directory on --pachost for results",
                       default='/data/pa')
    group.add_argument('--progressfile',
                       default="/tmp/progress_" + getpass.getuser()+
                               "_" + datetime.now().isoformat() +".txt",
                       help="File to which progress is written every few seconds. Default = %(default)s")
    group.add_argument('--bist',
                       action='store_true',
                       help="Runs a built in self test (BIST)")
    group = parser.add_argument_group('Simulation Files Setup')
    group.add_argument('--scpsimfiles',
                       default=False,
                       action='store_true',
                       help='When set, all of the simulation files will be copied to the sensorhost default: %(default)s')
    group.add_argument('--simfiledir',
                       default='/data/pa/simfiles',
                       help="Destination directory on sensorhost to store sim files for transmission default: %(default)s")
    group.add_argument('--simuser',
                       default="pbi",
                       help="User on sensorhost to receive the simfiles default: %(default)s")
    group.add_argument('--simkeyfile',
                       default=scriptDir + "/../common/pbi.id_rsa",
                       help="RSA private key file to use to log into sensorhost default: %(default)s")
    #
    # Acquisition calibration options
    #

    group = parser.add_argument_group('Dark Calibration Setup')
    # Dark calibration
    group.add_argument('--darkcalfile',
                       help="Dark calibration file to use. default: %(default)s",
                       default='/pbi/dept/primary/sim/sequel/sequel_calpattern_frm512.trc.h5')
    group.add_argument('--numdarkcalframestx',
                       help="Number of dark cal frames to transmit. default: %(default)s",
                       default=3000,type=int)
    group.add_argument('--darkcalstartframe',
                       help="Dark calibration start frame to use. default: %(default)s",
                       default=2510, type=int)
    group.add_argument('--darkcalexpcount',
                       help="Expected dark calibration counts. default: %(default)s",
                       default=100.0, type=float) # the sim file above consists many frames of "100", and the startframe points to these.
    group.add_argument('--numdarkcalframes',
                       help="Number of dark calibration frames to use. default: %(default)s",
                       default=100, type=int) # this is the number of the frames in the sim file, and also close to production dark frame movies.

    group = parser.add_argument_group('Loading Calibration Setup')
    # Loading calibration
    group.add_argument('--loadcalfile',
                       help="Loading calibration file to use. default: %(default)s",
                       default='/pbi/dept/primary/sim/sequel/sequel_calpattern_frm512.trc.h5')
    group.add_argument('--numloadcalframestx',
                       help="Number of loading cal frames to transmit. default: %(default)s",
                       default=25000,type=int)
    group.add_argument('--loadcalstartframe',
                       help="Loading calibration start frames to use (CSV format). default: %(default)s",
                       default='4196,10290,14923')
    group.add_argument('--loadcalexpcount',
                       help="Expected loading calibration counts (CSV format). default: %(default)s",
                       default='200.0,100.0,150.0')
    group.add_argument('--numloadcalframes',
                       help="Number of loading calibration frames to use (CSV format). default: %(default)s",
                       default='100,100,100')
    group.add_argument('--numloadcals',
                       help="Number of loading calibrations to perform. default: %(default)s",
                       default=3, type=int)  # TODO The code only supports 1 loadingcal right now.

    #
    # Acquisition options
    #
    group = parser.add_argument_group('Acquisition')
    group.add_argument('--framerate',
                       help="Acquisition frame rate (also used by --sensorhost for transmitting frames)",
                       default=100.0, type=float)
    group.add_argument('--runmetadata',
                       help="Acquisition run metadata to load",
                       default=os.path.dirname(os.path.abspath(__file__)) + '/test/m54004_170308_194231.metadata.xml')
    group.add_argument('--datafile',
                       help="Trace file simulation for acquisition",
                       default='/pbi/dept/primary/sim/sequel/designerE2E_Frm512_SNR-40.trc.h5')
    group.add_argument('--pixelrowduplication',
                       action='store_true',
                       help="Tile long-narrow movie in ZMW space",
                       default=False)
    group.add_argument('--notrace',
                       action='store_true',
                       help="Don't collect trace file for acquisition",
                       default=False)
    group.add_argument('--roi',
                       help="Override the sequencing ROI")
    group.add_argument('--traceroi',
                       default="default",
                       help="Override the tracefile ROI. Specify a using [[...]] notation, " +
                            "or give filepath, or specify \"none\", \"256\", \"16k\" or \"default\". " +
                            " Default is \"%(default)s\"")

    #
    # simulated failures
    #
    group = parser.add_argument_group('Failure Simulations')
    group.add_argument('--usefakecrosstalkfilter', help="Inject fake crosstalk filter for acquisition JSON update",
                       action='store_true',
                       default=False)
    group.add_argument('--usefakepsfs', help="Inject fake PSFs for acquisition JSON update",
                       action='store_true',
                       default=False)
    group.add_argument('--usefakespectrumvalues',
                       help="Inject fake spectrum values in the analogs for acquisition JSON update",
                       action='store_true',
                       default=False)
    group.add_argument('--failure',
                       help="list of failures to simulate, named by Jira story key, e.g. --failure=SEQ-1234",
                       action='append')
    group.add_argument('--expedite',
                       help='expedites the running to test architecture over accuracy',
                       action='store_true')

    # Run options
    group.add_argument('--numruns',
                       help="Number of back-to-back runs to simulate",
                       default=1, type=int)
    group.add_argument('--keepgoing',
                       action='store_true',
                       default=False,
                       help="if True, then runs will be run even if a run has an exception")
    group.add_argument('--fastcheck',
                       choices=['darkstartframe','loadstartframe'],
                       help="set this to run to a particular point quickly"
                       )
    group.add_argument('--acqRange',
                       type=int,
                       default=1,
                       help='Repeats the subreadsets in the run XML this many times')
    group.add_argument('--runSetupOnly',
                       action='store_true',
                       help='returns after the runs are set up')

    #
    # Data Verification options
    #
    group = parser.add_argument_group('Data Verification')
    group.add_argument('--zmwmatchthreshold',
                       help="ZMW match threshold for designer trace verification (set to -1 to disable verification)",
                       default=10000, type=int)
    group.add_argument('--verifymovie',
                       help="Run verification on collected movie file",
                       action='store_true', default=False)
    group.add_argument('--deconv',
                       help="Path to deconv utility for movie verification",
                       default=os.path.dirname(os.path.abspath(__file__)) +
                        '/../acquisition/build/x86_64/Release/deconvolutionSim/deconv')
    group.add_argument('--sequel_movie_diff',
                       help="Path to sequel_movie_diff utility for movie verification",
                       default=os.path.dirname(os.path.abspath(__file__)) +
                        '/../acquisition/build/x86_64/Release/pacbio-primary/movieDiffTool/sequel_movie_diff')
    group.add_argument('--smrtlinkimport',
                       help="Use pbservice and import data into smrtlink",
                       action='store_true',
                       default=False)
    group.add_argument('--waiver',default=0,type=int)

    try:
        args = parser.parse_args()

        SetupLogging()

        td = tempfile.TemporaryDirectory()
        if args.bist:
            from PaWsSim import PaWsSim
            from SensorSim import SensorSim
            from WxDaemonSim import WxDaemonSim
            wxdaemon = WxDaemonSim()
            wxdaemon.Run()
            paws = PaWsSim()
            paws.Run()
            args.sensorurl = wxdaemon.GetUrl()
            args.wxurl = wxdaemon.GetUrl()
            args.pawsurl = paws.GetUrl()
            args.progressfile = td.name + "/progress.txt"

        if args.failure is None:
            args.failure = [ ]

        if args.scpsimfiles:
            SecureCopySimFiles()
        else:
            args.datafile_on_sensorhost = args.datafile

        progressManager = ProgressManager(args.progressfile)
        e2e = EndToEnd(progressManager)
        e2e.Setup()
        exitCode = e2e.Run()
        logging.info("End Stats: %s",stats)
        sys.exit(exitCode)

    except Exception as ex:
        logging.exception('Exception caught during main, something is really broken!')
        tc = TestCase(str(ex))
        tc.add_failure_info(output=traceback.format_exc())
        testSuite.test_cases.append(tc)
        logging.info('Exception thrown causing end-to-end to fail')
        logging.info("End Stats (Exception): %s",stats)
        sys.exit(1)

