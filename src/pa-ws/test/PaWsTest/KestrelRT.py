import logging
import pytest
import tempfile
from time import monotonic, sleep


import HttpHelper
import ProgressHelper
import Acquisition

class RT(HttpHelper.SafeHttpClient):
    def __init__(self, progressMonitor):
        super(RT, self).__init__()
        self.progressMonitor = progressMonitor
        self.paws = 'http://localhost:23632'
        self.wxdaemon = 'http://localhost:23602'
        self.ids = [] # all IDs from all runs TODO: delete this hopefully
        self.exposure = 0.01
        self.crosstalkFilter = [ [1.0] ]

    def Init(self):        
        pass
#        with ProgressScope(self.progressMonitor, "PAC constructor") as pp:
#            self.WaitForState(1,'idle')

    def GetPlatform(self):
        return self.checkedGet(self.wxdaemon+'/status/platform')

    def Reset(self, socket, app=""):
        logging.info('RT.Reset:')
        if app == "":
            self.checkedPost("%s/sockets/%s/reset" % ( self.paws, socket), payload={})
        else :
            self.checkedPost("%s/sockets/%s/%s/reset" % ( self.paws, socket, app), payload={})

    def WaitForState(self,socket,app,allowedState,timeout=5):  
        self.WaitForStates(socket,app,[allowedState],timeout)

    def WaitForStates(self,socket,app,allowedStates,timeout=5):  
        self.progressMonitor.SetProgress("WaitForState:%s/%s=%s" % (socket,app,allowedStates))
        t0 = monotonic() 
        while True:
            stateActual = self.checkedGet("%s/sockets/%s/%s" % ( self.paws, socket, app ))
            stateActual = stateActual["processStatus"]["executionStatus"]
            if stateActual in allowedStates:
                logging.info('RT.WaitForState: DONE %s/%s %s' % (socket,app,allowedStates))
                return
            if monotonic() - t0 > timeout:
                raise TimeoutError("timeout waiting for %s, most recent state was %s" % (allowedStates,stateActual))    
            sleep(0.100)

    def WaitForArmed(self,socket,app,armed=True,timeout=5):  
        self.progressMonitor.SetProgress("WaitForArmed:%s/%s=%s" % (socket,app,armed))
        t0 = monotonic() 
        while True:
            armedActual = self.checkedGet("%s/sockets/%s/%s" % ( self.paws, socket, app ))
            armedActual = armedActual["processStatus"]["armed"]
            if armedActual == armed:
                logging.info('RT.WaitForArmed: DONE %s/%s %s' % (socket,app,armed))
                return
            if monotonic() - t0 > timeout:
                raise TimeoutError("timeout waiting for armed==%s" % (armed))    
            sleep(0.100)

    def GetCompletionStatus(self,socket,app):
        r = self.checkedGet("%s/sockets/%s/%s" % ( self.paws, socket, app ))
        r = r["processStatus"]
        if r["executionStatus"] != "COMPLETE":
            raise Exception("Can't get completionStatus if the executionStatus is not COMPLETE, was %s" % r["executionStatus"])
        return r["completionStatus"]

    def StartApp(self,socket,app,payload0, movieNumber):
        logging.info('RT.Start:')
        payload0["movieNumber"] = movieNumber
        self.checkedPost("%s/sockets/%s/%s/start" % ( self.paws, socket, app), payload=payload0, timeout=100000) # FIXME smaller timeout. 

    def StartBasecaller(self,socket, acq, movieNumber):
        logging.info('RT.StartBasecaller:')
        # TODO: generate the payload using AcquisitionUpdate from acq
        self.StartApp(socket,"basecaller", acq.GenerateBasecallerJsonPayload(), movieNumber)

    def StartDarkcal(self,socket, acq, movieNumber):
        logging.info('RT.StartDarkcal:')
        self.StartApp(socket,"darkcal", acq.GenerateDarkcalJsonPayload(), movieNumber)

    def StartLoadingcal(self,socket, acq, movieNumber):
        logging.info('RT.StartLoadingcal:')
        self.StartApp(socket,"loadingcal", acq.GenerateLoadingcalJsonPayload(), movieNumber)

    def GetApp(self, socket, app):
        return self.checkedGet("%s/sockets/%s/%s" % ( self.paws, socket, app))

    def VerifyPipelineIsUp(self):
        try:
            r = self.checkedGet(self.wxdaemon + '/status')
            logging.info('RT.VerifyPipelineIsUp: %s', r)
        except Exception as ex:
            logging.error("Exception caught during GET of /status. " + str(ex)[:400] + "...<snip>")
            raise Exception("Could not get component-versions from " + self.paws + ". Likely pa-ws is not running")


    def PaRealtimeIsReady(self):
        r = self.checkedGet(self.paws + '/system/status')
        return r["status"] == "up"

    # def AcquisitionPost(self,run):
    #     payload = {'rundetails': run.rmd}

    #     t0 = time.monotonic()
    #     while not self.PaRealtimeIsReady():
    #         logging.warning("PA is not ready yet")
    #         logging.warning(self.checkedGet(self.paws + "/system/internal"))
    #         if time.monotonic() - t0 > 100:
    #             raise RealtimeException("Waited too long for PaRealtimeIsReady")
    #         sleep(2)

    #     logging.info("AcquisitionPost is ready to post to " + self.paws)
    #     logging.info(self.checkedGet(self.paws + '/system/status'))
    #     logging.info(self.checkedGet(self.paws + "/system/internal"))

    #     logging.info("AcquisitionPost is posting")
    #     self.checkedPost(self.paws + '/acquisitions', payload, 20)

    #     # Check that all new acquisitions map to id.
    #     r = self.checkedGet(self.paws + '/acquisitions')
    #     logging.debug('RT.AcquisitionPost: %s', r)
    #     for acq in r:
    #         Test.assertIn(acq['acqId'], self.ids)

    # def AcquisitionUpdate(self, run, runIter):
    #     global args
    #     uuid = run.ids[runIter]
    #     # we assume here that run number and iteration are [0,99], because we create
    #     # a fake movie context that was based on some random movie context name
    #     # that was created a long time ago on Alpha4 in 2017 :)
    #     movieContext = 'm54004_1702{:02d}_00000{}'.format(run.runNumber, runIter)

    #     cellId = 'cell' + str(runIter)

    #     self.exposure = 1.0/args.framerate  # might want to truncate this to 6 decimal places

    #     # Determine the chip class.
    #     platform = self.GetPlatform()

    #     hqrfMethod = "M1"

    #     # this payload should look like the JSON directly created by ICS
    #     payload = {
    #         'timestamp': '123',
    #         'updates': {
    #             'movieContext' : movieContext,
    #             'cellId': cellId,
    #             'expectedFrameRate': args.framerate,
    #             'exposure': self.exposure
    #         }
    #     }

    #     if args.roi != "" and args.roi is not None:
    #         payload['updates']['roiMetaData'] = {
    #         }
    #         roi = json.loads(args.roi)
    #         payload['updates']['roiMetaData']['sequencingPixelROI'] = roi
    #         payload['updates']['roiMetaData']['traceFilePixelROI'] = []  # nothing
    #     else:

    #         if "Sequel2" in platform:
    #             payload['updates']['roiMetaData'] = {
    #                 'sequencingPixelROI': [[0, 0, 2756, 2912]],
    #                 'sensorPixelROI': [0, 0, 2756, 2912],
    #                 'remapVroiOption': False
    #             }
    #             trace_roi = traceSpider256
    #             if args.verifymovie:
    #                 trace_roi = [[0, 0, 2756, 2912]]
    #             else:
    #                 if args.notrace or args.traceroi == "none":
    #                     trace_roi = [[0, 0, 0, 0]]
    #                 elif args.traceroi == "16k":
    #                     trace_roi = traceSpider16K
    #                 elif args.traceroi == "256" or args.traceroi == "default":
    #                     trace_roi = traceSpider256
    #                 else :
    #                     # this can be either a JSON object or a filename
    #                     trace_roi = args.traceroi

    #             payload['updates']['roiMetaData']['traceFilePixelROI'] = trace_roi

    #             logging.info('RT.AcquisitionUpdate: will install Spider ROIs')
    #             payload['updates']['chipLayoutName'] = "Spider_1p0_NTO"
    #         else:
    #             raise Exception('platform not supported ' + platform)


    #     # Use pa-acq-cli to create a setup JSON object (but not send it to pa-acq)
    #     # fake "--nocalibration" for now, then fill it in later.

    #     (tmpfd, tmpFilename) = tempfile.mkstemp()
    #     command = ['pa-acq-cli', 'setup',
    #                '--getjson',
    #                "--jsonfile", tmpFilename,
    #                '--nocalibration',
    #                '--tracemetadatafile',
    #                args.datafile,
    #                ]
    #     logging.info('RT.AcquisitionUpdate: setup command: %s' , ' '.join(command))
    #     traceMetadata = check_output(command).decode(sys.stdout.encoding)
    #     logging.info('RT.AcquisitionUpdate: Getting trace metadata with\n%s', traceMetadata)
    #     jsonText = slurp(tmpFilename)
    #     logging.debug("slurped json\n%s", jsonText)
    #     j = json.loads(jsonText)
    #     logging.info("successfully loaded tracemetadata JSON")
    #     os.remove(tmpFilename)

    #     #        j["darkframecalfile"] = '/data/pa/cal/' + cellId + '_dark.h5'

    #     logging.info('RT.AcquisitionUpdate: Analogs Size = %s', len(j['analogs']))

    #     # Import these specific subfields which are derived from the .trc.h5 metadata
    #     try:
    #         if len(j['analogs']) == 4:
    #             payload['updates']['analogs'] = j['analogs']
    #     except KeyError:
    #         pass

    #     if args.usefakespectrumvalues:
    #         payload['updates']['analogs'][0]['spectrumValues'] = [0.5, 0.5]
    #         payload['updates']['analogs'][1]['spectrumValues'] = [0.5, 0.5]
    #         payload['updates']['analogs'][2]['spectrumValues'] = [0.2, 0.8]
    #         payload['updates']['analogs'][3]['spectrumValues'] = [0.2, 0.8]

    #     try:
    #         payload['updates']['refDwsSnr'] = j['refDwsSnr']
    #     except KeyError:
    #         pass

    #     try:
    #         if len(j['refSpectrum']) > 0:
    #             payload['updates']['refSpectrum'] = j['refSpectrum']
    #     except KeyError:
    #         pass

    #     logging.info('RT.AcquisitionUpdate: payload = %s' % payload)
    #     self.checkedPost(self.paws + '/acquisitions/' + uuid + '/update', payload)


    def GetAcquisitions(self, uuid):
        r = self.checkedGet(self.paws + '/sockets')
        return r

    def GetCurrentFrameIndex(self):
        """ the current frame is the index of the last frame header received via Aurora. The frame data will not
        be available until the current chunk is flushed through the personality queue."""
        r = self.checkedGet(self.wxdaemon + '/sras/0/status/currentFrameIndex.json')
        return r

def test_KestrelRT():  
    import WxDaemonSim
    import PaWsSim
#        from . import ProgressHelper

    td = tempfile.TemporaryDirectory()
    fn = td.name + "/progress.txt"
    progresser = ProgressHelper.ProgressManager(fn)

    wxdaemon = WxDaemonSim.WxDaemonSim()
    wxdaemon.Run()
    paws = PaWsSim.PaWsSim()
    paws.Run()

    rt = RT(progresser)
    # for testing, the port numbers are arbitrary assigned by the system, so we don't use 23602 or 23632
    rt.wxdaemon = wxdaemon.GetUrl()
    rt.paws = paws.GetUrl()
    logging.info("wx daemon running on %s",rt.wxdaemon)
    rt.VerifyPipelineIsUp()
    assert rt.GetPlatform() == "Kestrel"

    rt.Init()
    acq = Acquisition.Acquisition("1","m1234","/tmp")
    rt.Reset(acq.Socket())
    rt.WaitForStates(acq.Socket(),"basecaller","READY")
    rt.WaitForArmed(acq.Socket(),"basecaller",False)
    rt.StartBasecaller(acq.Socket(), acq, 100)
    rt.WaitForState(acq.Socket(),"basecaller","RUNNING")
    rt.WaitForArmed(acq.Socket(),"basecaller",True)
    rt.WaitForState(acq.Socket(),"basecaller","COMPLETE")

    wxdaemon.Shutdown()
    paws.Shutdown()
    td.cleanup()
