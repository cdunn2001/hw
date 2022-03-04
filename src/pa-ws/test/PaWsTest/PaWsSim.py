
import threading
import socketserver
import logging
import json
import pytest
import time

import HttpHelper

pawsGetDict = {}
pawsPostDict = {}

pawsGetDict["/status"] = {"platform":"Kestrel"}
pawsGetDict["/status/platform"] = "Kestrel"
pawsGetDict["/sockets/1/basecaller/processStatus/executionStatus"] = "UNKNOWN"
pawsGetDict["/sockets/1/darkcal/processStatus/executionStatus"] = "UNKNOWN"
pawsGetDict["/sockets/1/loadingcal/processStatus/executionStatus"] = "UNKNOWN"
pawsGetDict["/sockets"] = [ "1"] 
pawsGetDict["/sockets/1/basecaller"] = {"processStatus":{"executionStatus": "UNKNOWN"}}
pawsGetDict["/sockets/1/darkcal"] = {"processStatus":{"executionStatus": "UNKNOWN"}}
pawsGetDict["/sockets/1/loadingcal"] = {"processStatus":{"executionStatus": "UNKNOWN"}}


def BasecallerSim(payload):
    global pawsGetDict
    if payload == "":
        raise Exception("POST to basecaller/start had empty payload")
    if payload == "reset":
        logging.info("BasecallerSim %s RESET",payload)
        pawsGetDict["/sockets/1/basecaller/processStatus/executionStatus"] = "UNKNOWN"
    else:
        j = json.loads(payload)
        logging.info("BasecallerSim %s READY",payload)
        pawsGetDict["/sockets/1/basecaller/processStatus/executionStatus"] = "READY"
        time.sleep(1)
        logging.info("BasecallerSim %s RUNNING",payload)
        pawsGetDict["/sockets/1/basecaller/processStatus/executionStatus"] = "RUNNING"
        time.sleep(1)
        logging.info("BasecallerSim %s COMPLETE",payload)
        pawsGetDict["/sockets/1/basecaller/processStatus/executionStatus"] = "COMPLETE"

def DarkcalSim(payload):
    global pawsGetDict
    if payload == "":
        raise Exception("POST to darkcal/start had empty payload")
    if payload == "reset":
        logging.info("DarkcalSim %s RESET",payload)
        pawsGetDict["/sockets/1/darkcal/processStatus/executionStatus"] = "UNKNOWN"
    else:
        j = json.loads(payload)
        logging.info("DarkcalSim %s READY",payload)
        pawsGetDict["/sockets/1/darkcal/processStatus/executionStatus"] = "READY"
        time.sleep(1)
        logging.info("DarkcalSim %s RUNNING",payload)
        pawsGetDict["/sockets/1/darkcal/processStatus/executionStatus"] = "RUNNING"
        time.sleep(1)
        logging.info("DarkcalSim %s COMPLETE",payload)
        pawsGetDict["/sockets/1/darkcal/processStatus/executionStatus"] = "COMPLETE"

def LoadingcalSim(payload):
    global pawsGetDict
    if payload == "":
        raise Exception("POST to loadingcal/start had empty payload")
    if payload == "reset":
        logging.info("LoadingcalSim %s RESET",payload)
        pawsGetDict["/sockets/1/loadingcal/processStatus/executionStatus"] = "UNKNOWN"
    else:
        j = json.loads(payload)
        logging.info("LoadingcalSim %s READY",payload)
        pawsGetDict["/sockets/1/loadingcal/processStatus/executionStatus"] = "READY"
        time.sleep(1)
        logging.info("LoadingcalSim %s RUNNING",payload)
        pawsGetDict["/sockets/1/loadingcal/processStatus/executionStatus"] = "RUNNING"
        time.sleep(1)
        logging.info("LoadingcalSim %s COMPLETE",payload)
        pawsGetDict["/sockets/1/loadingcal/processStatus/executionStatus"] = "COMPLETE"

def StartBasecaller(payload):
    global pawsGetDict
    pawsGetDict["/sockets/1/basecaller/processStatus/executionStatus"] = "READY"
    t = threading.Thread(target=BasecallerSim,name="BasecallerSim",args=[payload])
    t.start()

def StartDarkcal(payload):
    global pawsGetDict
    pawsGetDict["/sockets/1/darkcal/processStatus/executionStatus"] = "READY"
    t = threading.Thread(target=DarkcalSim,name="DarkcalSim",args=[payload])
    t.start()

def StartLoadingcal(payload):
    global pawsGetDict
    pawsGetDict["/sockets/1/loadingcal/processStatus/executionStatus"] = "READY"
    t = threading.Thread(target=LoadingcalSim,name="LoadingcalSim",args=[payload])
    t.start()


pawsPostDict["/sockets/1/basecaller/reset"] = lambda payload : StartBasecaller("reset")
pawsPostDict["/sockets/1/basecaller/start"] = lambda payload : StartBasecaller(payload)
pawsPostDict["/sockets/1/darkcal/reset"] = lambda payload : StartDarkcal("reset")
pawsPostDict["/sockets/1/darkcal/start"] = lambda payload : StartDarkcal(payload)
pawsPostDict["/sockets/1/loadingcal/reset"] = lambda payload : StartLoadingcal("reset")
pawsPostDict["/sockets/1/loadingcal/start"] = lambda payload : StartLoadingcal(payload)

class PaWsHandler(socketserver.BaseRequestHandler):

    def handle(self):
        global pawsDict
        try:
            req = ''
#            while True:
            r = self.request.recv(1024)
            req +=  str(r, 'ascii')
#               if not r or Tru: 
#                   break
            logging.debug("PaWsHandler req:%s" % req)
            header,payload, = req.split("\r\n\r\n", maxsplit=2)
            req0,url,theRest = header.split(None, maxsplit=2)
            if req0 == "GET":
                if url in pawsGetDict:
                    code="200 OK"
                    response = pawsGetDict[url]
                else:
                    code="404 NOT FOUND"
                    response="URL " + url + " not found"    
            elif req0 == "POST":
                if payload == "":
                    payload = str(self.request.recv(1024), "ascii")
                if url in pawsPostDict:
                    code="200 OK"
                    f = pawsPostDict[url]
                    f(payload)
                    response = {"message":"ok"}
                else:
                    code="404 NOT FOUND"
                    response="URL " + url + " not found"    
                code="200 OK"
            else:
                code="404 NOT FOUND"
                response = {"message": "404 Not Found"}
        except Exception as ex:
            code="500 bad"
            response = {"message":format(ex)}
        response = bytes("HTTP/1.1 {}\n\n{}".format(code, json.dumps(response)), 'ascii')
        logging.debug("handler:response:%s" % response)
        self.request.sendall(response)

class PaWsSim:
    def Run(self):
        self.server = socketserver.TCPServer(("", 0), PaWsHandler)
        self.ip, self.port = self.server.server_address
        logging.info("serving at %s" % self.GetUrl())
        self.server_thread = threading.Thread(target=self.server.serve_forever,name="PaWs")
        self.server_thread.daemon = True
        self.server_thread.start()
        if not self.server_thread.is_alive():
            raise Exception("server is not alive")
            
        logging.info("Server loop running in thread:%s", self.server_thread.name)

    def GetUrl(self):
        return "http://%s:%d" % ( self.ip, self.port)

    def Shutdown(self):
        self.server.shutdown()
        self.server.server_close()
        self.server_thread.join()

def test_one():
    paws = PaWsSim()
    paws.Run()
    url = paws.GetUrl()
    client = HttpHelper.SafeHttpClient()
    assert "platform" in str(client.checkedGet(url +"/status"))
    assert "{" in str(client.checkedPost(url + "/sockets/1/basecaller/reset",payload={},timeout=60))
    assert "UNKNOWN" in str(client.checkedGet(url +"/sockets/1/basecaller/processStatus/executionStatus"))
    assert "{" in str(client.checkedPost(url +"/sockets/1/basecaller/start",payload={"mid":"m1234"},timeout=60))
    assert "READY" in str(client.checkedGet(url +"/sockets/1/basecaller/processStatus/executionStatus"))
    time.sleep(1.1)
    assert "RUNNING" in str(client.checkedGet(url +"/sockets/1/basecaller/processStatus/executionStatus"))
    time.sleep(1.1)
    assert "COMPLETE" in str(client.checkedGet(url +"/sockets/1/basecaller/processStatus/executionStatus"))

    paws.Shutdown()

