
import threading
import socketserver
import logging
import json
import os
import pytest
import time

import HttpHelper

pawsGetDict = {}
pawsPostDict = {}

pawsGetDict["/status"] = {"platform":"Kestrel"}
pawsGetDict["/status/platform"] = "Kestrel"
pawsGetDict["/sockets"] = [ "1"] 
pawsGetDict["/sockets/1/basecaller"] = {"processStatus":{}}
pawsGetDict["/sockets/1/darkcal"] = {"processStatus":{}}
pawsGetDict["/sockets/1/loadingcal"] = {"processStatus":{}}


def DoSim(app,root,payload):
    global pawsGetDict
    if payload == "":
        raise Exception("POST to %s/start had empty payload" % root)
    if payload == "reset":
        logging.info("%s %s RESET",app,payload)
        pawsGetDict[root]["processStatus"]["executionStatus"] = "READY"
        pawsGetDict[root]["processStatus"]["completionStatus"] = "UNKNOWN"
        pawsGetDict[root]["processStatus"]["armed"] = False
    else:
        j = json.loads(payload)
        logging.info("%s %s RUNNING",app,payload)
        pawsGetDict[root]["processStatus"]["executionStatus"] = "RUNNING"
        pawsGetDict[root]["processStatus"]["completionStatus"] = "UNKNOWN"
        pawsGetDict[root]["processStatus"]["armed"] = False
        time.sleep(1)
        pawsGetDict[root]["processStatus"]["armed"] = True
        time.sleep(1)
        logging.info("%s %s COMPLETE",app,payload)
        pawsGetDict[root]["processStatus"]["executionStatus"] = "COMPLETE"
        pawsGetDict[root]["processStatus"]["completionStatus"] = "SUCCESS"
        pawsGetDict[root]["processStatus"]["armed"] = False

def BasecallerSim(payload):
    DoSim("basecaller","/sockets/1/basecaller",payload)

def DarkcalSim(payload):
    DoSim("darkcal","/sockets/1/darkcal",payload)

def LoadingcalSim(payload):
    DoSim("loadingcal","/sockets/1/loadingcal",payload)

def StartBasecaller(payload):
    global pawsGetDict
    t = threading.Thread(target=BasecallerSim,name="BasecallerSim",args=[payload])
    t.start()

def StartDarkcal(payload):
    global pawsGetDict
    t = threading.Thread(target=DarkcalSim,name="DarkcalSim",args=[payload])
    t.start()

def StartLoadingcal(payload):
    global pawsGetDict
    t = threading.Thread(target=LoadingcalSim,name="LoadingcalSim",args=[payload])
    t.start()


def ResetAll(payload):
    print("ResetAll(%s)" % payload)
    StartBasecaller("reset"),
    StartDarkcal("reset"),
    StartLoadingcal("reset") 

pawsPostDict["/sockets/1/reset"] = ResetAll
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
                # Try to match the longest piece of the URL against the pawsGetDict
                # keys.  After a match is made, then drill down the remaining part of the 
                # URL into the response.
                paths = [ ]
                while url != "":
                    if url in pawsGetDict:
                        code="200 OK"
                        response = pawsGetDict[url]
                        for p in paths:
                            response = response[p]
                        break
                    p = os.path.basename(url)
                    url = os.path.dirname(url)
                    paths.insert(0,p)
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
        ResetAll("reset")
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
    client.checkedPost(url + "/sockets/1/reset",payload={},timeout=60)
    assert "{" in str(client.checkedPost(url + "/sockets/1/basecaller/reset",payload={},timeout=60))
    time.sleep(0.1)
    assert "{'processStatus': {" in str(client.checkedGet(url +"/sockets/1/basecaller"))
    assert "{'executionStatus': 'READY'" in str(client.checkedGet(url +"/sockets/1/basecaller/processStatus"))
    assert "READY" in str(client.checkedGet(url +"/sockets/1/basecaller/processStatus/executionStatus"))
    assert "{" in str(client.checkedPost(url +"/sockets/1/basecaller/start",payload={"mid":"m1234"},timeout=60))
    assert "RUNNING" in str(client.checkedGet(url +"/sockets/1/basecaller/processStatus/executionStatus"))
    time.sleep(1.1)
    assert client.checkedGet(url +"/sockets/1/basecaller/processStatus/armed")
    time.sleep(1.1)
    assert "COMPLETE" in str(client.checkedGet(url +"/sockets/1/basecaller/processStatus/executionStatus"))

    paws.Shutdown()
