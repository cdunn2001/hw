
import sys
import threading
import socketserver
import logging
import time
import HttpHelper
import json
import unittest

wsGetDict = {}
wsPostDict = {}
wsGetDict["/status"] = {"platform":"Kestrel"}
wsGetDict["/status/platform"] = "Kestrel"
wsGetDict["/sras/0/status"] = {"state":"bar"}
wsGetDict["/sras/0/status/state"] = "online"
wsGetDict["/status/wolverine/frameLatency"] = 0
wsGetDict["/sras/0/status/transmitter/state"] = "IDLE"
wsGetDict["/sras/0/status/transmitter/frameIndex"] = 0 # TODO. have this increment
wsGetDict["/sras/0/status/transmitter/frameOffset"] = 0 # TODO. have this increment

def TransmitterSim(payload):
    global wsGetDict
    ep = "/sras/0/status/transmitter/state"
    logging.info("TransmitterSim PRELOADING %s",payload)
    tconfig = json.loads(payload)
    period = 1.0 / float(tconfig["rate"])
    wsGetDict[ep] = "PRELOADING"
    wsGetDict["/sras/0/status/transmitter/frameIndex"] = 0
    wsGetDict["/sras/0/status/transmitter/numFrames"] = tconfig["frames"]
    wsGetDict["/sras/0/status/transmitter/frameOffset"] = 0
    time.sleep(1)
    wsGetDict[ep] = "TRANSMITTING"
    logging.info("TransmitterSim TRANSMITTING")
    for i in range(0,wsGetDict["/sras/0/status/transmitter/numFrames"]):
        wsGetDict["/sras/0/status/transmitter/frameIndex"] += 1
        time.sleep(period)

    logging.info("TransmitterSim IDLE")
    wsGetDict[ep] = "IDLE"

def ChangeTransmitterState(payload):
    if True:
        t = threading.Thread(target=TransmitterSim,name="TransmitterSim",args=[payload])
        t.start()

wsPostDict["/sras/0/transmit"] = lambda payload : ChangeTransmitterState(payload)

    

class WsHandler(socketserver.BaseRequestHandler):

    def handle(self):
        global wsDict
        try:
            req = str(self.request.recv(1024), 'ascii')
            logging.debug("wsHandler req:%s" % req)
            header,payload, = req.split("\r\n\r\n", maxsplit=2)
            req0,url,theRest = header.split(None, maxsplit=2)

            if req0 == "GET":
                if url in wsGetDict:
                    code="200 OK"
                    response = wsGetDict[url]
                else:
                    code="404 NOT FOUND"
                    response="URL " + url + " not found"    
            elif req0 == "POST":
                if payload == "":
                    payload = str(self.request.recv(1024), "ascii")
                if url in wsPostDict:
                    code="200 OK"
                    f = wsPostDict[url]
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


class WxDaemonSim:
    def Run(self):
        self.server = socketserver.TCPServer(("", 0), WsHandler)
        self.ip, self.port = self.server.server_address
        logging.info("serving at %s" % self.GetUrl())
        self.server_thread = threading.Thread(target=self.server.serve_forever,name="wxDaemonSim")
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

class TestWxDaemonSim(unittest.TestCase):
    def test_one(self):
        wx = WxDaemonSim()
        wx.Run()
        url = wx.GetUrl()
        client = HttpHelper.SafeHttpClient()
        self.assertIn("platform",str(client.checkedGet(url +"/status")))
        wx.Shutdown()

    def test_transmitter(self):
        wx = WxDaemonSim()
        wx.Run()
        url = wx.GetUrl()
        client = HttpHelper.SafeHttpClient()
        self.assertEqual("IDLE",str(client.checkedGet(url + "/sras/0/status/transmitter/state")))
        self.assertEqual("0",str(client.checkedGet(url + "/sras/0/status/transmitter/frameIndex")))

        self.assertIn("{",str(client.checkedPost(url +"/sras/0/transmit",{"rate":100.0,"frames":1000})))  # TODO, add tighter assert
        t0=time.monotonic()
        frameIndexLast = 0
        while time.monotonic() - t0 < 11.0:
            time.sleep(1)
            frameIndex = int(client.checkedGet(url + "/sras/0/status/transmitter/frameIndex"))
            logging.info("frameIndex:%d" % frameIndex)
            self.assertGreater(frameIndex, frameIndexLast)
            frameIndexLast = frameIndex
            if frameIndex >= 1000:
                break

        wx.Shutdown()


if __name__ == '__main__':
    # logging.disable(logging.DEBUG)
    logging.basicConfig(level=logging.INFO)
    unittest.main()

