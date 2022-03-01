
import sys
import threading
import socketserver
import logging
import HttpHelper
import json

pawsGetDict = {}
pawsPostDict = {}
pawsGetDict["/status"] = {"platform":"Kestrel"}
pawsGetDict["/sras/0/status"] = {"state":"bar"}
pawsGetDict["/sras/0/status/state"] = "online"
pawsGetDict["/status/wolverine/frameLatency"] = 0
pawsGetDict["/sras/0/status/transmitter/state"] = "idle"

#            "frameIndex": uint64 of the frame that was last transmitted,
#            "frameOffset": uint64 offset into the simulated file or pattern,
#            "numFrames": total number of frames in the simulated file or pattern,
#            "state" : state of the movie transmitter
#            {
#                "idle",
#                "preloading",
#                "transmitting"
#            }

pawsPostDict["/sras/0/transmit"] = lambda : logging.info("TRANSMITTING!")

class PaWsHandler(socketserver.BaseRequestHandler):

    def handle(self):
        global pawsDict
        try:
            req = str(self.request.recv(1024), 'ascii')
            logging.debug("PaWsHandler req:%s" % req)
            req0,url,other = req.split(None, maxsplit=2)
            if req0 == "GET":
                if url in pawsGetDict:
                    code="200 OK"
                    response = pawsGetDict[url]
                else:
                    code="404 NOT FOUND"
                    response="URL " + url + " not found"    
            elif req0 == "POST":
                if url in pawsPostDict:
                    code="200 OK"
                    f = pawsPostDict[url]
                    f()
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
        self.server = socketserver.TCPServer(("", 0), PaWsHandler)
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
        self.server_thread.join()

if __name__ == '__main__':
    logging.disable(logging.DEBUG)
    #logging.basicConfig(level=logging.DEBUG)

    try:
        wx = WxDaemonSim()
        wx.Run()
        url = wx.GetUrl()
        client = HttpHelper.SafeHttpClient()
        logging.info (str(client.checkedGet(url +"/status")))
        logging.info (str(client.checkedPost(url +"/sras/0/transmit",{"go":"now"})))
        wx.Shutdown()
        print("WxDaemonSim:PASS")
        sys.exit(0)
    except Exception as ex:
        logging.error("exception " + format(ex))
        print("WxDaemonSim:FAIL")
        sys.exit(1)