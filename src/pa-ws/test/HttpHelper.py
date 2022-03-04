import logging
import sys
from time import sleep
import requests
import unittest

class HttpException(Exception):
    pass

class SafeHttpClient(object):
    def __init__(self):
        pass

    def checkedGet(self, url):
        # logging.info('GET - %s', url)
        try:
            response = requests.get(url)
            logging.debug("Raw response:%s", response.text)
            r = response.json()
            if response.status_code / 100 != 2:
                raise HttpException('status_code:' + str(response.status_code) + str(r))
            return r
        except Exception as e:
            logging.error("checkGet failed with URL "+ url)
#            raise Exception("checkGet failed with URL "+ url) from e
            raise

    def checkedPost(self, url, payload, timeout=5):
        logging.info('POST - %s, %s', url, str(payload)[0:100] + " ...")
        logging.debug('POST - %s, %s', url, payload)
        # debugging REST calls, not needed now:
        if type(payload) is str:
            logging.info("payload is string")
            response = requests.post(url, json=payload, timeout=timeout)
        elif type(payload) is dict:
            logging.info("payload is dict")
            response = requests.post(url, json=payload, timeout=timeout)
#            response = requests.post(url, data=payload, timeout=timeout)
        else:
            print(type(payload))
            raise Exception("payload is not string")
        r = response.json()
        if response.status_code / 100 != 2 :
            raise HttpException('status_code:' + str(response.status_code) + str(r))
        return r

class TestHttpHelper(unittest.TestCase):
    def test_one(self):
        import threading
        import http.server
        import socketserver
        import socket

        class PaWsHandlers(socketserver.BaseRequestHandler):

            def handle(self):
                req = str(self.request.recv(1024), 'ascii')
                logging.debug("handler req:%s" % req)
                if req[0:15] == "GET /dummy.json":
                    code="200 OK"
                    response = "{ \"foo\": 456 }"
                elif req[0:15] == "POST /blackhole":
                    code="200 OK"
                    response = "{ \"foo\": 123 }"
                else:
                    code="404 NOT FOUND"
                    response = "{}"    
                logging.debug("handler:response:%s" % response)
                response = bytes("HTTP/1.1 {}\n\n{}".format(code,response), 'ascii')
                self.request.sendall(response)

        httpd = socketserver.TCPServer(("", 0), PaWsHandlers)
        with httpd:
            ip, port = httpd.server_address
            # print("serving at ip %s: port%s" %(ip, port))
            server_thread = threading.Thread(target=httpd.serve_forever)
            server_thread.daemon = True
            server_thread.start()
            self.assertTrue(server_thread.is_alive())

            logging.info("Server loop running in thread:", server_thread.name)

            hh = SafeHttpClient()
            hh.checkedGet("http://localhost:%d/dummy.json" % port)
            logging.disable(logging.ERROR)
            with self.assertRaises(HttpException):
                hh.checkedGet("http://localhost:%d/failure" % port)

            hh.checkedPost("http://localhost:%d/blackhole" % port, {"foo":123})
            with self.assertRaises(HttpException):
                hh.checkedPost("http://localhost:%d/failure" % port, "{}")

            logging.info("shutting down")
            httpd.shutdown()
            logging.info("shut down")
            server_thread.join()

if __name__ == '__main__':
    logging.disable(logging.ERROR)
    # logging.basicConfig(level=logging.INFO)

    unittest.main()
