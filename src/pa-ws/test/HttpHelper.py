import logging
from time import sleep
import requests

class HttpException(Exception):
    pass

class HttpHelper(object):
    def __init__(self):
        pass

    def checkedGet(self, url):
        # logging.info('GET - %s', url)
        try:
            response = requests.get(url)
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
        #        if payload is str:
        #            logging.info("payload is string")
        #        else:
        #            print(type(payload))
        #            logging.info("payload is not string")
        response = requests.post(url, json=payload, timeout=timeout)
        r = response.json()
        if response.status_code / 100 != 2 :
            raise HttpException('status_code:' + str(response.status_code) + str(r))
        return r

if __name__ == '__main__':
    import threading
    import http.server
    import socketserver
    import socket

    logging.disable(logging.ERROR)
    # logging.basicConfig(level=logging.INFO)

    try:
 
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
            if not server_thread.is_alive():
                print("server is not alive")
                exit(1)
            logging.info("Server loop running in thread:", server_thread.name)

            hh = HttpHelper()
            hh.checkedGet("http://localhost:%d/dummy.json" % port)
            try :
                hh.checkedGet("http://localhost:%d/failure" % port)
                raise Exception("Excepted 404 failure did not fail")
            except HttpException:
                pass # expected exception
            except :
                raise

            hh.checkedPost("http://localhost:%d/blackhole" % port, {"foo":123})
            try :
                hh.checkedPost("http://localhost:%d/failure" % port, "{}")
                raise Exception("Excepted 404 failure did not fail")
            except HttpException:
                pass # expected exception
            except :
                raise



            logging.info("shutting down")
            httpd.shutdown()
            logging.info("shut down")
            server_thread.join()

            print("PASS")
    except Exception as ex:
        logging.error("failed with exception %s" % format(ex))
        print("FAIL")


