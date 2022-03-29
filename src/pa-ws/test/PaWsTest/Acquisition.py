from xml.dom import minidom
import pytest
import pathlib

class Acquisition:
    def __init__(self, socket, mid, storageRoot):
        self.socket = socket
        self.mid = mid
        self.cellId = "xxx"
        self.storageRoot = storageRoot

    def __str__(self):
        return "Acq(%s,%s,%s)" % (self.socket, self.mid, self.storageRoot)

    def Socket(self):
        return self.socket

    def GenerateBasecallerJsonPayload(self):
        return {
            "mid":self.mid
            }

    def GenerateDarkcalJsonPayload(self):
# TODO        storagePrefix = "http://localhost:23632/storages/" + self.mid
        storagePrefix = self.storageRoot +"/" + self.mid
        pathlib.Path(storagePrefix).mkdir(parents=True, exist_ok=True)

        return {
            "mid": self.mid,
            "movieMaxFrames": 512,
            "movieMaxSeconds": 6,
            "movieNumber": 1,
            "calibFileUrl": storagePrefix + "/darkcal.h5",
            "logUrl": storagePrefix + "/darkcal.log",
            "logLevel": "INFO"
        }

    def GenerateLoadingcalJsonPayload(self):
# TODO        storagePrefix = "http://localhost:23632/storages/" + self.mid
        storagePrefix = self.storageRoot + "/" + self.mid
        pathlib.Path(storagePrefix).mkdir(parents=True, exist_ok=True)
        return {
            "mid": self.mid,
            "movieMaxFrames": 512,
            "movieMaxSeconds": 6,
            "movieNumber": 1,
            "calibFileUrl": storagePrefix + "/loadingcal.h5",
            "darkFrameFileUrl": storagePrefix +"/darkcal.h5",
            "logUrl": storagePrefix + "/loadingcal.log",
            "logLevel": "INFO"
        }

def test_Acquisition():  
    acq = Acquisition("1","m1234","/data/nrta/0")
    assert acq.GenerateBasecallerJsonPayload()["mid"] == "m1234"
    assert acq.GenerateDarkcalJsonPayload()["mid"] == "m1234"
    assert "m1234" in acq.GenerateDarkcalJsonPayload()["calibFileUrl"]
    assert "/data/nrta/0" in acq.GenerateDarkcalJsonPayload()["calibFileUrl"]
    assert acq.GenerateLoadingcalJsonPayload()["mid"] == "m1234"
