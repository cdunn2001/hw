from xml.dom import minidom
import pytest

class Acquisition:
    def __init__(self, mid):
        self.mid = mid
        self.cellId = "xxx"

    def __str__(self):
        return "Acq(%s)" % self.mid

    def GenerateBasecallerJsonPayload(self):
        return {
            "mid":self.mid
            }

    def GenerateDarkcalJsonPayload(self):
        storagePrefix = "http://localhost:23632/storages/" + self.mid
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
        storagePrefix = "http://localhost:23632/storages/" + self.mid
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
    acq = Acquisition("m1234")
    assert acq.GenerateBasecallerJsonPayload()["mid"] == "m1234"
    assert acq.GenerateDarkcalJsonPayload()["mid"] == "m1234"
    assert "m1234" in acq.GenerateDarkcalJsonPayload()["calibFileUrl"]
    assert acq.GenerateLoadingcalJsonPayload()["mid"] == "m1234"
