from xml.dom import minidom
import unittest

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

class TestAcquisition(unittest.TestCase):
    def test_one(self):  
        pass

if __name__ == '__main__':
    unittest.main()
