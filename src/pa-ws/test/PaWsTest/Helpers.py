import unittest

class RealtimeException(Exception):
    def __init__(self, message):
        super(RealtimeException, self).__init__(message)

def TerminateNamedProcess(processName):
    import psutil  # type: ignore
    for proc in psutil.process_iter():
        # check whether the process name matches
        # print "*** matching %s" % proc.name()
        if proc.name() == processName:
            print("**** found it, terminating! process number %d" % proc.pid)
            proc.terminate()
            return

        #            sleep(1)
        #            if psutil.pid_exists(proc.pid):
        #                raise Exception("kill failed for %d" % proc.pid)
    raise Exception("Process %s not found, not killed " % processName)

def slurp(filename):
    with open(filename) as x: f = x.read()
    return f

class TestHelpers(unittest.TestCase):
    def test_one(self):  
        with self.assertRaises(RealtimeException):
            raise RealtimeException("haha")

if __name__ == '__main__':
    unittest.main()
