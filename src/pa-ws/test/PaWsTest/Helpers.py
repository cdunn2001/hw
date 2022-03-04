import pytest

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

def test_Helpers():  
    with pytest.raises(RealtimeException):
        raise RealtimeException("haha")
