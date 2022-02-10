#!/bin/bash
# The first 14 lines are a little hack to start this script as bash, source the virtual env for python and continue in python.
# Putting double quotes around words are ignored in python, but evaluated in bash.

"echo" "switching from default python to VE python ..."
"export" "SCRIPT=`realpath $0`"
"export" "SCRIPTPATH=`dirname $SCRIPT`"
"echo" "running from $SCRIPTPATH"
"which" "python"
"python" "--version"
"." "$SCRIPTPATH/install_python3_ve.sh"
"which" "python"
"python" "--version"
"exec" "python3" "$SCRIPT" "$@"

# the rest is python
import h5py
import argparse

verbose = False

def diagonal(frame,row,col):
    """ The test pattern of the sensor"""
    return (row + (col % 384)) % 256

def alpha(frame,row,col):
    """ The traditional pattern generation. The wolverine FPGA generates this pattern by default"""
    if col >= 3072:
        col -= 3072
    return (frame * 100 + row * 10 + col) % 256

class MyValidator:
    """ Experimental class to aid in validating patterns in a trc.h5 file"""

    def __init__(self,file) :
        self.f_ = h5py.File(file, 'r')
        print("%s" % self.f_.keys())

        self.traceData = self.f_['TraceData']
        print("%s" % self.traceData.keys())
        self.traces = self.traceData['Traces']
        print("%s" % self.traces)
        self.holexy = self.traceData['HoleXY']
        print("%s" % self.holexy)

    def validate_roi(self, roispec):
        """ Validates the contents of the trc file with the expected ROI"""
        numFound = 0
        numLost = 0
        minRow = 1000000
        maxRow = 0
        minCol = 1000000
        maxCol = 0
        for zmw in range(0,self.holexy.shape[0]):
            row = self.holexy[zmw,0]
            col = self.holexy[zmw,1]
            minRow = min([row,minRow])
            maxRow = max([row,maxRow])
            minCol = min([col,minCol])
            maxCol = max([col,maxCol])
            #    print("[%d @ (%d,%d)]"  % (zmw,row,col))
            found = False
            for rect in roispec:
                if row >= rect[0] and row < rect[0] + rect[2] and col >= rect[1] and col < rect[1] + rect[3]:
                    found = True
                    break
            if found:
                numFound += 1
            else :
                numLost += 1
                global verbose
                if verbose:
                    for rect in roispec:
                        print("zmw:%d %d,%d not found in %d:%d, %d:%d" % (zmw,row,col,rect[0],rect[0] + rect[2], rect[1], rect[1] + rect[3]))
        print("bounding box:[%d,%d,%d,%d]" % ( minRow, minCol, maxRow - minRow+1, maxCol - minCol+1))                
        return (numFound, numLost, )




    def validate(self,roispec,args):
        """validates the whole file"""

        numValid = 0
        numWrong = 0

        if args.validate_roi:
            res = self.validate_roi(roispec)
            if res[1] != 0:
                print("ROI is invalid, %d in ROI, %d outside" % res)
            numValid += res[0]
            numWrong += res[1]

        if args.validate_traces:
            maxFrames = args.max_frames
            if maxFrames > self.traces.shape[2]:
                maxFrames = self.traces.shape[2]
            print("Validating first %d frames" % (maxFrames))
            if args.pattern == "alpha":
                func = lambda f, r, c: alpha(f,r,c)
            elif args.pattern == "diagonal1":
                func = lambda f, r, c: diagonal(f,r,c)
            elif args.pattern == "diagonal2":
                func = lambda f, r, c: diagonal(f,r+4,c)

            else :
                raise Exception("don't recognize pattern " + args.pattern)
            for zmw in range(0,self.holexy.shape[0]):
                traceValid = self.validate_trace(zmw,maxFrames,func)
                if traceValid:
                    numValid += 1
                else :
                    numWrong += 1
        print("valid:%d wrong:%d" % (numValid, numWrong))

    def validate_trace(self,zmw,maxFrames, func):
        """validates a single trace"""
        global verbose
        trace = self.traces[zmw,0,:]
        row = self.holexy[zmw,0]
        col = self.holexy[zmw,1]
        traceValid = True
        # print(zmw)
        # print(trace)
        for frame in range(0,maxFrames):
            expected = func(frame,row,col)
            valid = expected == trace[frame]
            if not valid or verbose:
                traceValid = False
                print("[%d/%d @ (%d,%d)] %d != %d %s" % (zmw,frame,row,col,expected,trace[frame],valid))
        return traceValid

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='trace file validation')

    #
    # Test setup options
    #
    group = parser.add_argument_group('Test Setup')
    group = parser.add_argument_group('Data Verification')
    group.add_argument('--roi',
                       help="ROI specification in JSON format, e.g. [[0,0,64,256]]",
                       default='[[0,0,4096,6144]]')
    group.add_argument('--file',
                       help="trc.h5 filename",
                       default='')
    group.add_argument('--validate_roi',
                       action=argparse.BooleanOptionalAction,
                       default=True,
                       help="If True, the HolesXY are checked to match the ROI"
                       )
    group.add_argument('--validate_traces',
                       action=argparse.BooleanOptionalAction,
                       default=True,
                       help="If True, the traces are compared against the Alpha test pattern")
    group.add_argument('--max_frames',
                       default=100,
                       type=int,
                       help="Maximum number of frames to verify")
    group.add_argument('--verbose',
                       action=argparse.BooleanOptionalAction,
                       default=False,
                       help="If True, lots of details are printed")
    group.add_argument('--pattern',
                       default="alpha",
                       help="Name of the test pattern: alpha or diagonal1 or diagonal2")

    args = parser.parse_args()

    verbose=args.verbose
    f = MyValidator(args.file)
    roispec=eval(args.roi)
    f.validate(roispec,args)
