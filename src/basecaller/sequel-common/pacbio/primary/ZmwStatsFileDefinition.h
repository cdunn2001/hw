// Copyright (c) 2014-2016, Pacific Biosciences of California, Inc.
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted (subject to the limitations in the
// disclaimer below) provided that the following conditions are met:
//
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
//  * Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
//  * Neither the name of Pacific Biosciences nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE
// GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY PACIFIC
// BIOSCIENCES AND ITS CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL PACIFIC BIOSCIENCES OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
// USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
// OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGE.

// Programmer: Mark Lakata
//
// Defines the API for reading and writing the std.h5 file.
//
//
// this file is included by other header files and should not be loaded by user code

DECLARE_SEQUELH5_SUPPORT()

// /ZmwMetrics

DECLARE_ZMWSTAT_START_GROUP(ZMWMetrics)

DECLARE_ZMWSTATDATASET_1D(,NumBases,             uint32,  "bases",  "Number of called bases", NOT_OVER_HQR, nH_)
DECLARE_ZMWSTATDATASET_1D(,ReadLength,           uint32,  "bases",  "Number of  bases in HQ-region", OVER_HQR, nH_)
DECLARE_ZMWSTATDATASET_1D(,Loading,	             uint8,   "",       "Classification based on the estimated number of active sequencing reactions in a ZMW", OVER_HQR, nH_)
DECLARE_ZMWSTAT_ENUM(Loading_t, // see LoadingClass.h in ppa
        Empty=0,
        Single=1,
        Multi=2,
        Indeterminate=3,
        Undefined=255
)
DECLARE_ZMWSTATDATASET_1D(,Productivity,	     uint8,   "",       "Classification based on the presence of high quality sequence information", OVER_HQR, nH_)
DECLARE_ZMWSTAT_ENUM(Productivity_t, /* see ProductivityClass.h in ppa */
        Empty=0,
        Productive=1,
        Other=2,
        Undefined=255
)
DECLARE_ZMWSTATDATASET_1D(,MedianInsertLength,	 uint32,  "bases",  "Median insert length over interior (flanked by adapters) subreads of the ZMW", OVER_HQR, nH_)
DECLARE_ZMWSTATDATASET_1D(,InsertReadLength,	 uint32,  "bases",  "Estimate of the insert read length", OVER_HQR, nH_)
DECLARE_ZMWSTATDATASET_1D(,ReadType,	         uint8,   "",       "Classification based on HQ-region start time and sequencing activity pre and post HQ-region", OVER_HQR, nH_)
DECLARE_ZMWSTAT_ENUM(ReadType_t,
        Empty=0,
        FullHqRead0=1,
        FullHqRead1=2,
        PartialHqRead0=3,
        PartialHqRead1=4,
        PartialHqRead2=5,
        Multiload=6,
        Indeterminate=7,
        Undefined=255
)
DECLARE_ZMWSTATDATASET_1D(,ReadScore,	         float32, "[0,1]",      "ZMW quality score", OVER_HQR, nH_)
DECLARE_ZMWSTATDATASET_1D(,HQRegionStart,	     uint32,  "bases",      "First base of HQ-region", OVER_HQR, nH_)
DECLARE_ZMWSTATDATASET_1D(,HQRegionEnd,	         uint32,  "bases",      "Last base of HQ-region", OVER_HQR, nH_)
DECLARE_ZMWSTATDATASET_1D(,HQRegionStartTime,	 uint32,  "sec",        "Start time of HQ-region", OVER_HQR, nH_)
DECLARE_ZMWSTATDATASET_1D(,HQRegionEndTime,	     uint32,  "sec",        "End time of HQ-region", OVER_HQR, nH_)
DECLARE_ZMWSTATDATASET_1D(,HQRegionScore,	     float32, "[0,1]",      "HQ-region score", OVER_HQR, nH_)
DECLARE_ZMWSTATDATASET_1D(,BaseRate,	         float32, "bases/sec",  "Mean rate of base-incorporation events", OVER_HQR, nH_)
DECLARE_ZMWSTATDATASET_1D(,BaseWidth,	         float32, "sec",        "Mean pulse width of base-incorporation events", OVER_HQR, nH_)
DECLARE_ZMWSTATDATASET_1D(,BaseIpd,	             float32, "sec",        "Robust estimate of the mean inter-pulse distance (IPD) of base-incorporation events", OVER_HQR, nH_)
DECLARE_ZMWSTATDATASET_1D(,LocalBaseRate,	     float32, "bases/sec",  "Robust estimate (excluding pauses) of the mean base-incorporation rate", OVER_HQR, nH_)
DECLARE_ZMWSTATDATASET_1D(,NumPulses,	         uint32,  "pulses",     "Number of pulses", OVER_HQR, nH_)
DECLARE_ZMWSTATDATASET_1D(,PulseRate,	         float32, "pulses/sec", "Mean pulse rate", OVER_HQR, nH_)
DECLARE_ZMWSTATDATASET_1D(,PulseWidth,	         float32, "sec",        "Mean pulse width", OVER_HQR, nH_)
DECLARE_ZMWSTATDATASET_1D(,Pausiness,	         float32, "[0,1]",      "Fraction of pause events over the HQ (sequencing) region", OVER_HQR, nH_)
DECLARE_ZMWSTATDATASET_1D(,IsControl,            uint8,   "",           "Indicates ZMW is control read", OVER_HQR, nH_)
DECLARE_ZMWSTATDATASET_1D(,ControlReadLength,    uint32,  "bases",      "Aligned control read length", OVER_HQR, nH_)
DECLARE_ZMWSTATDATASET_1D(,ControlReadAccuracy,  float32, "[0,1]",      "Aligned control read concordance", OVER_HQR, nH_)

// Analog metrics  4 values, one for each analog per ZMW

DECLARE_ZMWSTATDATASET_2D(,BaseFraction,	     float32, "[0,1]",      "Base fraction by color channel", OVER_HQR, nH_, nA_)
DECLARE_ZMWSTATDATASET_2D(,HQRegionSnrMean,	     float32, "",           "Signal-to-Noise Ratio in the HQ region", OVER_HQR, nH_, nA_)
DECLARE_ZMWSTATDATASET_2D(,SnrMean,	             float32, "",           "Mean Signal-to-Noise Ratio of DWS color channels", NOT_OVER_HQR, nH_, nA_)
DECLARE_ZMWSTATDATASET_2D(,HQPkmid,	             float32, "photo e-",   "Pkmid in HQ-region, 0 if no HQ-region", OVER_HQR, nH_, nA_)

// Categorical metrics, n values per ZMW
DECLARE_ZMWSTATDATASET_2D(,InsertionCounts,	         uint32, "count",      "Number of long insertions of each type (Base,Short,Burst,Pause)", NOT_OVER_HQR, nH_, four_)
DECLARE_ZMWSTATDATASET_2D(,InsertionLengths,	     uint32, "pulses",      "Total length of long insertions of each type (Base,Short,Burst,Pause)", NOT_OVER_HQR, nH_, four_)

// Filter metrics 2 values, one for each filter per ZMW

DECLARE_ZMWSTATDATASET_2D(,HQBaselineLevel,      float32, "photo e-",   "Trace-mean baseline bias estimates, by DWS color channel", OVER_HQR, nH_, nF_)
DECLARE_ZMWSTATDATASET_2D(,HQBaselineStd,        float32, "photo e-",   "Trace baseline noise (standard deviation) estimates, by DWS color channel", OVER_HQR, nH_, nF_)
DECLARE_ZMWSTATDATASET_2D(,BaselineLevel,        float32, "photo e-",   "The trace-mean baseline bias estimates, by DWS color channel", NOT_OVER_HQR, nH_, nF_)
DECLARE_ZMWSTATDATASET_2D(,BaselineStd,          float32, "photo e-",   "Trace baseline noise (standard deviation) estimates, by DWS color channel", NOT_OVER_HQR, nH_, nF_)
DECLARE_ZMWSTATDATASET_2D(,DyeAngle,             float32, "degrees",    "Dye angle estimate produced by DME stage of basecaller", OVER_HQR, nH_, nA_)
DECLARE_ZMWSTATDATASET_2D(,HQChannelMinSnr,      float32, "",           "Mean Signal-to-Noise Ratio of low analog estimated from relative amplitudes ", OVER_HQR, nH_, nF_)


// Diagnostic groups/datasets that are optionally added depending upon if they are requested to be added.
DECLARE_ZMWSTAT_START_GROUP_DIAGNOSTICS(VsMF)
DECLARE_ZMWSTATDATASET_2D_DIAGNOSTICS(VsMF,TraceAutoCorr, float32, "",  "Autocorrelation of tracedata within a block", NOT_OVER_HQR, nH_, nMF_)
DECLARE_ZMWSTATDATASET_3D_DIAGNOSTICS(VsMF,BaselineLevel, float32, "photo e-",   "DWS Baseline level vs time based on bin size", NOT_OVER_HQR, nH_, nMF_, nF_)
DECLARE_ZMWSTATDATASET_3D_DIAGNOSTICS(VsMF,BaselineStd,   float32, "photo e-",   "DWS Baseline sigma vs time based on bin size", NOT_OVER_HQR, nH_, nMF_, nF_)
DECLARE_ZMWSTATDATASET_3D_DIAGNOSTICS(VsMF,SnrMean,       float32, "",           "Mean Signal-to-Noise Ratio of DWS color channels", NOT_OVER_HQR, nH_, nMF_, nA_)
DECLARE_ZMWSTATDATASET_3D_DIAGNOSTICS(VsMF,Pkmid,         float32, "photo e-",   "Pkmid in HQ-region", NOT_OVER_HQR, nH_, nMF_, nA_)
DECLARE_ZMWSTATDATASET_2D_DIAGNOSTICS(VsMF,BaseRate,      float32, "bases/sec",  "Mean rate of base-incorporation events", NOT_OVER_HQR, nH_, nMF_)
DECLARE_ZMWSTATDATASET_2D_DIAGNOSTICS(VsMF,PulseRate,     float32, "pulses/sec", "Mean pulse rate over the trace", NOT_OVER_HQR, nH_, nMF_)
DECLARE_ZMWSTATDATASET_2D_DIAGNOSTICS(VsMF,BaseWidth,     float32, "frames",     "Mean width of base-incorporation events", NOT_OVER_HQR, nH_, nMF_)
DECLARE_ZMWSTATDATASET_2D_DIAGNOSTICS(VsMF,PulseWidth,    float32, "frames",     "Mean width of pulse events", NOT_OVER_HQR, nH_, nMF_)
DECLARE_ZMWSTATDATASET_2D_DIAGNOSTICS(VsMF,NumFrames,     float32, "frames",     "Block size in frames", NOT_OVER_HQR, nH_, nMF_)
//DECLARE_ZMWSTATDATASET_3D_DIAGNOSTICS(VsMF,NumPkmidFrames,float32, "frames",     "Number of frames per analog", NOT_OVER_HQR, nH_, nMF_, nA_)
DECLARE_ZMWSTAT_END_GROUP_DIAGNOSTICS(VsMF)

DECLARE_ZMWSTAT_END_GROUP(ZMWMetrics)

// /Zmw

DECLARE_ZMWSTAT_START_GROUP(ZMW)

DECLARE_ZMWSTATDATASET_2D(,HoleXY,               uint16,  "", "XY coordinate of the ZMWs on the chip", NOT_OVER_HQR, nH_ , two_)
DECLARE_ZMWSTATDATASET_1D(,HoleNumber,           uint32,  "", "Hole number", NOT_OVER_HQR, nH_)
DECLARE_ZMWSTATDATASET_1D(,UnitFeature,          uint32,  "", "Unit Feature", NOT_OVER_HQR, nH_)
DECLARE_ZMWSTATDATASET_1D(,HoleType,              uint8,  "", "Unit Type", NOT_OVER_HQR, nH_)

DECLARE_ZMWSTAT_END_GROUP(ZMW)
