
  $ ppa-reducestats --help | grep Version
  Version:* (glob)

  $ ppa-reducestats --showconfig
  {
  \t"ppa-reducestats" :  (esc)
  \t{ (esc)
  \t\t"BinCols" : 20, (esc)
  \t\t"BinRows" : 20, (esc)
  \t\t"MaxCols" : 65535, (esc)
  \t\t"MaxRows" : 65535, (esc)
  \t\t"MinOffsetX" : 0, (esc)
  \t\t"MinOffsetY" : 0, (esc)
  \t\t"Outputs" :  (esc)
  \t\t[ (esc)
  \t\t\t{ (esc)
  \t\t\t\t"Algorithm" : "Sum", (esc)
  \t\t\t\t"Filter" : "All", (esc)
  \t\t\t\t"Input" : "/ZMWMetrics/NumBases", (esc)
  \t\t\t\t"Type" : "float" (esc)
  \t\t\t}, (esc)
  \t\t\t{ (esc)
  \t\t\t\t"Algorithm" : "Median", (esc)
  \t\t\t\t"Filter" : "P1", (esc)
  \t\t\t\t"Input" : "/ZMWMetrics/HQPkmid", (esc)
  \t\t\t\t"Type" : "float" (esc)
  \t\t\t}, (esc)
  \t\t\t{ (esc)
  \t\t\t\t"Algorithm" : "Median", (esc)
  \t\t\t\t"Filter" : "Sequencing", (esc)
  \t\t\t\t"Input" : "/ZMWMetrics/SnrMean", (esc)
  \t\t\t\t"Type" : "float" (esc)
  \t\t\t}, (esc)
  \t\t\t{ (esc)
  \t\t\t\t"Algorithm" : "Median", (esc)
  \t\t\t\t"Filter" : "P1", (esc)
  \t\t\t\t"Input" : "/ZMWMetrics/HQRegionStart", (esc)
  \t\t\t\t"Type" : "float" (esc)
  \t\t\t}, (esc)
  \t\t\t{ (esc)
  \t\t\t\t"Algorithm" : "Median", (esc)
  \t\t\t\t"Filter" : "P1", (esc)
  \t\t\t\t"Input" : "/ZMWMetrics/HQRegionStartTime", (esc)
  \t\t\t\t"Type" : "uint16" (esc)
  \t\t\t}, (esc)
  \t\t\t{ (esc)
  \t\t\t\t"Algorithm" : "Count=0", (esc)
  \t\t\t\t"Filter" : "Sequencing", (esc)
  \t\t\t\t"Input" : "/ZMWMetrics/Loading", (esc)
  \t\t\t\t"Type" : "uint8" (esc)
  \t\t\t}, (esc)
  \t\t\t{ (esc)
  \t\t\t\t"Algorithm" : "Count=1", (esc)
  \t\t\t\t"Filter" : "Sequencing", (esc)
  \t\t\t\t"Input" : "/ZMWMetrics/Loading", (esc)
  \t\t\t\t"Type" : "uint8" (esc)
  \t\t\t}, (esc)
  \t\t\t{ (esc)
  \t\t\t\t"Algorithm" : "Count=2", (esc)
  \t\t\t\t"Filter" : "Sequencing", (esc)
  \t\t\t\t"Input" : "/ZMWMetrics/Loading", (esc)
  \t\t\t\t"Type" : "uint8" (esc)
  \t\t\t}, (esc)
  \t\t\t{ (esc)
  \t\t\t\t"Algorithm" : "Count=3", (esc)
  \t\t\t\t"Filter" : "Sequencing", (esc)
  \t\t\t\t"Input" : "/ZMWMetrics/Loading", (esc)
  \t\t\t\t"Type" : "uint8" (esc)
  \t\t\t}, (esc)
  \t\t\t{ (esc)
  \t\t\t\t"Algorithm" : "Count=0", (esc)
  \t\t\t\t"Filter" : "Sequencing", (esc)
  \t\t\t\t"Input" : "/ZMWMetrics/Productivity", (esc)
  \t\t\t\t"Type" : "uint8" (esc)
  \t\t\t}, (esc)
  \t\t\t{ (esc)
  \t\t\t\t"Algorithm" : "Count=1", (esc)
  \t\t\t\t"Filter" : "Sequencing", (esc)
  \t\t\t\t"Input" : "/ZMWMetrics/Productivity", (esc)
  \t\t\t\t"Type" : "uint8" (esc)
  \t\t\t}, (esc)
  \t\t\t{ (esc)
  \t\t\t\t"Algorithm" : "Count=2", (esc)
  \t\t\t\t"Filter" : "Sequencing", (esc)
  \t\t\t\t"Input" : "/ZMWMetrics/Productivity", (esc)
  \t\t\t\t"Type" : "uint8" (esc)
  \t\t\t}, (esc)
  \t\t\t{ (esc)
  \t\t\t\t"Algorithm" : "Subsample", (esc)
  \t\t\t\t"BinCols" : 1, (esc)
  \t\t\t\t"BinRows" : 1, (esc)
  \t\t\t\t"Filter" : "All", (esc)
  \t\t\t\t"Input" : "/ZMWMetrics/Productivity", (esc)
  \t\t\t\t"Type" : "uint8" (esc)
  \t\t\t}, (esc)
  \t\t\t{ (esc)
  \t\t\t\t"Algorithm" : "Median", (esc)
  \t\t\t\t"Filter" : "Sequencing", (esc)
  \t\t\t\t"Input" : "/ZMWMetrics/BaselineLevel", (esc)
  \t\t\t\t"Type" : "float" (esc)
  \t\t\t}, (esc)
  \t\t\t{ (esc)
  \t\t\t\t"Algorithm" : "Median", (esc)
  \t\t\t\t"Filter" : "P1", (esc)
  \t\t\t\t"Input" : "/ZMWMetrics/HQRegionSnrMean", (esc)
  \t\t\t\t"Type" : "float" (esc)
  \t\t\t}, (esc)
  \t\t\t{ (esc)
  \t\t\t\t"Algorithm" : "Mean", (esc)
  \t\t\t\t"Filter" : "Sequencing", (esc)
  \t\t\t\t"Input" : "/ZMWMetrics/BaselineLevel", (esc)
  \t\t\t\t"Type" : "float" (esc)
  \t\t\t}, (esc)
  \t\t\t{ (esc)
  \t\t\t\t"Algorithm" : "Median", (esc)
  \t\t\t\t"Filter" : "P1", (esc)
  \t\t\t\t"Input" : "/ZMWMetrics/LocalBaseRate", (esc)
  \t\t\t\t"Type" : "float" (esc)
  \t\t\t}, (esc)
  \t\t\t{ (esc)
  \t\t\t\t"Algorithm" : "Mean", (esc)
  \t\t\t\t"Filter" : "P1", (esc)
  \t\t\t\t"Input" : "/ZMWMetrics/ReadLength", (esc)
  \t\t\t\t"Type" : "float" (esc)
  \t\t\t}, (esc)
  \t\t\t{ (esc)
  \t\t\t\t"Algorithm" : "Mean", (esc)
  \t\t\t\t"Filter" : "P1", (esc)
  \t\t\t\t"Input" : "/ZMWMetrics/BaseWidth", (esc)
  \t\t\t\t"Type" : "float" (esc)
  \t\t\t}, (esc)
  \t\t\t{ (esc)
  \t\t\t\t"Algorithm" : "Mean", (esc)
  \t\t\t\t"Filter" : "P1", (esc)
  \t\t\t\t"Input" : "/ZMWMetrics/BaseIpd", (esc)
  \t\t\t\t"Type" : "float" (esc)
  \t\t\t} (esc)
  \t\t] (esc)
  \t} (esc)
  } (no-eol)

  $ a=/pbi/dept/primary/testdata/sim/softball_snr-50/softball_SNR-50_prod_min_360.sts.h5
  $ b=$CRAMTMP/out.rsts.h5
  $ ppa-reducestats --input $a --output $b >/dev/null
  $ ls -al  $b
  * /tmp/cramtests-*/out.rsts.h5 (glob)

  $ h5dump -g /ReducedZMWMetrics/ScanData $b
  HDF5 "/tmp/cramtests-*/out.rsts.h5" { (glob)
  GROUP "/ReducedZMWMetrics/ScanData" {
     ATTRIBUTE "FormatVersion" {
        DATATYPE  H5T_STRING {
           STRSIZE H5T_VARIABLE;
           STRPAD H5T_STR_NULLTERM;
           CSET H5T_CSET_ASCII;
           CTYPE H5T_C_S1;
        }
        DATASPACE  SCALAR
        DATA {
        (0): ""
        }
     }
     GROUP "AcqParams" {
        ATTRIBUTE "AduGain" {
           DATATYPE  H5T_IEEE_F32LE
           DATASPACE  SCALAR
           DATA {
           (0): 1
           }
        }
        ATTRIBUTE "CameraBias" {
           DATATYPE  H5T_IEEE_F32LE
           DATASPACE  SCALAR
           DATA {
           (0): 0
           }
        }
        ATTRIBUTE "CameraBiasStd" {
           DATATYPE  H5T_IEEE_F32LE
           DATASPACE  SCALAR
           DATA {
           (0): 0
           }
        }
        ATTRIBUTE "CameraGain" {
           DATATYPE  H5T_IEEE_F32LE
           DATASPACE  SCALAR
           DATA {
           (0): 0
           }
        }
        ATTRIBUTE "CameraType" {
           DATATYPE  H5T_STD_I32LE
           DATASPACE  SCALAR
           DATA {
           (0): 0
           }
        }
        ATTRIBUTE "FrameRate" {
           DATATYPE  H5T_IEEE_F32LE
           DATASPACE  SCALAR
           DATA {
           (0): 0
           }
        }
        ATTRIBUTE "HotStartFrame" {
           DATATYPE  H5T_STD_I32LE
           DATASPACE  SCALAR
           DATA {
           (0): 0
           }
        }
        ATTRIBUTE "HotStartFrameValid" {
           DATATYPE  H5T_STD_U8LE
           DATASPACE  SCALAR
           DATA {
           (0): 0
           }
        }
        ATTRIBUTE "LaserOnFrame" {
           DATATYPE  H5T_STD_I32LE
           DATASPACE  SCALAR
           DATA {
           (0): 0
           }
        }
        ATTRIBUTE "LaserOnFrameValid" {
           DATATYPE  H5T_STD_U8LE
           DATASPACE  SCALAR
           DATA {
           (0): 0
           }
        }
        ATTRIBUTE "NumFrames" {
           DATATYPE  H5T_STD_U32LE
           DATASPACE  SCALAR
           DATA {
           (0): 0
           }
        }
     }
     DATASET "AcquisitionXML" {
        DATATYPE  H5T_STRING {
           STRSIZE H5T_VARIABLE;
           STRPAD H5T_STR_NULLTERM;
           CSET H5T_CSET_ASCII;
           CTYPE H5T_C_S1;
        }
        DATASPACE  SCALAR
        DATA {
        (0): ""
        }
     }
     GROUP "ChipInfo" {
        ATTRIBUTE "LayoutName" {
           DATATYPE  H5T_STRING {
              STRSIZE H5T_VARIABLE;
              STRPAD H5T_STR_NULLTERM;
              CSET H5T_CSET_ASCII;
              CTYPE H5T_C_S1;
           }
           DATASPACE  SCALAR
           DATA {
           (0): "SequEL_4.0_RTO3"
           }
        }
        DATASET "AnalogRefSnr" {
           DATATYPE  H5T_IEEE_F32LE
           DATASPACE  SCALAR
           DATA {
           (0): 50
           }
        }
        DATASET "AnalogRefSpectrum" {
           DATATYPE  H5T_IEEE_F32LE
           DATASPACE  SIMPLE { ( 2 ) / ( 2 ) }
           DATA {
           (0): 0.102, 0.898
           }
        }
        DATASET "FilterMap" {
           DATATYPE  H5T_STD_U16LE
           DATASPACE  SIMPLE { ( 2 ) / ( 2 ) }
           DATA {
           (0): 1, 0
           }
        }
        DATASET "ImagePsf" {
           DATATYPE  H5T_IEEE_F32LE
           DATASPACE  SIMPLE { ( 2, 5, 5 ) / ( 2, 5, 5 ) }
           DATA {
           (0,0,0): 0, 0, 0, 0, 0,
           (0,1,0): 0, 0, 0, 0, 0,
           (0,2,0): 0, 0, 0, 0, 0,
           (0,3,0): 0, 0, 0, 0, 0,
           (0,4,0): 0, 0, 0, 0, 0,
           (1,0,0): 0, 0, 0, 0, 0,
           (1,1,0): 0, 0, 0, 0, 0,
           (1,2,0): 0, 0, 0, 0, 0,
           (1,3,0): 0, 0, 0, 0, 0,
           (1,4,0): 0, 0, 0, 0, 0
           }
        }
        DATASET "XtalkCorrection" {
           DATATYPE  H5T_IEEE_F32LE
           DATASPACE  SIMPLE { ( 7, 7 ) / ( 7, 7 ) }
           DATA {
           (0,0): 0, 0, 0, 0, 0, 0, 0,
           (1,0): 0, 0, 0, 0, 0, 0, 0,
           (2,0): 0, 0, 0, 0, 0, 0, 0,
           (3,0): 0, 0, 0, 0, 0, 0, 0,
           (4,0): 0, 0, 0, 0, 0, 0, 0,
           (5,0): 0, 0, 0, 0, 0, 0, 0,
           (6,0): 0, 0, 0, 0, 0, 0, 0
           }
        }
     }
     GROUP "DyeSet" {
        ATTRIBUTE "BaseMap" {
           DATATYPE  H5T_STRING {
              STRSIZE H5T_VARIABLE;
              STRPAD H5T_STR_NULLTERM;
              CSET H5T_CSET_ASCII;
              CTYPE H5T_C_S1;
           }
           DATASPACE  SCALAR
           DATA {
           (0): ""
           }
        }
        ATTRIBUTE "NumAnalog" {
           DATATYPE  H5T_STD_U16LE
           DATASPACE  SCALAR
           DATA {
           (0): 4
           }
        }
        DATASET "AnalogSpectra" {
           DATATYPE  H5T_IEEE_F32LE
           DATASPACE  SIMPLE { ( 4, 2 ) / ( 4, 2 ) }
           DATA {
           (0,0): 0.4727, 0.5273,
           (1,0): 0.4727, 0.5273,
           (2,0): 0.102, 0.898,
           (3,0): 0.102, 0.898
           }
        }
        DATASET "ExcessNoiseCV" {
           DATATYPE  H5T_IEEE_F32LE
           DATASPACE  SIMPLE { ( 4 ) / ( 4 ) }
           DATA {
           (0): 0.1, 0.1, 0.1, 0.1
           }
        }
        DATASET "Ipd2SlowStepRatio" {
           DATATYPE  H5T_IEEE_F32LE
           DATASPACE  SIMPLE { ( 4 ) / ( 4 ) }
           DATA {
           (0): 0, 0, 0, 0
           }
        }
        DATASET "IpdMean" {
           DATATYPE  H5T_IEEE_F32LE
           DATASPACE  SIMPLE { ( 4 ) / ( 4 ) }
           DATA {
           (0): 0.162402, 0.224865, 0.174895, 0.174895
           }
        }
        DATASET "PulseWidthMean" {
           DATATYPE  H5T_IEEE_F32LE
           DATASPACE  SIMPLE { ( 4 ) / ( 4 ) }
           DATA {
           (0): 0.124925, 0.0874474, 0.112432, 0.112432
           }
        }
        DATASET "Pw2SlowStepRatio" {
           DATATYPE  H5T_IEEE_F32LE
           DATASPACE  SIMPLE { ( 4 ) / ( 4 ) }
           DATA {
           (0): 0, 0, 0, 0
           }
        }
        DATASET "RelativeAmp" {
           DATATYPE  H5T_IEEE_F32LE
           DATASPACE  SIMPLE { ( 4 ) / ( 4 ) }
           DATA {
           (0): 0.83, 0.53618, 1, 0.529
           }
        }
     }
     GROUP "RunInfo" {
        ATTRIBUTE "BindingKit" {
           DATATYPE  H5T_STRING {
              STRSIZE H5T_VARIABLE;
              STRPAD H5T_STR_NULLTERM;
              CSET H5T_CSET_ASCII;
              CTYPE H5T_C_S1;
           }
           DATASPACE  SCALAR
           DATA {
           (0): ""
           }
        }
        ATTRIBUTE "Control" {
           DATATYPE  H5T_STRING {
              STRSIZE H5T_VARIABLE;
              STRPAD H5T_STR_NULLTERM;
              CSET H5T_CSET_ASCII;
              CTYPE H5T_C_S1;
           }
           DATASPACE  SCALAR
           DATA {
           (0): ""
           }
        }
        ATTRIBUTE "InstrumentName" {
           DATATYPE  H5T_STRING {
              STRSIZE H5T_VARIABLE;
              STRPAD H5T_STR_NULLTERM;
              CSET H5T_CSET_ASCII;
              CTYPE H5T_C_S1;
           }
           DATASPACE  SCALAR
           DATA {
           (0): ""
           }
        }
        ATTRIBUTE "IsControlUsed" {
           DATATYPE  H5T_STD_U8LE
           DATASPACE  SCALAR
           DATA {
           (0): 0
           }
        }
        ATTRIBUTE "MovieName" {
           DATATYPE  H5T_STRING {
              STRSIZE H5T_VARIABLE;
              STRPAD H5T_STR_NULLTERM;
              CSET H5T_CSET_ASCII;
              CTYPE H5T_C_S1;
           }
           DATASPACE  SCALAR
           DATA {
           (0): ""
           }
        }
        ATTRIBUTE "MoviePath" {
           DATATYPE  H5T_STRING {
              STRSIZE H5T_VARIABLE;
              STRPAD H5T_STR_NULLTERM;
              CSET H5T_CSET_ASCII;
              CTYPE H5T_C_S1;
           }
           DATASPACE  SCALAR
           DATA {
           (0): ""
           }
        }
        ATTRIBUTE "PlatformId" {
           DATATYPE  H5T_STD_U32LE
           DATASPACE  SCALAR
           DATA {
           (0): 0
           }
        }
        ATTRIBUTE "PlatformName" {
           DATATYPE  H5T_STRING {
              STRSIZE H5T_VARIABLE;
              STRPAD H5T_STR_NULLTERM;
              CSET H5T_CSET_ASCII;
              CTYPE H5T_C_S1;
           }
           DATASPACE  SCALAR
           DATA {
           (0): ""
           }
        }
        ATTRIBUTE "SequencingChemistry" {
           DATATYPE  H5T_STRING {
              STRSIZE H5T_VARIABLE;
              STRPAD H5T_STR_NULLTERM;
              CSET H5T_CSET_ASCII;
              CTYPE H5T_C_S1;
           }
           DATASPACE  SCALAR
           DATA {
           (0): ""
           }
        }
        ATTRIBUTE "SequencingKit" {
           DATATYPE  H5T_STRING {
              STRSIZE H5T_VARIABLE;
              STRPAD H5T_STR_NULLTERM;
              CSET H5T_CSET_ASCII;
              CTYPE H5T_C_S1;
           }
           DATASPACE  SCALAR
           DATA {
           (0): ""
           }
        }
     }
  }
  }
