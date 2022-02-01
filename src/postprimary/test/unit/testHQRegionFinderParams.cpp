#include <gtest/gtest.h>

#include <iostream>
#include <memory>
#include <vector>
#include <thread>
#include <chrono>
#include <random>
#include <string>
#include <ctime>
#include <chrono>
#include <sstream>
#include <assert.h>

#include <postprimary/hqrf/BlockHQRegionFinder.h>
#include <postprimary/hqrf/HQRegionFinderParams.h>

using namespace PacBio::Primary;
using namespace PacBio::Primary::Postprimary;

std::string JSON_HEADER =
    "{\"TYPE\":\"BAZ\", \"HEADER\":"
        "{\"MOVIE_NAME\":\"ArminsFakeMovie\",\"COMPLETE\":1,\"BASE_CALLER_VERSION\":\"1.2\",\"BAZWRITER_VERSION\":\"1.0\",\"BAZ2BAM_VERSION\":\"3.3\",\"BAZ_MAJOR_VERSION\" : 0,\"BAZ_MINOR_VERSION\" : "
        "1,\"BAZ_PATCH_VERSION\" : 0,\"FRAME_RATE_HZ\" : 80,\"MOVIE_LENGTH_FRAMES\" : 1080000,"
        "\"BASECALLER_CONFIG\": \"{}\","
        "\"EXPERIMENT_METADATA\" : \"{\\\"AcqParams\\\":{\\\"AduGain\\\":1.9531248807907104},\\\"ChipInfo\\\":{\\\"AnalogRefSnr\\\":11,\\\"AnalogRefSpectrum\\\":[0.10197792202234268,0.8980221152305603],\\\"CrosstalkFilter\\\":[[0.00011132592771900818,0.00043121803901158273,0.00081435841275379062,0.0010630703764036298,0.00087830686243250966,0.00046748583554290235,0.00012474130198825151],[0.00040487214573659003,-0.00203699991106987,-0.003298585768789053,-0.0051095252856612206,-0.0040731276385486126,-0.0018567438237369061,0.00044959798105992377],[0.0008288127719424665,-0.0025635270867496729,-0.030308688059449196,-0.092926815152168274,-0.030468275770545006,-0.0043287212029099464,0.00080724310828372836],[0.0012212963774800301,-0.011045822873711586,-0.076042637228965759,1.5124297142028809,-0.085587434470653534,-0.0050257253460586071,0.00087140261894091964],[0.00093532569007948041,-0.0050715520046651363,-0.027508752420544624,-0.095789402723312378,-0.027349235489964485,-0.00034580400097183883,0.0005108729237690568],[0.00046027876669541001,-0.0018478382844477892,-0.0040882616303861141,-0.0075986571609973907,-0.0033990705851465464,-0.00060037523508071899,0.00022889320098329335],[0.00012168431567260996,0.00047721769078634679,0.00092700554523617029,0.0012106377398595214,0.00080913538113236427,0.00030304939718917012,5.5799839174142107e-05]],\\\"FilterMap\\\":[1,0],\\\"ImagePsf\\\":[[[-0.0010000000474974513,0.0038000000640749931,0.0066999997943639755,0.0038000000640749931,0.0019000000320374966],[-0.0010000000474974513,0.017200000584125519,0.061999998986721039,0.015300000086426735,0.0038000000640749931],[0.0010000000474974513,0.047699999064207077,0.66920000314712524,0.037200000137090683,0.0076000001281499863],[0.002899999963119626,0.021900000050663948,0.06289999932050705,0.019099999219179153,0.0010000000474974513],[0.0019000000320374966,0.0048000002279877663,0.0048000002279877663,0.0038000000640749931,0.0019000000320374966]],[[0.0030000000260770321,0.0040000001899898052,0.0080000003799796104,0.004999999888241291,0.0020000000949949026],[0.004999999888241291,0.023000000044703484,0.054900001734495163,0.02500000037252903,0.0060000000521540642],[0.0099999997764825821,0.057900000363588333,0.60039997100830078,0.057900000363588333,0.0099999997764825821],[0.0060000000521540642,0.02199999988079071,0.05090000107884407,0.024000000208616257,0.0060000000521540642],[0.0020000000949949026,0.0040000001899898052,0.0070000002160668373,0.0040000001899898052,0.0020000000949949026]]],\\\"LayoutName\\\":\\\"SequEL_4.0_RTO3\\\"},\\\"DyeSet\\\":{\\\"AnalogSpectra\\\":[[-0.052940212190151215,1.0529402494430542],[-0.052940212190151215,1.0529402494430542],[-0.052940212190151215,1.0529402494430542],[-0.052940212190151215,1.0529402494430542]],\\\"BaseMap\\\":\\\"CTAG\\\",\\\"ExcessNoiseCV\\\":[0.10000000149011612,0.10000000149011612,0.10000000149011612,0.10000000149011612],\\\"IpdMean\\\":[0.30656999349594116,0.32460001111030579,0.28850001096725464,0.30656999349594116],\\\"NumAnalogs\\\":4,\\\"PulseWidthMean\\\":[0.21639999747276306,0.14427000284194946,0.18029999732971191,0.21639999747276306],\\\"RelativeAmp\\\":[1,0.63636398315429688,0.43181800842285156,0.22727300226688385]},\\\"basecallerVersion\\\":\\\"?\\\",\\\"bazfile\\\":\\\"\\\",\\\"cameraConfigKey\\\":0,\\\"chipId\\\":\\\"n/a\\\",\\\"dryrun\\\":false,\\\"expectedFrameRate\\\":100,\\\"exposure\\\":0.01,\\\"hdf5output\\\":\\\"\\\",\\\"instrumentName\\\":\\\"n/a\\\",\\\"metricsVerbosity\\\":\\\"MINIMAL\\\",\\\"minSnr\\\":4,\\\"movieContext\\\":\\\"\\\",\\\"noCalibration\\\":false,\\\"numFrames\\\":0,\\\"numPixelLanes\\\":[],\\\"numZmwLanes\\\":[],\\\"numZmws\\\":1171456,\\\"photoelectronSensitivity\\\":1,\\\"readout\\\":\\\"BASES\\\",\\\"refDwsSnr\\\":11,\\\"refSpectrum\\\":[0.10197792202234268,0.8980221152305603],\\\"remapVroiOption\\\":false,\\\"roiMetaData\\\":null}\","
        "\"LF_METRIC\" : {\"FIELDS\" : [[ \"BASELINE_RED_SD\",16,true,10],"
        "[\"PULSE_WIDTH\",16,false,1], [\"BASE_WIDTH\",16,false,1], [ "
        "\"BASELINE_GREEN_SD\",16,true,10]"
        ",[\"BASELINE_RED_MEAN\",16,true,10]"
        ",[\"BASELINE_GREEN_MEAN\",16,true,10]"
        ",[\"PKMID_A\",16,true,10]"
        ",[\"PKMID_C\",16,true,10]"
        ",[\"PKMID_G\",16,true,10]"
        ",[\"PKMID_T\",16,true,10]"
        ",[\"ANGLE_A\",16,true,100]"
        ",[\"ANGLE_C\",16,true,100]"
        ",[\"ANGLE_G\",16,true,100]"
        ",[\"ANGLE_T\",16,true,100]"
        ",[\"PKMID_FRAMES_A\",16,false,1]"
        ",[\"PKMID_FRAMES_C\",16,false,1]"
        ",[\"PKMID_FRAMES_G\",16,false,1]"
        ",[\"PKMID_FRAMES_T\",16,false,1]"
        ",[\"NUM_PULSES\",16,false,1]"
        ",[\"NUM_FRAMES\",16,false,1]"
        ",[ \"NUM_BASES\",16,false,1]],\"FRAMES\" : "
        "16384},\"MF_METRIC\" : {\"FIELDS\" : [[ \"NUM_FRAMES\",16,false,1],[\"NUM_PULSES\",16,false,1]"
        ",[\"NUM_PULSES_A\",16,false,1]"
        ",[\"NUM_PULSES_C\",16,false,1]"
        ",[\"NUM_PULSES_G\",16,false,1]"
        ",[\"NUM_PULSES_T\",16,false,1]"
        ",[\"NUM_BASES\",16,false,1]"
        ",[\"PULSE_WIDTH\",16,false,1]"
        ",[\"NUM_SANDWICHES\",16,false,1]"
        ",[\"NUM_HALF_SANDWICHES\",16,false,1]"
        ",[\"NUM_PULSE_LABEL_STUTTERS\",16,false,1]"
        "],\"FRAMES\" : "
        "4096},"
        "\"HF_METRIC\":{\"FIELDS\":[[\"NUM_BASES\",16,false,1],[\"NUM_FRAMES\",16,false,1],[\"NUM_PULSES\",16,false,1]],\"FRAMES\":1024},"
        "\"PACKET\" : [[ \"READOUT\", 2 ],[ \"DEL_TAG\", 3 ],[ "
        "\"SUB_TAG\", 3 ],[ \"DEL_QV\", 4 ],[ \"SUB_QV\", 4 ],[ \"INS_QV\", 4 "
        "],[ \"MRG_QV\", 4 ],[ \"IPD_LL\", 8, 255, \"IPD16_LL\", 16 "
        "]],\"SLICE_LENGTH_FRAMES\" : 16384}}";

std::string JSON_HEADER_SF0 =
    "{\"TYPE\":\"BAZ\", \"HEADER\":"
        "{\"MOVIE_NAME\":\"ArminsFakeMovie\",\"COMPLETE\":1,\"BASE_CALLER_VERSION\":\"1.2\",\"BAZWRITER_VERSION\":\"1.0\",\"BAZ2BAM_VERSION\":\"3.3\",\"BAZ_MAJOR_VERSION\" : 0,\"BAZ_MINOR_VERSION\" : "
        "1,\"BAZ_PATCH_VERSION\" : 0,\"FRAME_RATE_HZ\" : 80,\"MOVIE_LENGTH_FRAMES\" : 1080000,"
        "\"BASECALLER_CONFIG\" : \"{\\\"algorithm\\\":{\\\"BaselineFilter\\\":{\\\"Method\\\":\\\"MultiScaleSmall\\\"},\\\"Metrics\\\":{\\\"Method\\\":\\\"HFMetrics\\\",\\\"sandwichTolerance\\\":0},\\\"PulseDetection\\\":{\\\"Alpha\\\":1,\\\"Beta\\\":1,\\\"Gamma\\\":0.05000000074505806,\\\"LowerThreshold\\\":2,\\\"Method\\\":\\\"SubFrameHmm\\\",\\\"UpperThreshold\\\":6},\\\"PulseToBase\\\":{\\\"BasesPerBlock\\\":50,\\\"Method\\\":\\\"ExShortPulse\\\",\\\"ModelPhases\\\":1,\\\"Phase1TreeFilePath\\\":\\\"\\\",\\\"Phase2TreeFilePath\\\":\\\"\\\",\\\"SnrThresh\\\":100,\\\"XsfAmpThresh\\\":0.5,\\\"XspAmpThresh\\\":0.59999999999999998,\\\"XspWidthThresh\\\":3.5},\\\"dme\\\":{\\\"AngleEstPoolData\\\":true,\\\"BinSizeCoeff\\\":1,\\\"ChiSqrThresh\\\":14,\\\"ConvergeCoeff\\\":2.9999999242136255e-05,\\\"EnableDmeBinTruncation\\\":true,\\\"EnableDmeBinning\\\":true,\\\"FastExponential\\\":false,\\\"FirstAttemptsIterMultiple\\\":4,\\\"IterationLimit\\\":20,\\\"Method\\\":\\\"TwoPhase\\\",\\\"MinFramesForEstimate\\\":4000,\\\"MinSkipFrames\\\":0,\\\"ModelUpdateWeightMax\\\":0.5,\\\"NumBinsMin\\\":500,\\\"PeslFractile\\\":0.89999997615814209,\\\"PureUpdate\\\":false,\\\"SpiderSimModel\\\":{\\\"AAmp\\\":0.65909093618392944,\\\"CAmp\\\":1,\\\"GAmp\\\":0.38636362552642822,\\\"RefSNR\\\":60,\\\"TAmp\\\":0.22727271914482117,\\\"baselineMean\\\":200,\\\"baselineVar\\\":33,\\\"pulseCV\\\":0.10000000149011612,\\\"shotCoeff\\\":1.3700000047683716},\\\"ThreshAnalogSNR\\\":2.1489999294281006},\\\"pipe\\\":\\\"Sequel\\\"},\\\"init\\\":{\\\"bindCores\\\":false,\\\"micDieTemperatureFatalThreshold\\\":95,\\\"micDieTemperatureWarningThreshold\\\":80,\\\"numWorkerThreads_k1om\\\":240,\\\"numWorkerThreads_x86_64\\\":16,\\\"trancheFreeQueueWarningTimeout\\\":50000}}\","
        "\"EXPERIMENT_METADATA\" : \"{\\\"AcqParams\\\":{\\\"AduGain\\\":1.9531248807907104},\\\"ChipInfo\\\":{\\\"AnalogRefSnr\\\":11,\\\"AnalogRefSpectrum\\\":[0.10197792202234268,0.8980221152305603],\\\"CrosstalkFilter\\\":[[0.00011132592771900818,0.00043121803901158273,0.00081435841275379062,0.0010630703764036298,0.00087830686243250966,0.00046748583554290235,0.00012474130198825151],[0.00040487214573659003,-0.00203699991106987,-0.003298585768789053,-0.0051095252856612206,-0.0040731276385486126,-0.0018567438237369061,0.00044959798105992377],[0.0008288127719424665,-0.0025635270867496729,-0.030308688059449196,-0.092926815152168274,-0.030468275770545006,-0.0043287212029099464,0.00080724310828372836],[0.0012212963774800301,-0.011045822873711586,-0.076042637228965759,1.5124297142028809,-0.085587434470653534,-0.0050257253460586071,0.00087140261894091964],[0.00093532569007948041,-0.0050715520046651363,-0.027508752420544624,-0.095789402723312378,-0.027349235489964485,-0.00034580400097183883,0.0005108729237690568],[0.00046027876669541001,-0.0018478382844477892,-0.0040882616303861141,-0.0075986571609973907,-0.0033990705851465464,-0.00060037523508071899,0.00022889320098329335],[0.00012168431567260996,0.00047721769078634679,0.00092700554523617029,0.0012106377398595214,0.00080913538113236427,0.00030304939718917012,5.5799839174142107e-05]],\\\"FilterMap\\\":[1,0],\\\"ImagePsf\\\":[[[-0.0010000000474974513,0.0038000000640749931,0.0066999997943639755,0.0038000000640749931,0.0019000000320374966],[-0.0010000000474974513,0.017200000584125519,0.061999998986721039,0.015300000086426735,0.0038000000640749931],[0.0010000000474974513,0.047699999064207077,0.66920000314712524,0.037200000137090683,0.0076000001281499863],[0.002899999963119626,0.021900000050663948,0.06289999932050705,0.019099999219179153,0.0010000000474974513],[0.0019000000320374966,0.0048000002279877663,0.0048000002279877663,0.0038000000640749931,0.0019000000320374966]],[[0.0030000000260770321,0.0040000001899898052,0.0080000003799796104,0.004999999888241291,0.0020000000949949026],[0.004999999888241291,0.023000000044703484,0.054900001734495163,0.02500000037252903,0.0060000000521540642],[0.0099999997764825821,0.057900000363588333,0.60039997100830078,0.057900000363588333,0.0099999997764825821],[0.0060000000521540642,0.02199999988079071,0.05090000107884407,0.024000000208616257,0.0060000000521540642],[0.0020000000949949026,0.0040000001899898052,0.0070000002160668373,0.0040000001899898052,0.0020000000949949026]]],\\\"LayoutName\\\":\\\"SequEL_4.0_RTO3\\\"},\\\"DyeSet\\\":{\\\"AnalogSpectra\\\":[[-0.052940212190151215,1.0529402494430542],[-0.052940212190151215,1.0529402494430542],[-0.052940212190151215,1.0529402494430542],[-0.052940212190151215,1.0529402494430542]],\\\"BaseMap\\\":\\\"CTAG\\\",\\\"ExcessNoiseCV\\\":[0.10000000149011612,0.10000000149011612,0.10000000149011612,0.10000000149011612],\\\"IpdMean\\\":[0.30656999349594116,0.32460001111030579,0.28850001096725464,0.30656999349594116],\\\"NumAnalogs\\\":4,\\\"PulseWidthMean\\\":[0.21639999747276306,0.14427000284194946,0.18029999732971191,0.21639999747276306],\\\"RelativeAmp\\\":[1,0.63636398315429688,0.43181800842285156,0.22727300226688385]},\\\"basecallerVersion\\\":\\\"?\\\",\\\"bazfile\\\":\\\"\\\",\\\"cameraConfigKey\\\":0,\\\"chipId\\\":\\\"n/a\\\",\\\"dryrun\\\":false,\\\"expectedFrameRate\\\":100,\\\"exposure\\\":0.01,\\\"hdf5output\\\":\\\"\\\",\\\"instrumentName\\\":\\\"n/a\\\",\\\"metricsVerbosity\\\":\\\"MINIMAL\\\",\\\"minSnr\\\":4,\\\"movieContext\\\":\\\"\\\",\\\"noCalibration\\\":false,\\\"numFrames\\\":0,\\\"numPixelLanes\\\":[],\\\"numZmwLanes\\\":[],\\\"numZmws\\\":1171456,\\\"photoelectronSensitivity\\\":1,\\\"readout\\\":\\\"BASES\\\",\\\"refDwsSnr\\\":11,\\\"refSpectrum\\\":[0.10197792202234268,0.8980221152305603],\\\"remapVroiOption\\\":false,\\\"roiMetaData\\\":null}\","
        "\"LF_METRIC\" : {\"FIELDS\" : [[ \"BASELINE_RED_SD\",16,true,10],"
        "[\"PULSE_WIDTH\",16,false,1], [\"BASE_WIDTH\",16,false,1], [ "
        "\"BASELINE_GREEN_SD\",16,true,10]"
        ",[\"BASELINE_RED_MEAN\",16,true,10]"
        ",[\"BASELINE_GREEN_MEAN\",16,true,10]"
        ",[\"PKMID_A\",16,true,10]"
        ",[\"PKMID_C\",16,true,10]"
        ",[\"PKMID_G\",16,true,10]"
        ",[\"PKMID_T\",16,true,10]"
        ",[\"ANGLE_A\",16,true,100]"
        ",[\"ANGLE_C\",16,true,100]"
        ",[\"ANGLE_G\",16,true,100]"
        ",[\"ANGLE_T\",16,true,100]"
        ",[\"PKMID_FRAMES_A\",16,false,1]"
        ",[\"PKMID_FRAMES_C\",16,false,1]"
        ",[\"PKMID_FRAMES_G\",16,false,1]"
        ",[\"PKMID_FRAMES_T\",16,false,1]"
        ",[\"NUM_PULSES\",16,false,1]"
        ",[\"NUM_FRAMES\",16,false,1]"
        ",[ \"NUM_BASES\",16,false,1]],\"FRAMES\" : "
        "16384},\"MF_METRIC\" : {\"FIELDS\" : [[ \"NUM_FRAMES\",16,false,1],[\"NUM_PULSES\",16,false,1]"
        ",[\"NUM_PULSES_A\",16,false,1]"
        ",[\"NUM_PULSES_C\",16,false,1]"
        ",[\"NUM_PULSES_G\",16,false,1]"
        ",[\"NUM_PULSES_T\",16,false,1]"
        ",[\"NUM_BASES\",16,false,1]"
        ",[\"PULSE_WIDTH\",16,false,1]"
        ",[\"NUM_SANDWICHES\",16,false,1]"
        ",[\"NUM_HALF_SANDWICHES\",16,false,1]"
        ",[\"NUM_PULSE_LABEL_STUTTERS\",16,false,1]"
        "],\"FRAMES\" : "
        "4096},"
        "\"HF_METRIC\":{\"FIELDS\":[[\"NUM_BASES\",16,false,1],[\"NUM_FRAMES\",16,false,1],[\"NUM_PULSES\",16,false,1]],\"FRAMES\":1024},"
        "\"PACKET\" : [[ \"READOUT\", 2 ],[ \"DEL_TAG\", 3 ],[ "
        "\"SUB_TAG\", 3 ],[ \"DEL_QV\", 4 ],[ \"SUB_QV\", 4 ],[ \"INS_QV\", 4 "
        "],[ \"MRG_QV\", 4 ],[ \"IPD_LL\", 8, 255, \"IPD16_LL\", 16 "
        "]],\"SLICE_LENGTH_FRAMES\" : 16384}}";

std::string JSON_HEADER_SPIDER =
    "{\"TYPE\":\"BAZ\", \"HEADER\":"
        "{\"MOVIE_NAME\":\"ArminsFakeMovie\",\"COMPLETE\":1,\"BASE_CALLER_VERSION\":\"1.2\",\"BAZWRITER_VERSION\":\"1.0\",\"BAZ2BAM_VERSION\":\"3.3\",\"BAZ_MAJOR_VERSION\" : 0,\"BAZ_MINOR_VERSION\" : "
        "1,\"BAZ_PATCH_VERSION\" : 0,\"FRAME_RATE_HZ\" : 80,\"MOVIE_LENGTH_FRAMES\" : 1080000,"
        "\"BASECALLER_CONFIG\" : \"{\\\"algorithm\\\":{\\\"BaselineFilter\\\":{\\\"Method\\\":\\\"MultiScaleSmall\\\"},\\\"Metrics\\\":{\\\"Method\\\":\\\"HFMetrics\\\",\\\"sandwichTolerance\\\":0},\\\"PulseDetection\\\":{\\\"Alpha\\\":1,\\\"Beta\\\":1,\\\"Gamma\\\":0.05000000074505806,\\\"LowerThreshold\\\":2,\\\"Method\\\":\\\"SubFrameHmm\\\",\\\"UpperThreshold\\\":6},\\\"PulseToBase\\\":{\\\"BasesPerBlock\\\":50,\\\"Method\\\":\\\"ExShortPulse\\\",\\\"ModelPhases\\\":1,\\\"Phase1TreeFilePath\\\":\\\"\\\",\\\"Phase2TreeFilePath\\\":\\\"\\\",\\\"SnrThresh\\\":100,\\\"XsfAmpThresh\\\":0.5,\\\"XspAmpThresh\\\":0.59999999999999998,\\\"XspWidthThresh\\\":3.5},\\\"dme\\\":{\\\"AngleEstPoolData\\\":true,\\\"BinSizeCoeff\\\":1,\\\"ChiSqrThresh\\\":14,\\\"ConvergeCoeff\\\":2.9999999242136255e-05,\\\"EnableDmeBinTruncation\\\":true,\\\"EnableDmeBinning\\\":true,\\\"FastExponential\\\":false,\\\"FirstAttemptsIterMultiple\\\":4,\\\"IterationLimit\\\":20,\\\"Method\\\":\\\"TwoPhase\\\",\\\"MinFramesForEstimate\\\":4000,\\\"MinSkipFrames\\\":0,\\\"ModelUpdateWeightMax\\\":0.5,\\\"NumBinsMin\\\":500,\\\"PeslFractile\\\":0.89999997615814209,\\\"PureUpdate\\\":false,\\\"SpiderSimModel\\\":{\\\"AAmp\\\":0.65909093618392944,\\\"CAmp\\\":1,\\\"GAmp\\\":0.38636362552642822,\\\"RefSNR\\\":60,\\\"TAmp\\\":0.22727271914482117,\\\"baselineMean\\\":200,\\\"baselineVar\\\":33,\\\"pulseCV\\\":0.10000000149011612,\\\"shotCoeff\\\":1.3700000047683716},\\\"ThreshAnalogSNR\\\":2.1489999294281006},\\\"pipe\\\":\\\"Spider16\\\"},\\\"init\\\":{\\\"bindCores\\\":false,\\\"micDieTemperatureFatalThreshold\\\":95,\\\"micDieTemperatureWarningThreshold\\\":80,\\\"numWorkerThreads_k1om\\\":240,\\\"numWorkerThreads_x86_64\\\":16,\\\"trancheFreeQueueWarningTimeout\\\":50000}}\","
        "\"EXPERIMENT_METADATA\" : \"{\\\"AcqParams\\\":{\\\"AduGain\\\":1.9531248807907104},\\\"ChipInfo\\\":{\\\"AnalogRefSnr\\\":11,\\\"AnalogRefSpectrum\\\":[0.10197792202234268,0.8980221152305603],\\\"CrosstalkFilter\\\":[[0.00011132592771900818,0.00043121803901158273,0.00081435841275379062,0.0010630703764036298,0.00087830686243250966,0.00046748583554290235,0.00012474130198825151],[0.00040487214573659003,-0.00203699991106987,-0.003298585768789053,-0.0051095252856612206,-0.0040731276385486126,-0.0018567438237369061,0.00044959798105992377],[0.0008288127719424665,-0.0025635270867496729,-0.030308688059449196,-0.092926815152168274,-0.030468275770545006,-0.0043287212029099464,0.00080724310828372836],[0.0012212963774800301,-0.011045822873711586,-0.076042637228965759,1.5124297142028809,-0.085587434470653534,-0.0050257253460586071,0.00087140261894091964],[0.00093532569007948041,-0.0050715520046651363,-0.027508752420544624,-0.095789402723312378,-0.027349235489964485,-0.00034580400097183883,0.0005108729237690568],[0.00046027876669541001,-0.0018478382844477892,-0.0040882616303861141,-0.0075986571609973907,-0.0033990705851465464,-0.00060037523508071899,0.00022889320098329335],[0.00012168431567260996,0.00047721769078634679,0.00092700554523617029,0.0012106377398595214,0.00080913538113236427,0.00030304939718917012,5.5799839174142107e-05]],\\\"FilterMap\\\":[1,0],\\\"ImagePsf\\\":[[[-0.0010000000474974513,0.0038000000640749931,0.0066999997943639755,0.0038000000640749931,0.0019000000320374966],[-0.0010000000474974513,0.017200000584125519,0.061999998986721039,0.015300000086426735,0.0038000000640749931],[0.0010000000474974513,0.047699999064207077,0.66920000314712524,0.037200000137090683,0.0076000001281499863],[0.002899999963119626,0.021900000050663948,0.06289999932050705,0.019099999219179153,0.0010000000474974513],[0.0019000000320374966,0.0048000002279877663,0.0048000002279877663,0.0038000000640749931,0.0019000000320374966]],[[0.0030000000260770321,0.0040000001899898052,0.0080000003799796104,0.004999999888241291,0.0020000000949949026],[0.004999999888241291,0.023000000044703484,0.054900001734495163,0.02500000037252903,0.0060000000521540642],[0.0099999997764825821,0.057900000363588333,0.60039997100830078,0.057900000363588333,0.0099999997764825821],[0.0060000000521540642,0.02199999988079071,0.05090000107884407,0.024000000208616257,0.0060000000521540642],[0.0020000000949949026,0.0040000001899898052,0.0070000002160668373,0.0040000001899898052,0.0020000000949949026]]],\\\"LayoutName\\\":\\\"SequEL_4.0_RTO3\\\"},\\\"DyeSet\\\":{\\\"AnalogSpectra\\\":[[-0.052940212190151215,1.0529402494430542],[-0.052940212190151215,1.0529402494430542],[-0.052940212190151215,1.0529402494430542],[-0.052940212190151215,1.0529402494430542]],\\\"BaseMap\\\":\\\"CTAG\\\",\\\"ExcessNoiseCV\\\":[0.10000000149011612,0.10000000149011612,0.10000000149011612,0.10000000149011612],\\\"IpdMean\\\":[0.30656999349594116,0.32460001111030579,0.28850001096725464,0.30656999349594116],\\\"NumAnalogs\\\":4,\\\"PulseWidthMean\\\":[0.21639999747276306,0.14427000284194946,0.18029999732971191,0.21639999747276306],\\\"RelativeAmp\\\":[1,0.63636398315429688,0.43181800842285156,0.22727300226688385]},\\\"basecallerVersion\\\":\\\"?\\\",\\\"bazfile\\\":\\\"\\\",\\\"cameraConfigKey\\\":0,\\\"chipId\\\":\\\"n/a\\\",\\\"dryrun\\\":false,\\\"expectedFrameRate\\\":100,\\\"exposure\\\":0.01,\\\"hdf5output\\\":\\\"\\\",\\\"instrumentName\\\":\\\"n/a\\\",\\\"metricsVerbosity\\\":\\\"MINIMAL\\\",\\\"minSnr\\\":4,\\\"movieContext\\\":\\\"\\\",\\\"noCalibration\\\":false,\\\"numFrames\\\":0,\\\"numPixelLanes\\\":[],\\\"numZmwLanes\\\":[],\\\"numZmws\\\":1171456,\\\"photoelectronSensitivity\\\":1,\\\"readout\\\":\\\"BASES\\\",\\\"refDwsSnr\\\":11,\\\"refSpectrum\\\":[0.10197792202234268,0.8980221152305603],\\\"remapVroiOption\\\":false,\\\"roiMetaData\\\":null}\","
        "\"LF_METRIC\" : {\"FIELDS\" : [[ \"BASELINE_RED_SD\",16,true,10],"
        "[\"PULSE_WIDTH\",16,false,1], [\"BASE_WIDTH\",16,false,1], [ "
        "\"BASELINE_GREEN_SD\",16,true,10]"
        ",[\"BASELINE_RED_MEAN\",16,true,10]"
        ",[\"BASELINE_GREEN_MEAN\",16,true,10]"
        ",[\"PKMID_A\",16,true,10]"
        ",[\"PKMID_C\",16,true,10]"
        ",[\"PKMID_G\",16,true,10]"
        ",[\"PKMID_T\",16,true,10]"
        ",[\"ANGLE_A\",16,true,100]"
        ",[\"ANGLE_C\",16,true,100]"
        ",[\"ANGLE_G\",16,true,100]"
        ",[\"ANGLE_T\",16,true,100]"
        ",[\"PKMID_FRAMES_A\",16,false,1]"
        ",[\"PKMID_FRAMES_C\",16,false,1]"
        ",[\"PKMID_FRAMES_G\",16,false,1]"
        ",[\"PKMID_FRAMES_T\",16,false,1]"
        ",[\"NUM_PULSES\",16,false,1]"
        ",[\"NUM_FRAMES\",16,false,1]"
        ",[ \"NUM_BASES\",16,false,1]],\"FRAMES\" : "
        "16384},\"MF_METRIC\" : {\"FIELDS\" : [[ \"NUM_FRAMES\",16,false,1],[\"NUM_PULSES\",16,false,1]"
        ",[\"NUM_PULSES_A\",16,false,1]"
        ",[\"NUM_PULSES_C\",16,false,1]"
        ",[\"NUM_PULSES_G\",16,false,1]"
        ",[\"NUM_PULSES_T\",16,false,1]"
        ",[\"NUM_BASES\",16,false,1]"
        ",[\"PULSE_WIDTH\",16,false,1]"
        ",[\"NUM_SANDWICHES\",16,false,1]"
        ",[\"NUM_HALF_SANDWICHES\",16,false,1]"
        ",[\"NUM_PULSE_LABEL_STUTTERS\",16,false,1]"
        "],\"FRAMES\" : "
        "4096},"
        "\"HF_METRIC\":{\"FIELDS\":[[\"NUM_BASES\",16,false,1],[\"NUM_FRAMES\",16,false,1],[\"NUM_PULSES\",16,false,1]],\"FRAMES\":1024},"
        "\"PACKET\" : [[ \"READOUT\", 2 ],[ \"DEL_TAG\", 3 ],[ "
        "\"SUB_TAG\", 3 ],[ \"DEL_QV\", 4 ],[ \"SUB_QV\", 4 ],[ \"INS_QV\", 4 "
        "],[ \"MRG_QV\", 4 ],[ \"IPD_LL\", 8, 255, \"IPD16_LL\", 16 "
        "]],\"SLICE_LENGTH_FRAMES\" : 16384}}";

TEST(HQRFParams, UserInput)
{
    HQRFMethod crfHqrf;
    auto ppaAlgoConfig = std::make_shared<PpaAlgoConfig>();

    // Sequel header, will only work with sequel model:
    ppaAlgoConfig->SetPlatformDefaults(Platform::SEQUEL);
    ppaAlgoConfig->hqrf.method = HqrfPublicMethod::M1;
    EXPECT_NO_THROW(CoeffLookup(ppaAlgoConfig));
    ppaAlgoConfig->hqrf.method = HqrfPublicMethod::N1;
    EXPECT_NO_THROW(CoeffLookup(ppaAlgoConfig));
    ppaAlgoConfig->hqrf.method = HqrfPublicMethod::M2;
    EXPECT_NO_THROW(CoeffLookup(ppaAlgoConfig));
    ppaAlgoConfig->hqrf.method = HqrfPublicMethod::M3;
    EXPECT_NO_THROW(CoeffLookup(ppaAlgoConfig));
    ppaAlgoConfig->hqrf.method = HqrfPublicMethod::M4;
    EXPECT_NO_THROW(CoeffLookup(ppaAlgoConfig));
    ppaAlgoConfig->hqrf.method = HqrfPublicMethod::N2;
    EXPECT_NO_THROW(CoeffLookup(ppaAlgoConfig));

    ppaAlgoConfig->hqrf.method = HqrfPublicMethod::K1;
    crfHqrf = CoeffLookup(ppaAlgoConfig);
    EXPECT_EQ(crfHqrf, HQRFMethod::SEQUEL_CRF_HMM);

    // Spider header, will only work with spider models:
    ppaAlgoConfig->SetPlatformDefaults(Platform::SEQUELII);
    ppaAlgoConfig->hqrf.method = HqrfPublicMethod::K1;
    EXPECT_NO_THROW(CoeffLookup(ppaAlgoConfig));

    ppaAlgoConfig->hqrf.method = HqrfPublicMethod::M3;
    crfHqrf = CoeffLookup(ppaAlgoConfig);
    EXPECT_EQ(crfHqrf, HQRFMethod::TRAINED_CART_CART);

    ppaAlgoConfig->hqrf.method = HqrfPublicMethod::M1;
    crfHqrf = CoeffLookup(ppaAlgoConfig);
    EXPECT_EQ(crfHqrf, HQRFMethod::SPIDER_CRF_HMM);

    ppaAlgoConfig->hqrf.method = HqrfPublicMethod::N1;
    crfHqrf = CoeffLookup(ppaAlgoConfig);
    EXPECT_EQ(crfHqrf, HQRFMethod::ZOFFSET_CRF_HMM);

    ppaAlgoConfig->hqrf.method = HqrfPublicMethod::M4;
    crfHqrf = CoeffLookup(ppaAlgoConfig);
    EXPECT_EQ(crfHqrf, HQRFMethod::BAZ_HMM);

    ppaAlgoConfig->hqrf.method = HqrfPublicMethod::N2;
    crfHqrf = CoeffLookup(ppaAlgoConfig);
    EXPECT_EQ(crfHqrf, HQRFMethod::ZOFFSET_CART_HMM);
}

TEST(HQRFParams, evaluatePolynomial)
{
    EXPECT_NEAR(0.04f, evaluatePolynomial(std::vector<float>{0.04, 0.00}, 1.0f), 0.001);
    EXPECT_NEAR(0.08f, evaluatePolynomial(std::vector<float>{0.04, 0.00}, 2.0f), 0.001);

    EXPECT_NEAR(0.05f, evaluatePolynomial(std::vector<float>{0.04, 0.01}, 1.0f), 0.001);
    EXPECT_NEAR(0.09f, evaluatePolynomial(std::vector<float>{0.04, 0.01}, 2.0f), 0.001);

    EXPECT_NEAR(0.06f, evaluatePolynomial(std::vector<float>{0.01, 0.04, 0.01}, 1.0f), 0.001);
    EXPECT_NEAR(0.13f, evaluatePolynomial(std::vector<float>{0.01, 0.04, 0.01}, 2.0f), 0.001);

    EXPECT_NEAR(0.058f, evaluatePolynomial(std::vector<float>{0.008, -0.01, 0.06}, 1.0f), 0.001);
    EXPECT_NEAR(0.072f, evaluatePolynomial(std::vector<float>{0.008, -0.01, 0.06}, 2.0f), 0.001);
}
