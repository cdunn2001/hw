#include <iostream>
#include <utility>
#include <pacbio/ipc/JSON.h>
#include <pacbio/primary/SequelDefinitions.h>
#include <pacbio/primary/SequelMovieFileHDF5.h>
#include <pacbio/primary/SequelTraceFile.h>
#include <pacbio/primary/SequelMovieFactory.h>
#include <pacbio/process/ProcessBase.h>
#include <pacbio/text/String.h>
#include <pacbio/primary/ChipClass.h>
#include <pacbio/primary/ChipLayoutRTO3.h>
#include <boost/crc.hpp>
#include <tbb/pipeline.h>
#include <chrono>
#include <memory>

using namespace std;
using namespace PacBio::Primary;
using namespace PacBio::Process;

void GenerateCRC(const std::string& input, std::ostream& os)
{
#ifdef USE_TBB
    SequelMovieFileHDF5 a(input);

    int p = 0;
    int ntoken = 10; // ??
    uint32_t n = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

    tbb::parallel_pipeline
            (
                    ntoken,
                    tbb::make_filter<void, SequelMovieFrame<int16_t>*>(
                            tbb::filter::serial_in_order,
                            [&](tbb::flow_control& fc) -> SequelMovieFrame<int16_t>* {
                                if (n < a.NFRAMES)
                                {
                                    auto frame = new SequelMovieFrame<int16_t>(a.NROW, a.NCOL);
                                    a.ReadFrame(n, *frame);
                                    n++;
                                    return frame;
                                }
                                else
                                {
                                    fc.stop();
                                    return NULL;
                                }
                            })

                    & tbb::make_filter<SequelMovieFrame<int16_t>*, uint32_t>(
                            tbb::filter::parallel,
                            [](SequelMovieFrame<int16_t>* frame) -> uint32_t {
                                boost::crc_32_type crcEngine;
                                crcEngine.reset();
                                crcEngine.process_bytes(frame->data, frame->DataSize());
                                uint32_t crc = crcEngine.checksum();
                                delete frame;
                                return crc;
                            }
                    )

                    & tbb::make_filter<uint32_t, void>(
                            tbb::filter::serial_in_order,
                            [&](uint32_t result) {
                                os << result << "\n";
                                if (p++ % 1000 == 999)
                                {
                                    auto t = std::chrono::high_resolution_clock::now();
                                    auto dtt = std::chrono::duration_cast<std::chrono::nanoseconds>(t - t0).count();
                                    t0 = t;
                                    double rate = 1000.0 / dtt * 1e9;
                                    std::cerr << "CRC progress: frame " << p << " of " << a.NFRAMES << " (" <<
                                    (float) p * 100.0 / a.NFRAMES << "%) " << rate << " frames/sec" << std::endl;
                                }
                            })
            );

#else
    int p = 0;
            SequelMovieFileHDF5 a(argv[2]);
            boost::crc_32_type crcEngine;
            SequelMovieFrame<int16_t> aFrame(a.NROW,a.NCOL); // buffer used to load data
            auto t0 = std::chrono::high_resolution_clock::now();
            for (uint32_t n=0;n<a.NFRAMES;n++){
                a.ReadFrame(n,aFrame);

                crcEngine.reset();
                crcEngine.process_bytes(aFrame.data, aFrame.DataSize());
                uint32_t crc = crcEngine.checksum();
                std::cout << crc << "\n";
                // std::cout << aFrame.index << "\t" << crc << "\n";
                if (p++ % 1000 == 999)
                {
                    auto t = std::chrono::high_resolution_clock::now();
                    auto dtt = std::chrono::duration_cast<std::chrono::nanoseconds>(t - t0).count();
                    t0 = t;
                    double rate = 1000.0/dtt*1e9;
                    std::cerr << "CRC progress: frame " << p << " of " << a.NFRAMES <<  " (" << (float)p*100.0/a.NFRAMES << "%) " << rate << " frames/sec" << std::endl;
                }
            }
#endif
}

AnalogSet CreateSequelAnalogs2()
{
    const std::string sequelAnalogs = R"(
        [
            {"base":"T", "spectrumValues":[0.9,0.1], "relativeAmplitude": 0.8, "intraPulseXsnCV": 0.01,
             "pw2SlowStepRatio": 0.0, "ipd2SlowStepRatio":0.0},
            {"base":"G", "spectrumValues":[0.9,0.1], "relativeAmplitude": 0.5, "intraPulseXsnCV": 0.02,
             "pw2SlowStepRatio": 0.0, "ipd2SlowStepRatio":0.0},
            {"base":"C", "spectrumValues":[0.1,0.9], "relativeAmplitude": 1.0, "intraPulseXsnCV": 0.03,
             "pw2SlowStepRatio": 0.0, "ipd2SlowStepRatio":0.0},
            {"base":"A", "spectrumValues":[0.1,0.9], "relativeAmplitude": 0.6, "intraPulseXsnCV": 0.04,
             "pw2SlowStepRatio": 0.0, "ipd2SlowStepRatio":0.0}
        ]
    )";
    Json::Value analogsJson=PacBio::IPC::ParseJSON(sequelAnalogs);
    return ParseAnalogSet(analogsJson);
}

AnalogSet CreateSpiderAnalogs2()
{
    const std::string spiderAnalogs = R"(
        [
            {"base":"T", "spectrumValues":[1.0], "relativeAmplitude": 0.8, "intraPulseXsnCV": 0.01,
             "pw2SlowStepRatio": 0.0, "ipd2SlowStepRatio":0.0},
            {"base":"G", "spectrumValues":[1.0], "relativeAmplitude": 0.5, "intraPulseXsnCV": 0.02,
             "pw2SlowStepRatio": 0.0, "ipd2SlowStepRatio":0.0},
            {"base":"C", "spectrumValues":[1.0], "relativeAmplitude": 1.0, "intraPulseXsnCV": 0.03,
             "pw2SlowStepRatio": 0.0, "ipd2SlowStepRatio":0.0},
            {"base":"A", "spectrumValues":[1.0], "relativeAmplitude": 0.6, "intraPulseXsnCV": 0.04,
             "pw2SlowStepRatio": 0.0, "ipd2SlowStepRatio":0.0}
        ]
    )";
    Json::Value analogsJson=PacBio::IPC::ParseJSON(spiderAnalogs);
    return ParseAnalogSet(analogsJson);
}

using namespace PacBio::Primary;

const std::string version("0.3");
int main(int argc, char* argv[])
{

    OptionParser parser = ProcessBase::OptionParserFactory();
    parser.description("General purpose tool for reading and writing mov.h5 and trc.h5 files.");
    parser.version(version);

    auto groupMand = OptionGroup(parser, "Command option");
    groupMand.add_option("--crc").action_store_true().type_bool().set_default(false).help("Output file is CRC format.\n--crc <fileA>");
    groupMand.add_option("--diff").action_store_true().type_bool().set_default(false).help("Diff two files.\n--diff <fileA> <fileB>\n--diff <fileA>   # compares against test pattern");
    groupMand.add_option("--copy").action_store_true().type_bool().set_default(false).help("Copy movie to another, modifying trc.h5 to mov.h5 or number of frames.\n--copy <inputh5> <outputh5>");
    groupMand.add_option("--generate").action_store_true().type_bool().set_default(false).help("generate a movie.\n--generate <outputH5>");
    groupMand.add_option("--extractroi").action_store_true().type_bool().set_default(false).help("Write ROI to stdout.\n--extractroi <trcH5>");
    groupMand.add_option("--dumpsummary").action_store_true().type_bool().help("dump summary");
    groupMand.add_option("--dumpjson").action_store_true().type_bool().help("dump summary of file in JSON format");
    groupMand.add_option("--checksum").action_store_true().type_bool().help("generate checksum output compatible with baz2bam checksum");
    parser.add_option_group(groupMand);

    auto otherOptions = OptionGroup(parser, "Other options");
    otherOptions.add_option("--frames").type_int().set_default(128).help("Number of frames. default: %default");
    otherOptions.add_option("--rows").type_int().set_default(0).help("Number of rows. ");
    otherOptions.add_option("--cols").type_int().set_default(0).help("Number of cols. ");
    otherOptions.add_option("--dataonly").action_store_true().type_bool().set_default(false).help("Compare only trace/frame data");
    std::vector<std::string> patterns = {"testpattern","zeros","lotsapulses","alpha","beta","designerdark","designerdynamicloading"};
    otherOptions.add_option("--pattern").choices(patterns.begin(), patterns.end()).set_default("alpha").help("pattern to use.  default: %default");
    auto chipclasses = ChipClass::allValuesAsStrings();
    otherOptions.add_option("--chipclass").choices(chipclasses.begin(), chipclasses.end()).set_default(ChipClass(ChipClass::UNKNOWN).toString()).help("default: %default");
    otherOptions.add_option("--chiplayout").choices(ChipLayout::GetLayoutNames().begin(), ChipLayout::GetLayoutNames().end()).set_default(ChipLayout::GetLayoutNames()[0]).help("default: %default");
    otherOptions.add_option("--subset").action_store_true().type_bool().set_default(false).help("Check that ZMWS of <fileA> are a subset of the ZMWS of <file<B>");
#if 0
    parser.add_option("--dumptiles").set_default(0).help("dump N raw tiles as binary files");
    parser.add_option("--filt").action("store").type("string").help("Cross talk filter file to input coefficients");
#endif

    parser.add_option_group(otherOptions);
    Values& options = parser.parse_args(argc, argv);
    vector<string> args = parser.args();

    uint64_t nFrames = options.get("frames");
    ChipClass chipClass(options["chipclass"]);

    std::string layout = options["chiplayout"];
    std::unique_ptr<ChipLayout> chiplayout(ChipLayout::Factory(layout));
    if (!options.is_set_by_user("chipclass"))
    {
        chipClass = chiplayout->GetChipClass();
    }

    if (chiplayout->GetChipClass() != chipClass)
    {
        throw PBException("inconsistent --chipclass=" + chipClass.toString() + " and --chiplayout=" +
             chiplayout->Name() + " -> " + chiplayout->GetChipClass().toString() + " options");
    }
    if (!options.is_set_by_user("rows")) options["rows"] = std::to_string(chiplayout->GetSensorROI().PhysicalRows());
    if (!options.is_set_by_user("cols")) options["cols"] = std::to_string(chiplayout->GetSensorROI().PhysicalCols());

    ProcessBase::HandleGlobalOptions(options);

    // backwards compatibility. Previous versions of the tool supported:
    // sequel_movie_diff copy file1 file2
    if (args.size()>= 1 && (args[0] == "convert" || args[0] == "copy"))
    {
        args.erase(args.begin()); // remove first arg
        options["copy"] = true;
    }
    bool subsetFlag = options.get("subset");

    uint32_t commands = 0;
    commands += ((bool)options.get("crc")?1:0);
    commands += ((bool)options.get("diff")?1:0);
    commands += ((bool)options.get("copy")?1:0);
    commands += ((bool)options.get("generate")?1:0);
    commands += ((bool)options.get("extractroi")?1:0);
    commands += ((bool)options.get("dumpsummary")?1:0);
    commands += ((bool)options.get("dumpjson")?1:0);
    commands += ((bool)options.get("checksum")?1:0);
    if (commands ==0) {
        std::cerr << "Need to give at least one command (--copy, --diff, etc)" << std::endl;
        return 1;
    }
    if (commands > 1) {
        std::cerr << "Can only give one command (--copy, --diff, etc)" << std::endl;
        return 1;
    }
    if (options.get("crc"))
    {
        if (args.size() == 1)
        {
            GenerateCRC(args[0], std::cout);
        } else if (args.size() == 2)
        {
            std::ofstream f(args[1].c_str());
            GenerateCRC(args[0], f);
        }
        return 0;
    }
    else if (options.get("diff"))
    {
        std::unique_ptr<SequelMovieFileBase> a(std::move(SequelMovieFactory::CreateInput(args[0])));

        if (args.size() == 1)
        {
            if (a->Type() == SequelMovieFileBase::SequelMovieType::Frame)
            {
                return !! dynamic_cast<SequelMovieFileHDF5&>(*a).VerifyTest(100000,0.0);
            }
            else
            {
                throw PBException("not supported");
            }
        }

        else if (args.size() == 2)
        {
            std::unique_ptr<SequelMovieFileBase> b(std::move(SequelMovieFactory::CreateInput(args[1])));

            if (a->Type() == SequelMovieFileBase::SequelMovieType::Frame &&
                b->Type() == SequelMovieFileBase::SequelMovieType::Frame)
            {
                return !! dynamic_cast<SequelMovieFileHDF5&>(*a).CompareFrames(
                          dynamic_cast<SequelMovieFileHDF5&>(*b), options.get("dataonly")
                );
            }
            else if  (a->Type() == SequelMovieFileBase::SequelMovieType::Trace &&
                  b->Type() == SequelMovieFileBase::SequelMovieType::Trace)
            {
                return !! dynamic_cast<SequelTraceFileHDF5&>(*a).CompareTraces(
                          dynamic_cast<SequelTraceFileHDF5&>(*b), subsetFlag);
            }
            else
            {
                throw PBException("not supported");
            }
        }
    }
    else if (options.get("generate"))
    {
        if (args.size() != 1) throw PBException("need output filename");
        string filename = args[0];
        if (PacBio::Text::String::EndsWith(filename,".trc.h5"))
        {
            const SequelSensorROI sroi = chiplayout->GetSensorROI();

            SequelRectangularROI roi(RowPixels(0),
                                     ColPixels(0),
                                     RowPixels(options.get("rows")),
                                     ColPixels(options.get("cols")),
                                     sroi
                                     );

            SequelMovieConfig mc;
            mc.path = args[0];
            mc.roi.reset(roi.Clone());
            mc.chipClass = chipClass;
            mc.numFrames = nFrames;
            SequelTraceFileHDF5 a(mc);

            AnalogSet analogs;
            int numChannels = 2;
            if (chipClass == ChipClass::Sequel)
            {
                analogs = CreateSequelAnalogs2();
                numChannels = 2;
            }
            else if (chipClass == ChipClass::Spider) {
                analogs = CreateSpiderAnalogs2();
                numChannels = 1;
            }
            else throw PBException("dont know how to make analogs");

            a.SetAnalogs(analogs,*chiplayout);

            a.SoftwareVersion << version;
            a.AcquisitionXML << "";
            a.RI_InstrumentName << "sim";
            a.RI_PlatformId << SequelTraceFileHDF5::PlatformId_SequelAlpha4;
            a.RI_PlatformName << "SequelAlpha";
            a.LayoutName << chiplayout->Name();


//            traceWriter->DetermineAntiZMWs(*setup.fileROI.get(), cl);

            boost::multi_array<float, 3> imagePsf(boost::extents[numChannels][5][5]);
            boost::multi_array<double, 2> xtalkCorrection(boost::extents[7][7]);
            // set up the PSF multi_array
            for (int j = 0; j < numChannels; j++)
            {
                imagePsf[j][2][2] = 1.0;
            }

            xtalkCorrection[3][3] = 1.0;

            a.FilterMap << chiplayout->FilterMap();
            a.ImagePsf << imagePsf;
            a.XtalkCorrection << xtalkCorrection;


            // ref spectrum and SNR
            std::vector<float> analogRefSpectrum(numChannels);
            std::copy(analogs[0].dyeSpectrum.begin(),analogs[0].dyeSpectrum.end(), analogRefSpectrum.begin() );
            float analogRefSnr = 12.0;
            a.AnalogRefSpectrum << analogRefSpectrum;
            a.AnalogRefSnr << analogRefSnr;



            uint32_t numZmws = roi.CountZMWs();

            if (options["pattern"] == "testpattern")
            {
                if (chipClass == ChipClass::Spider) throw PBException("fix me");

                std::vector<std::pair<int16_t,int16_t> > traceData(nFrames);
                for (uint64_t itrace = 0; itrace < numZmws; itrace++)
                {
                    for (unsigned int iframe = 0; iframe < nFrames; iframe++)
                    {
                        int16_t grn = itrace * 2 % 2048; //(itrace * 10 + iframe * 2 + 0) % 2048;
                        int16_t red = itrace * 2 % 2048; // (itrace * 10 + iframe * 2 + 1) % 2048;
                        if (iframe == itrace)
                        {
                            grn = itrace;
                        }
                        traceData[iframe].first = grn;
                        traceData[iframe].second = red;
                    }
                    a.Write2CTrace(itrace, 0, traceData); // fixme, Sequel only !
#if 1
                    if (nFrames >= 16384)
                    {
                        int16_t sum = 0;
                        for (int iframe = 0; iframe < 16384; iframe++)
                        {
                            sum += traceData[iframe].first;
                            sum += traceData[iframe].second;
                        }
                        cout << itrace << " " << sum << std::endl;
                    }

#endif
                }
            }
            else if (options["pattern"] == "designerdark")
            {
                if (chipClass == ChipClass::Sequel) throw PBException("fix me");
                std::vector<int16_t> traceData(nFrames);
                for (uint64_t itrace = 0; itrace < numZmws; itrace++)
                {
                    for (unsigned int iframe = 0; iframe < nFrames; iframe++)
                    {
                        int16_t v;
                        if (iframe < 100 || iframe > nFrames - 100)
                            v = 100;
                        else
                            v = 300;

                        traceData[iframe] = v;
                    }
                    a.Write1CTrace(itrace, 0, traceData); // fixme, Spider only !
                    cout << "\rTrace:" << itrace << "\r" << std::flush;
                }
            }
            else
            {
                throw PBException("pattern not supported " + options["pattern"]);
            }
            cout << "write " << nFrames << " to " << a.FullFilePath() << endl;
        }
        else if (PacBio::Text::String::EndsWith(filename,".h5"))
        {
            auto pixelsPerZmw = PixelsPerZmw(chipClass);
            const uint32_t rowPixelsPerZmw = pixelsPerZmw.first;
            const uint32_t colPixelsPerZmw = pixelsPerZmw.second;

            SequelSensorROI sroi(0,
                                 0,
                                 options.get("rows"),
                                 options.get("cols"),
                                 rowPixelsPerZmw,
                                 colPixelsPerZmw);
            SequelRectangularROI roi(RowPixels(0),
                                     ColPixels(0),
                                     RowPixels(options.get("rows")),
                                     ColPixels(options.get("cols")),
                                     sroi);

            SequelMovieConfig mc;
            mc.path = args[0];
            mc.roi.reset(roi.Clone());
            mc.chipClass = chipClass;

            SequelMovieFileHDF5 a(mc);

            if (options["pattern"] == "alpha")
            {
                a.CreateTest(nFrames, 0, 12);
            }
            else if (options["pattern"] == "beta")
            {
                a.CreateTestBeta(nFrames, 0, 12);
            }
            else
            {
                for (uint64_t iframe = 0; iframe < nFrames; ++iframe)
                {
                    SequelMovieFrame<int16_t> frame(roi);
                    if (options["pattern"] == "testpattern")
                    {
                        frame.CreateRandomPatternForSequel(11);
                    }
                    else if (options["pattern"] == "zeros")
                    {
                        frame.SetDefaultValue(0);
                    }
                    else if (options["pattern"] == "lotsapulses")
                    {
                        frame.SetDefaultValue(static_cast<int16_t>((iframe % 2) ? 2047 : 0));
                    }
                    else if (options["pattern"] == "designerdark")
                    {
                        int16_t v;
                        if (iframe < 100 || iframe > nFrames - 100)
                            v= 100;
                        else
                            v = 300;
                        frame.SetDefaultValue(v);
                    }
                    else if (options["pattern"] == "designerdynamicloading")
                    {
                        throw PBException("pattern not supported yet");
                    }

                    else
                    {
                        throw PBException("pattern not supported " + options["pattern"]);
                    }
                    a.AddFrame(frame);
                    cout << "\r" << iframe << "\r" << std::flush;
                }
            }
            cout << "write " << nFrames << " to " << a.FullFilePath() << endl;
        }
    }
    else if (options.get("copy"))
    {
        if (args.size() != 2)
        {
            std::cerr << "wrong number of args. Type --help." << std::endl;
            exit(1);
        }
        std::string inputFileName = args[0];
        std::string outputFileName = args[1];

        auto srcFile = SequelMovieFactory::CreateInput(inputFileName);
        ChipClass chipClassSrc = srcFile->GetChipClass();
        std::unique_ptr<SequelRectangularROI> roi;
        switch (chipClassSrc)
        {
            case ChipClass::Sequel:
                roi.reset(new SequelRectangularROI(RowPixels(0), ColPixels(0),
                                                   RowPixels(options.is_set_by_user("rows")
                                                             ? options.get("rows")
                                                             : PacBio::Primary::Sequel::maxPixelRows),
                                                   ColPixels(options.is_set_by_user("cols")
                                                             ? options.get("cols")
                                                             : PacBio::Primary::Sequel::maxPixelCols),
                                                   SequelSensorROI::SequelAlpha()));
                break;
            case ChipClass::Spider:
                roi.reset(new SequelRectangularROI(RowPixels(0), ColPixels(0),
                                                   RowPixels(options.is_set_by_user("rows")
                                                             ? options.get("rows")
                                                             :  PacBio::Primary::Spider::maxPixelRows),
                                                   ColPixels(options.is_set_by_user("cols")
                                                             ? options.get("cols")
                                                             : PacBio::Primary::Spider::maxPixelCols),
                                                   SequelSensorROI::Spider()));
                break;
            default:
                std::cerr << "Unknown chip class!" << std::endl;
                exit(1);
                break;
        }

        SequelMovieConfig mc;
        mc.path = outputFileName;
        mc.roi.reset(roi->Clone());
        mc.chipClass = chipClassSrc;
        mc.numFrames = nFrames;
        auto dstFile = SequelMovieFactory::CreateOutput(mc);

        cout << "copying from " << inputFileName << " " << *(srcFile->GetROI()) << std::endl;
        cout << "         to  " << outputFileName << " " << *roi << endl;
        cout << "copying header" << endl;

        if (dstFile->Type()== SequelMovieFileBase::SequelMovieType::Frame)
        {
            auto destMovieFile = dynamic_cast<SequelMovieFileHDF5*>(dstFile.get());
            destMovieFile->CopyHeader(srcFile.get());
        }

        SequelMovieFrame<int16_t> f(*roi);
        for (uint64_t i = 0; i < nFrames; i++)
        {
            cout << "frame " << i << std::endl;
            f.index = i;
            srcFile->ReadFrame(i, f);
            dstFile->AddFrame(f);
        }
    }
    else if (options.get("extractroi"))
    {
        string inputFileName = args[0];
        auto srcFile = SequelMovieFactory::CreateInput(inputFileName);
        const SequelROI* roi = srcFile->GetROI();
        cout << roi->GetJson() << std::endl;
    }
    else if (options.get("dumpsummary"))
    {
        SequelMovieFileHDF5::SetMovieDebugStream(std::cout);
        std::unique_ptr<SequelMovieFileBase> a(std::move(SequelMovieFactory::CreateInput(args[0])));
        a->DumpSummary(std::cout);
    }
    else if (options.get("dumpjson"))
    {
        SequelMovieFileHDF5::SetMovieDebugStream(std::cout);
        std::unique_ptr<SequelMovieFileBase> a(std::move(SequelMovieFactory::CreateInput(args[0])));
        Json::Value jv = a->DumpJson();
        std::cout << jv << std::endl;
    }
    else if (options.get("checksum"))
    {
        SequelMovieFileHDF5::SetMovieDebugStream(std::cout);
        SequelTraceFileHDF5 traceFile(args[0]);
        std::vector<uint32_t> holeNumbers;
        traceFile.HoleNumber >> holeNumbers;
        uint64_t numFrames;
        traceFile.NumFrames >> numFrames; // fix me, this should be a member of the class

        std::cout << "NFRAMES:" << numFrames << " NUM_CHANNELS:" << traceFile.NUM_CHANNELS << std::endl;
        uint32_t dumps = 0;
        for(uint32_t zmwOffset=0;zmwOffset < traceFile.NUM_HOLES; zmwOffset++)
        {
            const uint64_t frameOffset = 0;
            if (traceFile.NUM_CHANNELS == 1)
            {
                std::vector<int16_t> traceData;
                traceFile.Read1CTrace(zmwOffset, frameOffset, traceData);
                int16_t checkSum = 0;
                int16_t blockChecksum = 0;
                uint32_t frameCount = 0;
                uint64_t number = holeNumbers[zmwOffset];
                for(const int16_t sample : traceData)
                {
                    checkSum += sample;
                    blockChecksum += sample;
                    if ((frameCount % 1024) == 1023)
                    {
                        if (dumps < 100)
                        {
                            std::cout << "zmw:" << number << " block:" << (frameCount / 1024) << " blockChecksum:" << blockChecksum << std::endl;
                            dumps++;
                        }
                        blockChecksum = 0;
                    }
                    frameCount++;
                }

                std::cout << zmwOffset << " " << number << " " << checkSum << "\n";
            }
            else if (traceFile.NUM_CHANNELS == 2)
            {
                std::vector<std::pair<int16_t,int16_t>> traceData;
                traceFile.Read2CTrace(zmwOffset, frameOffset, traceData);
                int16_t checkSum = 0;
                int16_t blockChecksum = 0;
                uint32_t frameCount = 0;
                uint64_t number = holeNumbers[zmwOffset];
                for(const auto samplePair : traceData)
                {
                    int16_t cs = 0;
                    cs += samplePair.first;
                    cs += samplePair.second;
                    checkSum += cs;
                    blockChecksum += cs;
                    if ((frameCount % 1024) == 1023)
                    {
                        if (dumps < 100)
                        {
                            std::cout << "zmw:" << number << " block:" << (frameCount / 1024) << " blockChecksum:" << blockChecksum << std::endl;
                            dumps++;
                        }
                        blockChecksum = 0;
                    }
                    frameCount++;
                }

                std::cout << zmwOffset << " " << number << " " << checkSum << "\n";
            }
            else
            {
                throw PBException("only NUM_CHANNELS == 1 is supported, file has " + std::to_string(traceFile.NUM_CHANNELS));
            }
        }
    }
    else
    {
        std::cerr << "Internal error. Invalid command selected." << std::endl;
        return 1;
    }
    return 0;
}
