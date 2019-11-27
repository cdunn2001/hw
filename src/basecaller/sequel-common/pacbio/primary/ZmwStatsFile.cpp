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

#include <string>
#include <type_traits> // std::remove_reference
#include <pacbio/primary/ZmwStatsFile.h>

namespace PacBio {
namespace Primary {

class DataSpace : public H5::DataSpace
{
public:
    DataSpace(hsize_t dim1)
            : H5::DataSpace(1, &dim1)
    {}

    DataSpace(hsize_t dim1, hsize_t dim2)
            : H5::DataSpace()
    {
        hsize_t dim[] = {dim1, dim2};
        this->setExtentSimple(2, dim);
    };

    DataSpace(hsize_t dim1, hsize_t dim2, hsize_t dim3)
            : H5::DataSpace()
    {
        hsize_t dim[] = {dim1, dim2, dim3};
        this->setExtentSimple(3, dim);
    };
};

template <typename T>
struct H5DataTypeTraits;

template<>
struct H5DataTypeTraits<uint8_t>
{
    static H5::PredType getH5Type() { return H5::PredType::STD_U8LE; }
};

template<>
struct H5DataTypeTraits<uint16_t>
{
    static H5::PredType getH5Type() { return H5::PredType::STD_U16LE; }
};

template<>
struct H5DataTypeTraits<uint32_t>
{
    static H5::PredType getH5Type() { return H5::PredType::STD_U32LE; }
};

template<>
struct H5DataTypeTraits<int32_t>
{
    static H5::PredType getH5Type() { return H5::PredType::STD_I32LE; }
};

template<>
struct H5DataTypeTraits<float>
{
    static H5::PredType getH5Type() { return H5::PredType::IEEE_F32LE; }
};

template<>
struct H5DataTypeTraits<std::string>
{
    static H5::StrType getH5Type() { return H5::StrType(H5::PredType::C_S1, H5T_VARIABLE); }
};

template<typename T>
void CopyAttribute(H5::Group& srcGroup, H5::Group& destGroup, const std::string& attrName)
{
    H5::DataSpace scalar;
    H5::Attribute destAttribute = destGroup.createAttribute(attrName, H5DataTypeTraits<T>::getH5Type(), scalar);
    const H5::Attribute srcAttribute = srcGroup.openAttribute(attrName);
    T var;
    srcAttribute >> var;
    destAttribute << var;
}

template<typename T>
void CopyDataset(H5::Group& srcGroup, H5::Group& destGroup, const std::string& dataSetName)
{
    auto srcDataSet = srcGroup.openDataSet(dataSetName);
    H5::DataSpace srcDs = srcDataSet.getSpace();
    auto destDataSet = destGroup.createDataSet(dataSetName, H5DataTypeTraits<T>::getH5Type(), srcDs);
    std::vector<hsize_t> dims(srcDs.getSimpleExtentNdims());
    srcDs.getSimpleExtentDims(dims.data());
    size_t s = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());
    std::vector<T> data(s);
    srcDataSet.read(data.data(), H5DataTypeTraits<T>::getH5Type());
    destDataSet.write(data.data(), H5DataTypeTraits<T>::getH5Type());
}

hsize_t ZmwStatDataSet::unused = 0;
unsigned int ZmwStatDataSet::ZLibCompressionLevel = 3;

static thread_local ZmwStatsOption options_ = ZmwStatsOption::NONE;
bool ZmwStatsQuiet() { return (options_ & ZmwStatsOption::QUIET) == ZmwStatsOption::QUIET; }
void ZmwStatsSetQuiet() {  options_ = ZmwStatsOption::QUIET; }
void ZmwStatsSetVerbose() {  options_ = ZmwStatsOption::NONE; }


ZmwStatsFile::ZmwStatsFile(const std::string& filename) :
        filename_(filename)
{
    OpenForRead();
}

ZmwStatsFile::ZmwStatsFile(const std::string& filename, uint32_t numHoles, uint32_t numAnalogs, uint32_t numFilters,
                           uint32_t numTimeslices, uint32_t binSize) :
        filename_(filename)
{
    if (numHoles == 0 || numAnalogs == 0 || numFilters == 0 || numTimeslices == 0 || binSize == 0)
    {
        throw PBException("ZmwStatsFile constructor called with a 0 dimension");
    }

    NewFileConfig config;
    config.numHoles = numHoles;
    config.numAnalogs = numAnalogs;
    config.numFilters = numFilters;
    config.numFrames = numTimeslices * binSize;
    config.binSize = binSize;
    OpenForWrite(config, true);
    workerThread_ = std::move(std::async(std::launch::async, [this](){ return this->WriterThread(); }));
}

ZmwStatsFile::ZmwStatsFile(const std::string& filename, const NewFileConfig& config, bool dummyScanData)
        :
        filename_(filename)
{
    if (config.numHoles == 0    || config.numAnalogs == 0 ||
        config.numFilters == 0 || config.numFrames == 0  ||
        config.binSize == 0 || config.mfBinSize == 0)
    {
        throw PBException("ZmwStatsFile constructor called with a 0 dimension");
    }

    OpenForWrite(config, dummyScanData);
    workerThread_ = std::move(std::async(std::launch::async, [this](){ return this->WriterThread(); }));
}

ZmwStatsFile::~ZmwStatsFile()
{
    if (workerThread_.valid())
    {
#ifdef OLDWAY
        workQueue_.Push(nullptr);
#else
        workVectorQueue_.Push(nullptr);
#endif
        workerThread_.wait();
        try
        {
            (void) workerThread_.get();

        } catch (const std::exception& ex)
        {
            PBLOG_ERROR << ex.what();
        }
    }
    std::lock_guard<std::mutex> _(mutex_);
}


void ZmwStatsFile::CopyScanData(H5::Group& destScanData) const
{
    auto srcScanData = objectManager_.topGroup_.ReadOnlyGroup().openGroup("ScanData");


    CopyAttribute<std::string>(srcScanData, destScanData, "FormatVersion");

    //
    // /ScanData/AcqParams
    //
    auto srcAcqParams = srcScanData.openGroup("AcqParams");
    auto destAcqParams = destScanData.createGroup("AcqParams");
    CopyAttribute<int32_t>(srcAcqParams, destAcqParams, "LaserOnFrame");
    CopyAttribute<uint8_t>(srcAcqParams, destAcqParams, "LaserOnFrameValid");
    CopyAttribute<int32_t>(srcAcqParams, destAcqParams, "HotStartFrame");
    CopyAttribute<uint8_t>(srcAcqParams, destAcqParams, "HotStartFrameValid");
    CopyAttribute<int32_t>(srcAcqParams, destAcqParams, "CameraType");
    CopyAttribute<float>(srcAcqParams, destAcqParams, "CameraGain");
    CopyAttribute<float>(srcAcqParams, destAcqParams, "AduGain");
    CopyAttribute<float>(srcAcqParams, destAcqParams, "CameraBias");
    CopyAttribute<float>(srcAcqParams, destAcqParams, "CameraBiasStd");
    CopyAttribute<float>(srcAcqParams, destAcqParams, "FrameRate");
    CopyAttribute<uint32_t>(srcAcqParams, destAcqParams, "NumFrames");

    //
    // /ScanData/AcquisitionXML
    //
    H5::DataSpace scalar;
    auto srcAcquisitionXML = srcScanData.openDataSet("AcquisitionXML");
    auto destAcquisitionXML = destScanData.createDataSet("AcquisitionXML", SeqH5string(), scalar);
    std::string acquisitionXML;
    srcAcquisitionXML >> acquisitionXML;
    destAcquisitionXML << acquisitionXML;

    //
    // /ScanData/ChipInfo
    //
    auto srcChipInfo = srcScanData.openGroup("ChipInfo");
    auto destChipInfo = destScanData.createGroup("ChipInfo");
    CopyAttribute<std::string>(srcChipInfo, destChipInfo, "LayoutName");
    CopyDataset<uint16_t>(srcChipInfo, destChipInfo, "FilterMap");
    CopyDataset<float>(srcChipInfo, destChipInfo, "ImagePsf");
    CopyDataset<float>(srcChipInfo, destChipInfo, "XtalkCorrection");
    CopyDataset<float>(srcChipInfo, destChipInfo, "AnalogRefSpectrum");
    CopyDataset<float>(srcChipInfo, destChipInfo, "AnalogRefSnr");

    //
    // /ScanData/DyeSet
    //
    auto srcDyeSet = srcScanData.openGroup("DyeSet");
    auto destDyeSet = destScanData.createGroup("DyeSet");
    CopyAttribute<uint16_t>(srcDyeSet, destDyeSet, "NumAnalog");
    CopyAttribute<std::string>(srcDyeSet, destDyeSet, "BaseMap");
    CopyDataset<float>(srcDyeSet, destDyeSet, "AnalogSpectra");
    CopyDataset<float>(srcDyeSet, destDyeSet, "RelativeAmp");
    CopyDataset<float>(srcDyeSet, destDyeSet, "ExcessNoiseCV");
    CopyDataset<float>(srcDyeSet, destDyeSet, "PulseWidthMean");
    CopyDataset<float>(srcDyeSet, destDyeSet, "IpdMean");
    CopyDataset<float>(srcDyeSet, destDyeSet, "Pw2SlowStepRatio");
    CopyDataset<float>(srcDyeSet, destDyeSet, "Ipd2SlowStepRatio");

    //
    // /ScanData/RunInfo
    //
    auto srcRunInfo = srcScanData.openGroup("RunInfo");
    auto destRunInfo = destScanData.createGroup("RunInfo");
    CopyAttribute<uint32_t>(srcRunInfo, destRunInfo, "PlatformId");
    CopyAttribute<std::string>(srcRunInfo, destRunInfo, "PlatformName");
    CopyAttribute<std::string>(srcRunInfo, destRunInfo, "InstrumentName");
    CopyAttribute<std::string>(srcRunInfo, destRunInfo, "MovieName");
    CopyAttribute<std::string>(srcRunInfo, destRunInfo, "MoviePath");
    CopyAttribute<std::string>(srcRunInfo, destRunInfo, "SequencingChemistry");
    CopyAttribute<std::string>(srcRunInfo, destRunInfo, "SequencingKit");
    CopyAttribute<std::string>(srcRunInfo, destRunInfo, "BindingKit");
    CopyAttribute<std::string>(srcRunInfo, destRunInfo, "Control");
    CopyAttribute<uint8_t>(srcRunInfo, destRunInfo, "IsControlUsed");
}

void ZmwStatsFile::WriteScanData(const Json::Value& experimentMetadata)
{
    std::lock_guard<std::mutex> _(mutex_);

    //
    // Most of the below was cobbled together from:
    // SequelTraceFileHDF5::CreateFileForWrite()
    // SequelTraceFileHDF5::SetAnalogs()
    //

    // NOTE: The commented out data is not filled in the experimental data JSON
    // These need to be filled our properly in Acquisition_Setup.cpp.

    auto ScanData = objectManager_.topGroup_.Group().createGroup("ScanData");
    H5::DataSpace scalar;

    auto FormatVersion = ScanData.createAttribute("FormatVersion", SeqH5string(), scalar);
    //FormatVersion << experimentMetadata["FormatVersion"].asString();

    //
    // /ScanData/AcqParams
    //
    auto AcqParams          = ScanData.createGroup("AcqParams");

    auto LaserOnFrame       = AcqParams.createAttribute("LaserOnFrame", H5::PredType::STD_I32LE, scalar);
    auto LaserOnFrameValid  = AcqParams.createAttribute("LaserOnFrameValid", uint8(), scalar);
    auto HotStartFrame      = AcqParams.createAttribute("HotStartFrame", H5::PredType::STD_I32LE, scalar);
    auto HotStartFrameValid = AcqParams.createAttribute("HotStartFrameValid", uint8(), scalar);
    auto CameraType         = AcqParams.createAttribute("CameraType", H5::PredType::STD_I32LE, scalar);
    auto CameraGain         = AcqParams.createAttribute("CameraGain", float32(), scalar);
    auto AduGain            = AcqParams.createAttribute("AduGain", float32(), scalar);
    auto CameraBias         = AcqParams.createAttribute("CameraBias", float32(), scalar);
    auto CameraBiasStd      = AcqParams.createAttribute("CameraBiasStd", float32(), scalar);
    auto FrameRate          = AcqParams.createAttribute("FrameRate", float32(), scalar);
    auto NumFrames          = AcqParams.createAttribute("NumFrames", uint32(), scalar);

    //LaserOnFrame            << experimentMetadata["AcqParams"]["LaserOnFrame"].asInt();
    //LaserOnFrameValid       << static_cast<uint8_t>(experimentMetadata["AcqParams"]["LaserOnFrameValid"].asUInt());
    //HotStartFrame           << experimentMetadata["AcqParams"]["HotStartFrame"].asInt();
    //HotStartFrameValid      << static_cast<uint8_t>(experimentMetadata["AcqParams"]["HotStartFrameValid"].asUInt());
    //CameraType              << experimentMetadata["AcqParams"]["CameraType"].asInt();
    //CameraGain              << experimentMetadata["AcqParams"]["CameraGain"].asFloat();
    AduGain                 << experimentMetadata["AcqParams"]["AduGain"].asFloat();
    //CameraBias              << experimentMetadata["AcqParams"]["CameraBias"].asFloat();
    //CameraBiasStd           << experimentMetadata["AcqParams"]["CameraBiasStd"].asFloat();
    //FrameRate               << experimentMetadata["AcqParams"]["FrameRate"].asFloat();
    //NumFrames               << experimentMetadata["AcqParams"]["NumFrames"].asUInt();

    //
    // /ScanData/AcquisitionXML
    //
    auto AcquisitionXML   = ScanData.createDataSet("AcquisitionXML", SeqH5string(), scalar);
    //AcquisitionXML << experimentMetadata["AcquisitionXML"].asString();

    //
    // /Scan/ChipInfo
    //
    auto ChipInfo           = ScanData.createGroup("ChipInfo");

    hsize_t imagePsfNRows = experimentMetadata["ChipInfo"]["ImagePsf"][0].isArray()
            ? experimentMetadata["ChipInfo"]["ImagePsf"][0].size()
            : 0;
    hsize_t imagePsfNCols = (imagePsfNRows != 0)
            ? experimentMetadata["ChipInfo"]["ImagePsf"][0][0].size()
            : 0;
    hsize_t xtalkNRows = experimentMetadata["ChipInfo"]["CrosstalkFilter"].isArray()
            ? experimentMetadata["ChipInfo"]["CrosstalkFilter"].size()
            : 0;
    hsize_t xtalkNCols = (xtalkNRows != 0)
            ? experimentMetadata["ChipInfo"]["CrosstalkFilter"][0].size()
            : 0;

    auto LayoutName         = ChipInfo.createAttribute ("LayoutName", SeqH5string(), scalar);
    auto FilterMap          = ChipInfo.createDataSet("FilterMap", uint16(), DataSpace(nF_));
    auto ImagePsf           = ChipInfo.createDataSet("ImagePsf", float32(), DataSpace(nF_, imagePsfNRows, imagePsfNCols));
    auto XtalkCorrection    = ChipInfo.createDataSet("XtalkCorrection", float32(), DataSpace(xtalkNRows, xtalkNCols));
    auto AnalogRefSpectrum  = ChipInfo.createDataSet("AnalogRefSpectrum", float32(), DataSpace(nF_));
    auto AnalogRefSnr       = ChipInfo.createDataSet("AnalogRefSnr", float32(), scalar);

    LayoutName << experimentMetadata["ChipInfo"]["LayoutName"].asString();

    std::vector<uint16_t> filterMap(nF_);
    boost::multi_array<float,3> imagePsf(boost::extents[nF_][imagePsfNRows][imagePsfNCols]);
    boost::multi_array<float,2> xtalkCorrection(boost::extents[xtalkNRows][xtalkNCols]);
    std::vector<float> analogRefSpectrum(nF_);

    for (uint32_t ch = 0; ch < nF_; ch++)
    {
        filterMap[ch] = static_cast<uint16_t>(experimentMetadata["ChipInfo"]["FilterMap"][ch].asUInt());
        analogRefSpectrum[ch] = experimentMetadata["ChipInfo"]["AnalogRefSpectrum"][ch].asFloat();
        for (uint32_t r = 0; r < imagePsfNRows; r++)
        {
            for (uint32_t c = 0; c < imagePsfNCols; c++)
            {
                imagePsf[ch][r][c] = experimentMetadata["ChipInfo"]["ImagePsf"][ch][r][c].asFloat();
            }
        }
    }

    for (uint32_t r = 0; r < xtalkNRows; r++)
    {
        for (uint32_t c = 0; c < xtalkNCols; c++)
        {
            xtalkCorrection[r][c] = experimentMetadata["ChipInfo"]["XtalkCorrection"][r][c].asFloat();
        }
    }

    FilterMap           << filterMap;
    ImagePsf            << imagePsf;
    XtalkCorrection     << xtalkCorrection;
    AnalogRefSpectrum   << analogRefSpectrum;
    AnalogRefSnr        << experimentMetadata["ChipInfo"]["AnalogRefSnr"].asFloat();

    //
    // /ScanData/DyeSet
    //
    auto DyeSet             = ScanData.createGroup("DyeSet");

    auto NumAnalog          = DyeSet.createAttribute("NumAnalog", uint16(), scalar);
    auto BaseMap            = DyeSet.createAttribute("BaseMap", SeqH5string(), scalar);
    auto AnalogSpectra      = DyeSet.createDataSet("AnalogSpectra", float32(), DataSpace(nA_, nF_));
    auto RelativeAmp        = DyeSet.createDataSet("RelativeAmp", float32(), DataSpace(nA_));
    auto ExcessNoiseCV      = DyeSet.createDataSet("ExcessNoiseCV", float32(), DataSpace(nA_));
    auto PulseWidthMean     = DyeSet.createDataSet("PulseWidthMean", float32(), DataSpace(nA_));
    auto IpdMean            = DyeSet.createDataSet("IpdMean", float32(), DataSpace(nA_));
    auto Pw2SlowStepRatio   = DyeSet.createDataSet("Pw2SlowStepRatio" , float32(), DataSpace(nA_));
    auto Ipd2SlowStepRatio  = DyeSet.createDataSet("Ipd2SlowStepRatio", float32(), DataSpace(nA_));

    NumAnalog << static_cast<uint16_t>(nA_);
    //BaseMap << experimentMetadata["DyeSet"]["BaseMap"].asString();
    boost::multi_array<float,2> analogSpectra(boost::extents[nA_][nF_]);
    std::vector<float> relativeAmp(nA_);
    std::vector<float> excessNoiseCV(nA_);
    std::vector<float> pulseWidth(nA_);
    std::vector<float> ipd(nA_);
    std::vector<float> pw2SlowStepRatio(nA_);
    std::vector<float> ipd2SlowStepRatio(nA_);
    for (uint32_t a = 0; a < nA_; a++)
    {
        for (uint32_t c = 0; c < nF_; c++)
        {
            analogSpectra[a][c] = experimentMetadata["DyeSet"]["AnalogSpectra"][a][c].asFloat();
        }
        relativeAmp[a] = experimentMetadata["DyeSet"]["RelativeAmp"][a].asFloat();
        excessNoiseCV[a] = experimentMetadata["DyeSet"]["ExcessNoiseCV"][a].asFloat();
        pulseWidth[a] = experimentMetadata["DyeSet"]["PulseWidthMean"][a].asFloat();
        ipd[a] = experimentMetadata["DyeSet"]["IpdMean"][a].asFloat();
        pw2SlowStepRatio[a] = experimentMetadata["DyeSet"]["Pw2SlowStepRatio"][a].asFloat();
        ipd2SlowStepRatio[a] = experimentMetadata["DyeSet"]["Ipd2SlowStepRatio"][a].asFloat();
    }
    AnalogSpectra       << analogSpectra;
    RelativeAmp         << relativeAmp;
    ExcessNoiseCV       << excessNoiseCV;
    PulseWidthMean      << pulseWidth;
    IpdMean             << ipd;
    Pw2SlowStepRatio    << pw2SlowStepRatio;
    Ipd2SlowStepRatio   << ipd2SlowStepRatio;

    //
    // /ScanData/RunInfo
    //
    auto RunInfo                = ScanData.createGroup("RunInfo");

    auto RI_PlatformId          = RunInfo.createAttribute("PlatformId", uint32(), scalar);
    auto RI_PlatformName        = RunInfo.createAttribute("PlatformName", SeqH5string(), scalar);
    auto RI_InstrumentName      = RunInfo.createAttribute("InstrumentName", SeqH5string(), scalar);
    auto MovieName              = RunInfo.createAttribute("MovieName", SeqH5string(), scalar);
    auto MoviePath              = RunInfo.createAttribute("MoviePath", SeqH5string(), scalar);
    auto SequencingChemistry    = RunInfo.createAttribute("SequencingChemistry", SeqH5string(), scalar);
    auto SequencingKit          = RunInfo.createAttribute("SequencingKit", SeqH5string(), scalar);
    auto BindingKit             = RunInfo.createAttribute("BindingKit", SeqH5string(), scalar);
    auto Control                = RunInfo.createAttribute("Control", SeqH5string(), scalar);
    auto IsControlUsed          = RunInfo.createAttribute("IsControlUsed", uint8(), scalar);

    //RI_PlatformId         << experimentMetadata["RunInfo"]["PlatformId"].asUInt();
    //RI_PlatformName       << experimentMetadata["RunInfo"]["PlatformName"].asString();
    //RI_InstrumentName     << experimentMetadata["RunInfo"]["InstrumentName"].asString();
    //MovieName             << experimentMetadata["RunInfo"]["MovieName"].asString();
    //MoviePath             << experimentMetadata["RunInfo"]["MoviePath"].asString();
    //SequencingChemistry   << experimentMetadata["RunInfo"]["SequencingChemistry"].asString();
    //SequencingKit         << experimentMetadata["RunInfo"]["SequencingKit"].asString();
    //BindingKit            << experimentMetadata["RunInfo"]["BindingKit"].asString();
    //Control               << experimentMetadata["RunInfo"]["Control"].asString();
    //IsControlUsed         << static_cast<uint8_t>(experimentMetadata["RunInfo"]["IsControlUsed"].asUInt());
}

static uint32_t GetDesiredChunkSize()
{
    if (getenv("CHUNKSIZE"))
    {
        return atoi(getenv("CHUNKSIZE"));
    } else {
        return 65536;
    }
}
const uint32_t DesiredChunkSize = GetDesiredChunkSize();

/// create the underlying H5 object corresponding to this
void ZmwStatDataSet::Create(H5::Group& group, uint32_t binSize)
{
    const char* n = Name();
    H5::DataType dt = GetDataType();
    H5::DataSpace ds = GetDataSpace();
    std::vector<hsize_t> dims(ds.getSimpleExtentNdims());
    std::vector<hsize_t> maxdims(ds.getSimpleExtentNdims());
    ds.getSimpleExtentDims(dims.data(),maxdims.data());

    const hsize_t desiredChunkSize = DesiredChunkSize;
    if (dims[0] >= desiredChunkSize)
    {
        H5::DSetCreatPropList propList;
        propList.setDeflate(ZLibCompressionLevel);
        std::vector<hsize_t> chunk_dims = dims; // (ds.getSimpleExtentNdims());
        chunk_dims[0] = desiredChunkSize;
        propList.setChunk(chunk_dims.size(), chunk_dims.data());
        int fill_val = 0;
        propList.setFillValue(H5::PredType::NATIVE_INT, &fill_val);
        dataSet_ = group.createDataSet(n, dt, ds, propList);
    }
    else
    {
        dataSet_ = group.createDataSet(n, dt, ds);
    }

    CreateAttribute("Description",SeqH5string(),description_);
    CreateAttribute("HQRegion",uint8(),hqRegion_);
    CreateAttribute("UnitsOrEncoding",SeqH5string(),units_);
    CreateAttribute("BinSize",uint32(),binSize);
}

/// and all values are concatenated with '/' delimeters.
std::string ZmwStatDataSet::GetValue(hsize_t index) const
{
    std::vector<hsize_t> foffset(dimensions_,0);
    foffset[0] = index;
    std::vector<hsize_t> fsize(dimensions_);
    fsize[0] = 1;

    size_t valueDims = 1;
    if (dimensions_ >= 2) {
        valueDims *= dim1_;
        foffset[1] = 0;
        fsize[1] = dim1_;
    }
    if (dimensions_ >= 3) {
        valueDims *= dim2_;
        foffset[2] = 0;
        fsize[2] = dim2_;
    }


    H5::DataSpace memspace(dimensions_, fsize.data());
    H5::DataSpace filespace = dataSet_.getSpace();
    filespace.selectHyperslab(H5S_SELECT_SET, fsize.data(), foffset.data());

    if (dataType_.detectClass(H5T_class_t::H5T_INTEGER))
    {
        std::vector<uint32_t> value(valueDims);
        dataSet_.read(value.data(), GetType<uint32_t>(), memspace, filespace);
        return PacBio::Text::String::Join(value.begin(),value.end(),'/');
    }
    else if (dataType_.detectClass(H5T_class_t::H5T_FLOAT))
    {
        std::vector<double> value(valueDims);
        dataSet_.read(value.data(), GetType<double>(), memspace, filespace);
        return PacBio::Text::String::Join(value.begin(),value.end(),'/');
    }
    else
    {
        throw PBException("data type not supported " );
    }
}

std::string ZmwStatDataSet::GetValue(hsize_t index, hsize_t t) const
{
    if (dimensions_ <2) throw PBException("dimensions<2 for " + std::string(name_));

    std::vector<hsize_t> foffset(dimensions_,0);
    foffset[0] = index;
    foffset[1] = t;
    std::vector<hsize_t> fsize(dimensions_);
    fsize[0] = 1;
    fsize[1] = 1;

    size_t valueDims = 1;
    if (dimensions_ >= 3) {
        valueDims *= dim2_;
        foffset[2] = 0;
        fsize[2] = dim2_;
    }

    H5::DataSpace memspace(dimensions_, fsize.data());
    H5::DataSpace filespace = dataSet_.getSpace();
    filespace.selectHyperslab(H5S_SELECT_SET, fsize.data(), foffset.data());

    if (dataType_.detectClass(H5T_class_t::H5T_INTEGER))
    {
        std::vector<uint32_t> value(valueDims);
        dataSet_.read(value.data(), GetType<uint32_t>(), memspace, filespace);
        return PacBio::Text::String::Join(value.begin(),value.end(),'/');
    }
    else if (dataType_.detectClass(H5T_class_t::H5T_FLOAT))
    {
        std::vector<double> value(valueDims);
        dataSet_.read(value.data(), GetType<double>(), memspace, filespace);
        return PacBio::Text::String::Join(value.begin(),value.end(),'/');
    }
    else
    {
        throw PBException("data type not supported " );
    }
}

void ZmwStatsFile::OpenForRead()
{
    std::lock_guard<std::mutex> _(mutex_);

    objectManager_.OpenAll(filename_);

    nH_ = NumBases.GetDims().at(0);
    nA_ = BaseFraction.GetDims().at(1);
    nF_ = BaselineLevel.GetDims().at(1);

    try
    {
        auto acqParamsGrp = objectManager_.topGroup_.ReadOnlyGroup().openGroup("ScanData").openGroup("AcqParams");
        if (acqParamsGrp.attrExists("NumFrames"))
        {
            uint32_t numFrames;
            const H5::Attribute srcAttribute = acqParamsGrp.openAttribute("NumFrames");
            srcAttribute >> numFrames;
            numFrames_ = numFrames;
        }
    }
    catch(const H5::GroupIException& )
    {
        PBLOG_WARN << "Defaulting frames_ = 0";
        numFrames_ = 0;
    }

    nAB_ = nA_ + 1;
    nFtriangle_ = nF_ + nF_ * (nF_ + 1) / 2;

    if (nH_ != GetDims(BaselineLevel.DataSet()).at(0)) throw PBException("BaselineLevel weird");
    if (nF_ != GetDims(BaselineLevel.DataSet()).at(1)) throw PBException("BaselineLevel weird");

    if (nH_ != GetDims(BaseFraction.DataSet()).at(0)) throw PBException("BaseFraction weird");
    if (nA_ != GetDims(BaseFraction.DataSet()).at(1)) throw PBException("BaseFraction weird");

    ///if (nH_ != GetDims(ClusterDistance.DataSet()).at(0)) throw PBException("ClusterDistance weird");
    ///if (nAB_ != GetDims(ClusterDistance.DataSet()).at(1)) throw PBException("ClusterDistance weird");
    ///if (nFtriangle_ != GetDims(ClusterDistance.DataSet()).at(2)) throw PBException("ClusterDistance weird");
}


void ZmwStatsFile::OpenForWrite( const NewFileConfig& config, bool dummyScanData)
{
    std::lock_guard<std::mutex> _(mutex_);

    nH_ = config.numHoles;
    nA_ = config.numAnalogs;
    nF_ = config.numFilters;
    nMF_ = config.NumMFBlocks();
    binSize_ = config.binSize;
    numFrames_ = config.numFrames;
    uint32_t lastMFBinFrames = config.LastMFBinFrames();
    addDiagnostics_ = config.addDiagnostics;

    nAB_ = nA_ + 1;
    nFtriangle_ = nF_ + nF_ * (nF_ + 1) / 2;

    objectManager_.CreateAll(filename_, config.binSize, addDiagnostics_);

    if (addDiagnostics_)
    {
        VsMF.CreateAttribute("LastBinFrames",uint32(),lastMFBinFrames);
    }

    if (dummyScanData)
    {
        //
        // /Scan/ChipInfo/LayoutName
        //
        auto ScanData = objectManager_.topGroup_.Group().createGroup("ScanData");
        H5::DataSpace scalar;
        auto ChipInfo = ScanData.createGroup("ChipInfo");
        auto LayoutName = ChipInfo.createAttribute("LayoutName", SeqH5string(), scalar);
        LayoutName << "SequEL_4.0_RTO3";
    }

    //ClusterDistance.CreateAttribute("BaseOrder",SeqH5string(),"BACGT");
}

ZmwStats ZmwStatsFile::GetZmwStatsTemplate() const
{
    ZmwStats zmw(nA(), nF(), nMF());
    return zmw;
}

void ZmwStatsFile::Set(hsize_t index, const ZmwStats& zmw)
{
    if (index >= nH_) throw PBException("Index out of range:" + std::to_string(index));

    std::lock_guard<std::mutex> _(mutex_);

//    std::cout << "Set(" << index << " )" << std::endl;

#define DECLARE_SEQUELH5_SUPPORT()
#define DECLARE_ZMWSTAT_START_GROUP(name)
#define DECLARE_ZMWSTATDATASET_1D(prefix, name, type, units, description, hqr, dim0)             prefix##name.Set(index,zmw.prefix##name);
#define DECLARE_ZMWSTATDATASET_2D(prefix, name, type, units, description, hqr, dim0, dim1)       prefix##name.Set(index,zmw.prefix##name[0]);
#define DECLARE_ZMWSTATDATASET_3D(prefix, name, type, units, description, hqr, dim0, dim1, dim2) prefix##name.Set(index,zmw.prefix##name[0][0]);
#define DECLARE_ZMWSTAT_END_GROUP(name)
#define DECLARE_ZMWSTAT_ENUM(...)

#define DECLARE_ZMWSTAT_START_GROUP_DIAGNOSTICS(name)
#define DECLARE_ZMWSTATDATASET_1D_DIAGNOSTICS(prefix, name, type, units, description, hqr, dim0)             prefix##name.Set(index,zmw.prefix##name, addDiagnostics_);
#define DECLARE_ZMWSTATDATASET_2D_DIAGNOSTICS(prefix, name, type, units, description, hqr, dim0, dim1)       prefix##name.Set(index,zmw.prefix##name[0], addDiagnostics_);
#define DECLARE_ZMWSTATDATASET_3D_DIAGNOSTICS(prefix, name, type, units, description, hqr, dim0, dim1, dim2) prefix##name.Set(index,zmw.prefix##name[0][0], addDiagnostics_);
#define DECLARE_ZMWSTAT_END_GROUP_DIAGNOSTICS(name)

#include "ZmwStatsFileDefinition.h"

#undef DECLARE_ZMWSTATDATASET_1D
#undef DECLARE_ZMWSTATDATASET_2D
#undef DECLARE_ZMWSTATDATASET_3D
#undef DECLARE_ZMWSTATDATASET_1D_DIAGNOSTICS
#undef DECLARE_ZMWSTATDATASET_2D_DIAGNOSTICS
#undef DECLARE_ZMWSTATDATASET_3D_DIAGNOSTICS

//    if ((index % 1000) == 0) objectManager_.Flush();

}

std::unique_ptr<ZmwStats> ZmwStatsFile::GetZmwStatsBuffer()
{
    return std::unique_ptr<ZmwStats>(new ZmwStats(nA(), nF(), nMF()));
}

#ifdef OLDWAY
void ZmwStatsFile::SetBuffer(hsize_t index, std::unique_ptr<ZmwStats>&& zmw )
{
    if (index >= nH_) throw PBException("Index out of range:" + std::to_string(index));

    zmw->index_ = index;
    workQueue_.Push(std::move(zmw));
}
#else
void ZmwStatsFile::WriteBuffers(std::unique_ptr<std::vector<ZmwStats> >&& bufferList)
{
    workVectorQueue_.Push(std::move(bufferList));
}
#endif

ZmwStats ZmwStatsFile::Get(uint32_t index) const
{
    if (index >= nH_) throw PBException("Index Out of range:" + std::to_string(index));

    ZmwStats zmw(nA(),nF(),nMF());
    zmw.Init();

    std::lock_guard<std::mutex> _(mutex_);

#define float32_t float
#define DECLARE_ZMWSTATDATASET_1D(prefix,name,type,units,description,hqr, dim0)           prefix##name.Get<type##_t>(index,&zmw.prefix##name);
#define DECLARE_ZMWSTATDATASET_2D(prefix,name,type,units,description,hqr, dim0,dim1)      prefix##name.Get<type##_t>(index,&zmw.prefix##name[0]);
#define DECLARE_ZMWSTATDATASET_3D(prefix,name,type,units,description,hqr, dim0,dim1,dim2) prefix##name.Get<type##_t>(index,&zmw.prefix##name[0][0]);
#define DECLARE_ZMWSTATDATASET_1D_DIAGNOSTICS(prefix,name,type,units,description,hqr, dim0)           prefix##name.Get<type##_t>(index,&zmw.prefix##name, addDiagnostics_);
#define DECLARE_ZMWSTATDATASET_2D_DIAGNOSTICS(prefix,name,type,units,description,hqr, dim0,dim1)      prefix##name.Get<type##_t>(index,&zmw.prefix##name[0], addDiagnostics_);
#define DECLARE_ZMWSTATDATASET_3D_DIAGNOSTICS(prefix,name,type,units,description,hqr, dim0,dim1,dim2) prefix##name.Get<type##_t>(index,&zmw.prefix##name[0][0], addDiagnostics_);

#include "ZmwStatsFileDefinition.h"

#undef float32_t
#undef DECLARE_SEQUELH5_SUPPORT
#undef DECLARE_ZMWSTAT_START_GROUP
#undef DECLARE_ZMWSTATDATASET_1D
#undef DECLARE_ZMWSTATDATASET_2D
#undef DECLARE_ZMWSTATDATASET_3D
#undef DECLARE_ZMWSTAT_END_GROUP
#undef DECLARE_ZMWSTAT_ENUM

#undef DECLARE_ZMWSTAT_START_GROUP_DIAGNOSTICS
#undef DECLARE_ZMWSTATDATASET_1D_DIAGNOSTICS
#undef DECLARE_ZMWSTATDATASET_2D_DIAGNOSTICS
#undef DECLARE_ZMWSTATDATASET_3D_DIAGNOSTICS
#undef DECLARE_ZMWSTAT_END_GROUP_DIAGNOSTICS

    return zmw;
}

int ZmwStatsFile::WriterThread()
{
    size_t loops = 0;
    while(true)
    {
        loops++;
#ifdef OLDWAY
        std::unique_ptr<ZmwStats> zmwStats( workQueue_.Pop() );

        if (!zmwStats) break;

        std::lock_guard<std::mutex> _(mutex_);

#define DECLARE_SEQUELH5_SUPPORT()
#define DECLARE_ZMWSTAT_START_GROUP(name)
#define DECLARE_ZMWSTATDATASET_1D(prefix,name,type,units,description,hqr, dim0)           prefix##name.Set(zmwStats->index_,zmwStats->prefix##name);
#define DECLARE_ZMWSTATDATASET_2D(prefix,name,type,units,description,hqr, dim0,dim1)      prefix##name.Set(zmwStats->index_,zmwStats->prefix##name[0]);
#define DECLARE_ZMWSTATDATASET_3D(prefix,name,type,units,description,hqr, dim0,dim1,dim2) prefix##name.Set(zmwStats->index_,zmwStats->prefix##name[0][0]);
#define DECLARE_ZMWSTAT_END_GROUP(name)
#define DECLARE_ZMWSTAT_ENUM(...)

#include "ZmwStatsFileDefinition.h"

#undef DECLARE_ZMWSTATDATASET_1D
#undef DECLARE_ZMWSTATDATASET_2D
#undef DECLARE_ZMWSTATDATASET_3D
#else
        std::unique_ptr<std::vector<ZmwStats>> bufferList( workVectorQueue_.Pop() );

        if (!bufferList) break;

        std::lock_guard<std::mutex> _(mutex_);

//        NumBases.SetVector<decltype(bufferList->at(0).NumBases)>(*bufferList, [](const ZmwStats&z)->const decltype(z.NumBases)*{return &z.NumBases;});

#define DECLARE_SEQUELH5_SUPPORT()
#define DECLARE_ZMWSTAT_START_GROUP(name)
#define float32_t float
#define DECLARE_ZMWSTATDATASET_1D(prefix,name,type,units,description,hqr, dim0)           prefix##name.SetVector<type##_t>(*bufferList, [](const ZmwStats&z){return &z.prefix##name;});
#define DECLARE_ZMWSTATDATASET_2D(prefix,name,type,units,description,hqr, dim0,dim1)      prefix##name.SetVector<type##_t>(*bufferList, [](const ZmwStats&z){return &z.prefix##name[0];});
#define DECLARE_ZMWSTATDATASET_3D(prefix,name,type,units,description,hqr, dim0,dim1,dim2) prefix##name.SetVector<type##_t>(*bufferList, [](const ZmwStats&z){return &z.prefix##name[0][0];});
#define DECLARE_ZMWSTAT_END_GROUP(name)
#define DECLARE_ZMWSTAT_ENUM(...)

#define DECLARE_ZMWSTAT_START_GROUP_DIAGNOSTICS(name)
#define DECLARE_ZMWSTATDATASET_1D_DIAGNOSTICS(prefix,name,type,units,description,hqr, dim0)           prefix##name.SetVector<type##_t>(*bufferList, [](const ZmwStats&z){return &z.prefix##name;}, addDiagnostics_);
#define DECLARE_ZMWSTATDATASET_2D_DIAGNOSTICS(prefix,name,type,units,description,hqr, dim0,dim1)      prefix##name.SetVector<type##_t>(*bufferList, [](const ZmwStats&z){return &z.prefix##name[0];}, addDiagnostics_);
#define DECLARE_ZMWSTATDATASET_3D_DIAGNOSTICS(prefix,name,type,units,description,hqr, dim0,dim1,dim2) prefix##name.SetVector<type##_t>(*bufferList, [](const ZmwStats&z){return &z.prefix##name[0][0];}, addDiagnostics_);
#define DECLARE_ZMWSTAT_END_GROUP_DIAGNOSTICS(name)

#include "ZmwStatsFileDefinition.h"

#undef float32_t
#undef DECLARE_ZMWSTATDATASET_1D
#undef DECLARE_ZMWSTATDATASET_2D
#undef DECLARE_ZMWSTATDATASET_3D
#undef DECLARE_ZMWSTATDATASET_1D_DIAGNOSTICS
#undef DECLARE_ZMWSTATDATASET_2D_DIAGNOSTICS
#undef DECLARE_ZMWSTATDATASET_3D_DIAGNOSTICS

#endif
    }

    PBLOG_DEBUG << "ZmwStatsFile::WriterThread Loops:" << loops;

    return 0;
}


std::pair<uint32_t,uint32_t> ZmwStatsFile::GetCoordinate(int index) const
{
    if (coordinates_.size() == 0)
    {
        std::vector<uint16_t> holexy;
        HoleXY.DataSet() >> holexy;
        coordinates_.resize(holexy.size()/2);
        uint32_t i;
        uint32_t j;
        for(i=0,j=0; i< nH(); i++)
        {
            coordinates_[i].first  = holexy[j++];
            coordinates_[i].second = holexy[j++];
        }
    }
    return coordinates_.at(index);
}

}}
