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
// A wrapper class for dealing with rsts.h5 (Reduced Stats) files.
// See http://smrtanalysis-docs/primary/ppa/doc/perzmwstats.html
// for complete documentation.

#include <cstdint>
#include <string>
#include <float.h>
#include <pacbio/primary/ZmwReducedStatsFile.h>
#include <pacbio/primary/ZmwStatsFile.h>
#include <pacbio/text/String.h>
#include <pacbio/image/Netpbm.h>


namespace PacBio {
namespace Primary {

static const char* topGroup = "ReducedZMWMetrics";

ZmwReducedDataSet::ZmwReducedDataSet()
{

}



void ZmwReducedDataSet::Copy(const ZmwStatDataSet& dataset)
{
    CopyAttribute<std::string>(dataset,*this,"Description");
    CopyAttribute<uint16_t>(dataset,*this,"HQRegion");
    CopyAttribute<std::string>(dataset,*this,"UnitsOrEncoding");
    CopyAttribute<uint32_t>(dataset,*this,"BinSize");
}


void ZmwReducedStatsFile::Reduce(const std::string& name, ZmwReducedDataSet& zrds, const ZmwStatDataSet& ds, Reducer::Binning& binning,
                                 Reducer::Algorithm& algo, Reducer::Filter& filter)
{
    PBLOG_DEBUG << "Reduce";
    zrds.Copy(ds);

    zrds.CreateAttribute<uint32_t>("BinRows", uint32(), binning.BinRows());
    zrds.CreateAttribute<uint32_t>("BinCols", uint32(), binning.BinCols());
    zrds.CreateAttribute<uint32_t>("UnitCellOffsetX", uint32(), binning.UnitCellOffsetX());
    zrds.CreateAttribute<uint32_t>("UnitCellOffsetY", uint32(), binning.UnitCellOffsetY());
    zrds.CreateAttribute<std::string>("Filter", SeqH5string(), filter.Name());
    zrds.CreateAttribute<std::string>("Algorithm", SeqH5string(), algo.FullName());

    auto dims = ds.GetDims();
    // throw away first dimension
    dims.erase(dims.begin());
    uint32_t sliceSize = 1;
    for(auto d : dims)
    {

        sliceSize *= d;
    }

    std::vector<double> destImage(binning.NumOutputCount() * sliceSize);

    std::vector<double> binData(binning.NumBinValues());
    std::vector<uint8_t> binFilter(binning.NumBinValues());

    std::vector<double> sourceZmws;
    ds.DataSet() >> sourceZmws;

    if (imageOption_)
    {
        PacBio::Image::Netpbm::WritePGM(std::string("all_") + name + "_" + "all" + "_source.pgm", sourceZmws, sourceZmws.size(), 1);
    }

    for(uint32_t slice = 0; slice < sliceSize; slice++)
    {
        PBLOG_DEBUG << "Slice  " << slice;

        std::vector<double> sourceImage(binning.NumInputCount() , Reducer::NaN() );
        std::vector<uint8_t> filterImage(binning.NumInputCount() , 0 ); // 1= selected, 0=not selected

        // convert the linear list of ZMWs into a source image. ZMWs that are not
        // present will have NaN.
        for (uint32_t iH = 0; iH < currentZmwStatsFile_->nH(); iH++)
        {
            auto coord = GetCoordinate(iH);
            int32_t x = coord.first - binning.UnitCellOffsetX();
            int32_t y = coord.second - binning.UnitCellOffsetY();

            if (x >= 0 && x < (int32_t) binning.UnitCellRows() &&
                y >= 0 && y < (int32_t) binning.UnitCellCols())
            {
                uint32_t offset = x * binning.UnitCellCols() + y;
                sourceImage[offset] = sourceZmws[iH*sliceSize + slice];
                filterImage[offset] = filter.IsSelected(iH);
            }
        }
        if (imageOption_)
        {
            PacBio::Image::Netpbm::WritePGM(std::string("") + name + "_" + std::to_string(slice) + "_source.pgm", sourceImage,
                     binning.UnitCellRows(), binning.UnitCellCols());
            PacBio::Image::Netpbm::WritePGM(std::string("") + name + "_" + std::to_string(slice) + "_filter.pgm", filterImage,
                     binning.UnitCellRows(), binning.UnitCellCols());
        }
        int i =0;
        for (uint32_t outputRow = 0; outputRow < binning.NumOutputRows(); outputRow++)
        {
            uint32_t inputRow = outputRow * binning.BinRows();
            for (uint32_t outputCol = 0; outputCol < binning.NumOutputCols(); outputCol++)
            {
                uint32_t inputCol = outputCol * binning.BinCols();
                int j = 0;

                // grab up the bin data
                for (uint32_t binRow = 0; binRow < binning.BinRows(); binRow++)
                {
                    for (uint32_t binCol = 0; binCol < binning.BinCols(); binCol++)
                    {
                        if (inputRow + binRow >= binning.UnitCellRows()
                            || inputCol + binCol >= binning.UnitCellCols())
                        {
                            binData[j] = Reducer::NaN();
                            binFilter[j] = 0;
                        }
                        else
                        {
                            int k = (inputRow + binRow) * binning.UnitCellCols() + (inputCol + binCol);
                            binData[j]   = sourceImage[k];
                            binFilter[j] = filterImage[k];
                        }
                        j++;
                    }
                }

                // summarize the bin data
                double value = algo.Apply(binData, binFilter);

                destImage[(i++)*sliceSize + slice] = value;
            }
        }
    }
    if (imageOption_)
    {
        if (sliceSize == 1)
        {
            PacBio::Image::Netpbm::WritePGM(std::string("") + name + "_dest.pgm", destImage, binning.NumOutputRows(), binning.NumOutputCols());
        }
    }

    zrds.DataSet() << destImage;
}


ZmwReducedStatsFile::ZmwReducedStatsFile(const std::string& filename)
{
    hdf5file_.openFile(filename.c_str(), H5F_ACC_RDONLY);
}

ZmwReducedStatsFile::ZmwReducedStatsFile(const std::string& filename, const ReducedStatsConfig& config)
{
    Create(filename,config);
}

ZmwReducedStatsFile::~ZmwReducedStatsFile()
{
    try
    {
        Close();
    }
    catch(const std::exception& ex)
    {
        PBLOG_ERROR << "ZmwReducedStatsFile::~ZmwReducedStatsFile caught exception:" << ex.what();
    }
}


void ZmwReducedStatsFile::Create(const std::string& filename, const ReducedStatsConfig& /*config*/)
{
    if (filename == "")
    {
        throw PBException("Empty filename passed to ZmwReducesStatsFile");
    }

    try
    {
        hdf5file_ = H5::H5File(filename.c_str(), H5F_ACC_TRUNC);
        auto top = hdf5file_.createGroup(topGroup);
        scanData_ = top.createGroup("ScanData");
    }
    catch(const H5::Exception&)
    {
        PBLOG_ERROR << "H5 Exception during creation of " << filename;
        throw;
    }
}

std::pair<uint32_t,uint32_t> ZmwReducedStatsFile::GetCoordinate(int index) const
{
    return currentZmwStatsFile_->GetCoordinate(index);
}




void ZmwReducedStatsFile::Reduce(const ZmwStatsFile& inputFile, const ReducedStatsConfig& config)
{
    PBLOG_DEBUG << config;

    currentZmwStatsFile_ = &inputFile;
    boost::multi_array<uint16_t, 2> holeXY;
    inputFile.HoleXY.DataSet() >> holeXY;
    Reducer::Binning::Sizes sizes;
    sizes.minX = holeXY[0][0];
    sizes.minY = holeXY[0][1];
    sizes.maxX = holeXY[0][0];
    sizes.maxY = holeXY[0][1];
    for (size_t i = 0; i < holeXY.shape()[0]; ++i)
    {
        sizes.minX = std::min(sizes.minX, holeXY[i][0]);
        sizes.maxX = std::max(sizes.maxX, holeXY[i][0]);
        sizes.minY = std::min(sizes.minY, holeXY[i][1]);
        sizes.maxY = std::max(sizes.maxY, holeXY[i][1]);
    }

    PBLOG_INFO << "Input stats file dimensions:" << sizes.minX << " " << sizes.minY << " " << sizes.maxX << " " << sizes.maxY;

    for(const auto& output : config.Outputs)
    {
        if (output.Input() == "")
        {
            throw PBException("No input data set given in Outputs. Use the \"Input\" field with full HDF5 path to sts.h5 dataset.");
        }
        PBLOG_DEBUG << "Reducing: " << output.Input();

        Reducer::Algorithm algo = Reducer::Algorithm(output.Algorithm);
        Reducer::Filter filter = Reducer::Filter::Factory(output.Filter);

        filter.Load(inputFile);

        ZmwStatDataSet inputDataSet;
        try
        {
            inputDataSet = inputFile.GetDataSet(output.Input());
        }
        catch(const H5::Exception& ex)
        {
            throw PBException("can't open dataset " + output.Input() + " H5 Exception:" + ex.getDetailMsg());
        }
        boost::filesystem::path p(output.Input());
        std::string o = std::string("/") + topGroup + "/" + filter.Name() + "/" + algo.FullName() + "/"  + p.filename().string();
        PBLOG_DEBUG<< "Output: " << o << " type:" << output.Type();

        H5::DataType dt = GetDataType(output.Type());

        ReducedStatsConfig outputConfig{};
        outputConfig.Copy(config);
        outputConfig.Load(output.Json()); // this is a trick to overwrite the parent with the child values
        Reducer::Binning binning(outputConfig, sizes);

        // the first dimension from the source dataset is always nH (Num Holes)
        // we're going to replace that single dimension with 2 dimensions and turn it into an image.
        //* nH (ZMW dimension) will be replaced by nRROW x nRCOL dimensions, where nRROW is the number of reduced rows and nRCOL is the number of reduced columns
        //* nA (Number Analogs) will remain the same
        //* nC (Number of Channels) will remain the same
        //* nT (Number of time slices) will be replaced by nRT, the number of reduced time slices

        auto dimspair = PBDataSpace::GetDimensions(inputDataSet.DataSet().getSpace());
        auto dims = dimspair.first;
        dims.insert(dims.begin(), 0);
        dims[0] = binning.NumOutputRows();
        dims[1] = binning.NumOutputCols();
        H5::DataSpace newds = PBDataSpace::Create(dims);

        // Compress entire dataset.
        H5::DSetCreatPropList propList;
        propList.setDeflate(3);
        propList.setChunk(dims.size(), dims.data());

        ZmwReducedDataSet zrds = CreateDataSet(o, dt, newds, propList);

        std::string name = p.filename().string();
        Reduce(name, zrds, inputDataSet, binning, algo, filter);
    }

    currentZmwStatsFile_ = nullptr;
}


void ZmwReducedStatsFile::Close()
{
    hdf5file_.close();
}

ZmwReducedDataSet ZmwReducedStatsFile::CreateDataSet(const std::string& dataSetPath,
                                                     H5::DataType& dt, H5::DataSpace& ds,
                                                     H5::DSetCreatPropList& propList)
{

    boost::filesystem::path p = dataSetPath;
    p = p.parent_path();
    boost::filesystem::path q = "";

    PBFile::MakePath(hdf5file_, p.string());

    ZmwReducedDataSet zrds;

    zrds.dataSet_ = hdf5file_.createDataSet(dataSetPath, dt, ds, propList);

    return zrds;
}

ZmwReducedDataSet ZmwReducedStatsFile::GetDataSet(const std::string& dataSetPath)
{
    ZmwReducedDataSet zrds;

    try
    {
        zrds.dataSet_ = hdf5file_.openDataSet(dataSetPath.c_str());
    }
    catch(const H5::Exception& /*ex*/)
    {
        PBLOG_ERROR << "Exception trying to open data " << dataSetPath ;
        throw;
    }

    return zrds;
}

namespace Reducer {

const double nan = std::numeric_limits<double>::quiet_NaN();

double NaN() { return nan; }

template<typename T>
std::ostream& operator<<(std::ostream& s, const std::vector<T>& v)
{
    for(auto x : v)
    {
        s << x << " ";
    }
    return s;
}

Algorithm::Algorithm(Algorithm_t a) : Algorithm(a.toString()) {}

Algorithm::Algorithm(const std::string& s)
{
   fullName_ = s;

    std::vector<std::string> ss = PacBio::Text::String::Split(s,'=');
    if (ss.size() == 1)
    {
        ss.push_back("");
    }
    if (ss.size() != 2)
    {
        throw PBException("Invalid algorithm: " + s);
    }

    type_ = Algorithm_t::fromString(ss[0]);
    std::string option = ss[1];

    switch(type_)
    {
    case Algorithm_t::Count:
    {
        int value = std::stoi(option);

        function_ = [value](const double data[], const uint8_t filter[], int points) {
            double count = 0;
            for (int i = 0; i < points; i++)
            {
                if (filter[i] && data[i] == value)
                { count++; }
            }
            return count;
        };
    }
        break;

    case Algorithm_t::MAD:
        throw PBException("mad Not supported");

    case Algorithm_t::Mean:
        function_ = [](const double data[], const uint8_t filter[], int points) {
            double sum = 0;
            uint32_t count =0;
            for(int i=0; i < points ; i++)
            {
                if (filter[i] && !std::isnan(data[i])) { sum += data[i]; count ++; }
            }
            return sum/count; // allow NaN
        };
        break;

    case Algorithm_t::Median:
        function_ = [](const double data[], const uint8_t filter[], int points) {

            double median;
            std::vector<double> v;
            for(int i=0;i<points;i++)
            {
                if (filter[i] && !std::isnan(data[i])) v.push_back(data[i]);
            }
            if (v.size() == 0) return nan;
            if (v.size() & 1)
            {
                // odd (true median element)
                std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
                median = v[v.size() / 2];
            }
            else
            {
                // even number of members (no true median element)
                int offset =  v.size() / 2 -1;
                std::nth_element(v.begin(), v.begin() + offset, v.end());
                double m0 = v[offset];
                std::nth_element(v.begin(), v.begin() + offset + 1, v.end());
                double m1 = v[offset+1];
                median = (m0 + m1) * 0.5;
            }
            return median;
        };
        break;

    case Algorithm_t::Max:
        function_ = [](const double data[], const uint8_t filter[], int points) {
            double value = -DBL_MAX; // DBL_MIN is smallest positive number!
            for(int i=0; i < points ; i++)
            {
                if (filter[i] && !std::isnan(data[i])) value = std::max(data[i],value);
            }
            return (value != -DBL_MAX) ? value : nan;
        };
        break;

    case Algorithm_t::Min:
        function_ = [](const double data[], const uint8_t filter[], int points) {
            double value = DBL_MAX;
            for(int i=0; i < points ; i++)
            {
                if (filter[i] && !std::isnan(data[i])) value = std::min(data[i],value);
            }
            return (value != DBL_MAX) ? value : nan;
        };
        break;

    case Algorithm_t::None:
        function_ = [](const double */*data*/, const uint8_t */*filter*/, int /*points*/) {
            return NAN;
        };
        break;

    case Algorithm_t::Stddev:
        throw PBException("stddev Not supported");

    case Algorithm_t::Subsample:

        function_ = [](const double data[], const uint8_t */*filter*/, int points) {
            if (points == 0) return nan;
            return data[0];
        };
        break;

    case Algorithm_t::Sum:

        function_ = [](const double data[], const uint8_t filter[], int points) {
            double total = 0;
            bool valid= false;
            for(int i=0; i < points ; i++)
            {
                if (filter[i] && !std::isnan(data[i])) {
                    valid = true;
                    total += data[i];
                }
            }
            return valid ? total : nan;
        };
        break;

    default:
        throw PBException(type_.toString() + " not supported");
    }
}

void Filter::Load( const ZmwStatsFile& file)
{
    data_.resize(file.nH());

    std::vector<uint32_t> holeNumbers(file.nH());

    switch (type_)
    {
    case Filter_t::All:
        std::fill(data_.begin(), data_.end(), 1);
        break;

    case Filter_t::P0:
    {
        ZmwStatDataSet ds = file.GetDataSet("/ZMWMetrics/Productivity");
        ds.DataSet() >> data_;
        for (auto& d : data_)
        {
            d = (d == 0);
        }
    }
        break;

    case Filter_t::P1:
    {
        ZmwStatDataSet ds = file.GetDataSet("/ZMWMetrics/Productivity");
        ds.DataSet() >> data_;
        for (auto& d : data_)
        {
            d = (d == 1);
        }
    }
        break;

    case Filter_t::P2:
    {
        ZmwStatDataSet ds = file.GetDataSet("/ZMWMetrics/Productivity");
        ds.DataSet() >> data_;
        for (auto& d : data_)
        {
            d = (d == 2);
        }
    }
        break;

    case Filter_t::Sequencing:
    {
        static bool warnOnce = [](){PBLOG_WARN << "ZmwReducedStats treating all ZMW as sequencing"; return true;}();
        (void) warnOnce;
        //std::string layoutName = file.ChipLayoutName();
        //auto chipLayout = ChipLayout::Factory(layoutName);
        ZmwStatDataSet ds = file.GetDataSet("/ZMW/HoleNumber");
        ds.DataSet() >> holeNumbers;
        for (uint32_t i=0;i<data_.size();i++)
        {
            //UnitCell uc(holeNumbers[i]);
            //data_[i] = static_cast<uint8_t>(chipLayout->IsSequencing(static_cast<uint16_t>(uc.x),
            //                                                         static_cast<uint16_t>(uc.y)));
            data_[i] = 1;
        }
    }
        break;

    case Filter_t::NonSequencing:
    {
        //std::string layoutName = file.ChipLayoutName();
        //auto chipLayout = ChipLayout::Factory(layoutName);
        ZmwStatDataSet ds = file.GetDataSet("/ZMW/HoleNumber");
        ds.DataSet() >> holeNumbers;
        for (uint32_t i=0;i<data_.size();i++)
        {
            //UnitCell uc(holeNumbers[i]);
            //data_[i] = static_cast<uint8_t>(!chipLayout->IsSequencing(static_cast<uint16_t>(uc.x),
            //                                                          static_cast<uint16_t>(uc.y)));
            data_[i] = 0;
        }
    }
        break;

    case Filter_t::Control:
    {
        ZmwStatDataSet ds = file.GetDataSet("/ZMWMetrics/IsControl");
        ds.DataSet() >> data_;
        for (auto& d : data_)
        {
            d = (d == 1);
        }
    }

    default:
        throw PBException("Filter type not supported:" + Name());
    }

}

uint8_t Filter::IsSelected(uint32_t zmwIndex) const
{
    return data_.at(zmwIndex);
}

}}}


