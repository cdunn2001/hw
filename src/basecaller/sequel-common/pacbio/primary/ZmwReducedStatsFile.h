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

#ifndef PPA_ZMWREDUCEDSTATSFILE_H
#define PPA_ZMWREDUCEDSTATSFILE_H

#include <functional>

#include <pacbio/logging/Logger.h>
#include <pacbio/process/ConfigurationBase.h>
#include <pacbio/primary/SequelHDF5.h>
#include <pacbio/text/String.h>

namespace Json {
class Value;
}

namespace PacBio {
namespace Primary {

class ZmwStatsFile;   // see ZmwStatsFile.h
class ZmwStatDataSet; // see ZmwStatsFile.h

/// Configuration for the datasets in the rsts.h5 file. In addition to these explicit fields,
/// all of the binning parameters in the parent can be specified, which will override the values in the true parent.
class ReducedStatsDatasetConfig : public PacBio::Process::ConfigurationObject
{
    ADD_PARAMETER(std::string,Input,""); ///< The full HDF5 path name of the input dataset from the sts.h5 file.
    ADD_PARAMETER(std::string,Type,"float"); ///< The type for the HDF5 output datasets. Float should be used in almost all cases.
    ADD_PARAMETER(std::string,Algorithm,"Subsample"); ///< the method used to aggregate the data in the bin to a single value.
    ADD_PARAMETER(std::string,Filter,"All"); ///< The filter used to select certain classes of ZMWs within each ibin

};

/// Configuration for the rsts.h5, which will be applied to all datasets.
class ReducedStatsConfig : public PacBio::Process::ConfigurationObject
{
    ADD_PARAMETER(uint32_t, BinRows, 20);    ///< the number of rows aggregated into the reduced row. For example, 8 rows from the ``.sts.h5`` file are reduced to one row in the ``.rsts.h5`` file. A typical value will be 8.
    ADD_PARAMETER(uint32_t, BinCols, 20);    ///< the number of columns aggregated in to the reduced column. For example, 10 columns from the ``.sts.h5`` file are reduced to one column in the ``.rsts.h5`` file.  A typical value will be 8.
    ADD_PARAMETER(uint16_t, MinOffsetX, 0);  ///< the minimum Unit Cell X coordinate of the first Unit Cell of the first bin stored in the ``.rsts.h5`` file.  Actual value used will not be smaller than the smallest X coordinate of the input ```sts.h5`` file
    ADD_PARAMETER(uint16_t, MinOffsetY, 0);  ///< the minimum Unit Cell Y coordinate of the first Unit Cell of the first bin stored in the ``.rsts.h5`` file.   Actual value used will not be smaller than the smallest X coordinate of the input ```sts.h5`` file
    ADD_PARAMETER(uint32_t, MaxRows, std::numeric_limits<uint16_t>::max());   ///< Number max of rows of unit cells to use an input.  Actual value used will be bounded by the actual input ``sts.h5`` file
    ADD_PARAMETER(uint32_t, MaxCols, std::numeric_limits<uint16_t>::max());   ///< Number max of colulmns of unit cells to use as input.  Actual value used will be bounded by the actual input ``sts.h5`` file
    ADD_ARRAY(ReducedStatsDatasetConfig,Outputs);

public:
    /// todo: refactor this to use ChipLayout instead of ChipClass.
    /// This constructor sets up a bunch of default values for the rsts.h5 file. It currently keys off of the
    /// chipClass argument, but this is too coarse of a key. The correct argument should be a ChipLayout reference,
    /// and then all of the row and column dependent settings can be extracted straight from the chip layout.
    /// The `switch(chipClass)` logic below should be removed, as well as SetSpiderDefaults.
    explicit ReducedStatsConfig()
    {
        const std::string defaults = R"json(
        { "Outputs":
            [
                { "Input": "/ZMWMetrics/NumBases",          "Algorithm": "Sum",         "Filter": "All"},
                { "Input": "/ZMWMetrics/HQPkmid",           "Algorithm": "Median",      "Filter": "P1"},
                { "Input": "/ZMWMetrics/SnrMean",           "Algorithm": "Median",      "Filter": "Sequencing"},
                { "Input": "/ZMWMetrics/HQRegionStart",     "Algorithm": "Median",      "Filter": "P1"},
                { "Input": "/ZMWMetrics/HQRegionStartTime", "Algorithm": "Median",      "Filter": "P1", "Type": "uint16" },
                { "Input": "/ZMWMetrics/Loading",           "Algorithm": "Count=0",     "Filter": "Sequencing", "Type": "uint8"},
                { "Input": "/ZMWMetrics/Loading",           "Algorithm": "Count=1",     "Filter": "Sequencing", "Type": "uint8"},
                { "Input": "/ZMWMetrics/Loading",           "Algorithm": "Count=2",     "Filter": "Sequencing", "Type": "uint8"},
                { "Input": "/ZMWMetrics/Loading",           "Algorithm": "Count=3",     "Filter": "Sequencing", "Type": "uint8"},
                { "Input": "/ZMWMetrics/Productivity",      "Algorithm": "Count=0",     "Filter": "Sequencing", "Type": "uint8"},
                { "Input": "/ZMWMetrics/Productivity",      "Algorithm": "Count=1",     "Filter": "Sequencing", "Type": "uint8"},
                { "Input": "/ZMWMetrics/Productivity",      "Algorithm": "Count=2",     "Filter": "Sequencing", "Type": "uint8"},
                { "Input": "/ZMWMetrics/Productivity",      "Algorithm": "Subsample",   "Filter": "All",        "Type": "uint8", "BinRows": 1, "BinCols": 1},
                { "Input": "/ZMWMetrics/BaselineLevel",     "Algorithm": "Median",      "Filter": "Sequencing" },
                { "Input": "/ZMWMetrics/HQRegionSnrMean",   "Algorithm": "Median",      "Filter": "P1"},
                { "Input": "/ZMWMetrics/BaselineLevel",     "Algorithm": "Mean",        "Filter": "Sequencing"},
                { "Input": "/ZMWMetrics/LocalBaseRate",     "Algorithm": "Median",      "Filter": "P1"},
                { "Input": "/ZMWMetrics/ReadLength",        "Algorithm": "Mean",        "Filter": "P1"},
                { "Input": "/ZMWMetrics/BaseWidth",         "Algorithm": "Mean",        "Filter": "P1"},
                { "Input": "/ZMWMetrics/BaseIpd",           "Algorithm": "Mean",        "Filter": "P1"}
            ]
        }
    )json";
        this->Load(defaults);
    }
};

namespace Reducer {

/// NaN is used to represent an output value that could not be determined because
/// no input data was selected from the filter function.  For example, if the filter
/// was "hqr" but no HQR ZMWS were found in a bin, the reduced value will be NaN().
double NaN();

/// A convenience class to hold all binning configurations in one place.
class Binning
{
public:
    /// Constructor
    struct Sizes
    {
        uint16_t minX;
        uint16_t minY;
        uint16_t maxX;
        uint16_t maxY;
    };
    Binning(const ReducedStatsConfig& config,
            const Sizes& sizes)
    {
        binRows_ = config.BinRows();
        binCols_ = config.BinCols();

        unitCellOffsetX_ = std::max(config.MinOffsetX(), sizes.minX);
        unitCellOffsetY_ = std::max(config.MinOffsetY(), sizes.minY);
        unitCellRows_ = std::min(config.MaxRows(), sizes.maxX - sizes.minX + 1u);
        unitCellCols_ = std::min(config.MaxCols(), sizes.maxY - sizes.minY + 1u);

        if (binRows_ == 0) throw PBException("BinRows was zero");
        if (binCols_ == 0) throw PBException("BinCols was zero");

        numOutputRows_ = (unitCellRows_+binRows_-1)/binRows_;
        numOutputCols_ = (unitCellCols_+binCols_-1)/binCols_;

        PBLOG_DEBUG<< "numOutputRows_" << numOutputRows_;
        PBLOG_DEBUG << "numOutputCols_" << numOutputCols_;
        PBLOG_DEBUG << "binRows_" << binRows_;
        PBLOG_DEBUG << "binCols_" << binCols_;
        PBLOG_DEBUG << "unitCellOffsetX_" << unitCellOffsetX_;
        PBLOG_DEBUG << "unitCellOffsetY_" << unitCellOffsetY_;
        PBLOG_DEBUG << "unitCellRows_" << unitCellRows_;
        PBLOG_DEBUG << "unitCellCols_" << unitCellCols_;
    }

private:
    uint32_t numOutputRows_ = 0;
    uint32_t numOutputCols_ = 0;

    uint32_t binRows_;
    uint32_t binCols_;
    uint32_t unitCellOffsetX_;
    uint32_t unitCellOffsetY_;
    uint32_t unitCellRows_;
    uint32_t unitCellCols_;

public:

    uint32_t BinRows() const { return binRows_; } ///< the number of rows aggregated into the reduced row. For example, 8 rows from the ``.sts.h5`` file are reduced to one row in the ``.rsts.h5`` file. A typical value will be 8.
    uint32_t BinCols() const { return binCols_; } ///< he number of columns aggregated in to the reduced column. For example, 10 columns from the ``.sts.h5`` file are reduced to one column in the ``.rsts.h5`` file.  A typical value will be 8.

    uint32_t UnitCellOffsetX() const { return unitCellOffsetX_;} ///< the Unit Cell X coordinate of the first Unit Cell of the first bin
    uint32_t UnitCellOffsetY() const { return unitCellOffsetY_;} ///< the Unit Cell Y coordinate of the first Unit Cell of the first bin

    uint32_t UnitCellRows() const { return unitCellRows_;} ///< The number of rows of unit cells to use as input
    uint32_t UnitCellCols() const { return unitCellCols_;} ///< The number of columns of unit cells to use as input

    uint32_t NumOutputRows() const { return numOutputRows_;} ///< This is derived value, which indicates how many rows will be output
    uint32_t NumOutputCols() const { return numOutputCols_;} ///< This is a derived value, which indicates how many columns will be output

    uint32_t NumBinValues() const { return binRows_ * binCols_;} ///< The total number of ZMWS in a rectangular bin
    uint32_t NumOutputCount() const { return numOutputRows_ * numOutputCols_; } ///< The total number of output values
    uint32_t NumInputCount() const { return unitCellRows_ * unitCellCols_; } ///< The total number of unit cellls used as input
};


/// This class choses the algorithm to use for reducing the data in the bin to a single value.
class Algorithm
{
public:
    /// * ``"Count=N"`` - count the values that match the enumeration value. The numeration value N can be any integer value.
    ///   For example, ``"count=1"`` for ``Productivity`` will match enumeration value 1 : "Productive HQ-Region".
    /// * ``"Sum"`` - the aggregate value is the sum of values.
    /// * ``"Subsample"`` - the aggregate  value is simply the value of the first ZMW in the bin
    /// * ``"Mean"`` - the aggregate value is the mean in the bin.
    /// * ``"Median"`` - the aggregate value is the median of values.
    ///
    /// Other future algorithms that may be supported are ``min``, ``max``, ``stddev``,
    /// ``MAD`` (`Median Absolute Deviation <https://en.wikipedia.org/wiki/Median_absolute_deviation>`_)
    SMART_ENUM (Algorithm_t,None,Count,Sum,Subsample,Mean,Median,Min,Max,Stddev,MAD);
    std::string ShortName() const { return type_.toString(); }
    std::string FullName() const { return fullName_; }

    Algorithm (const std::string& s);
    Algorithm (Algorithm_t a);
    double Apply(const std::vector<double>& data, const std::vector<uint8_t>& filter)
    {
        return function_(data.data(), filter.data(), data.size());
    }
private:
    Algorithm_t type_;
    std::string fullName_;
    std::function<double(const double[],const uint8_t[],int)> function_;
};

/// The filter used to select ZMWs within the bin.
/// * ``"all"`` - all ZMWS are used (for example, if the bin size is 8x10, then 80 ZMWS are selected). This will include
///   nonsequencing ZMWS, such as fiducials and anti-ZMWs.
/// * ``"normal"`` - only normal ZMWS that are identified as "sequencing" (i.e. not fiducials, anti-ZMWS, etc) are
///   including in the aggregation.
/// * ``"notnormal"`` - only ZMWs that are not identified as normal ZMWS,
///   i.e. fiducials, anti-ZMWS, etc. Base calling is performed on all
///   ZMWS, regardless of type, so selecting ZMWS that are not expected to produce base calls can be a useful
///   sanity check by verifying there is no significant activity.
/// * ``"hqr"`` - only ZMWs with a High Quality Region (HQR) are used in the aggregate.
///   This does not apply to the ``"subsample"`` aggregation.

class Filter
{
public:
    SMART_ENUM(Filter_t,All,P0,P1,P2,Sequencing,NonSequencing,Control);

    Filter(Filter_t type) : type_(type) {}
    std::string Name() const { return type_.toString(); }
    static Filter Factory(const std::string& s) { return Filter(Filter_t::fromString(s));}
    void Load(const ZmwStatsFile& file);
    uint8_t IsSelected(uint32_t zmwIndex) const;
private:
    Filter_t type_;
    std::vector<uint8_t> data_;
};

} //namespace


/// This class wraps around the H5 Data set object, offering some convenience
/// functions for accessing attributes.

class ZmwReducedDataSet : public PBDataSet
{
public:
    /// read-only constructor
    ZmwReducedDataSet();

    /// initialize from source dataset
    void Copy(const ZmwStatDataSet& dataset);

    /// returns description from the H5 attribute
    std::string Description() const
    {
        return ReadAttribute<std::string>("Description");
    }

    /// Returns whether the dataset is over a HQ Region or not
    /// This is from an attribute.
    bool HQRegion() const
    {
        uint32_t x = ReadAttribute<uint32_t>("HQRegion");
        return x != 0;
    }

    /// Returns the units or encoding of the dataset.
    /// This is from an attribute.
    std::string UnitsOrEncoding() const
    {
        return ReadAttribute<std::string>("UnitsOrEncoding");
    }

    /// If the UnitsorEncoding is an encoding, then this will
    /// return a map of integral values to strings.
    std::map<int,std::string> Encoding() const
    {
        std::string mapstring = UnitsOrEncoding();
        std::map<int,std::string> map;
        if (mapstring!="")
        {
            auto mapping = PacBio::Text::String::Split(mapstring, ',');
            for (auto& elem : mapping)
            {
                auto pieces = PacBio::Text::String::Split(elem, ':');
                int i = std::stoi(pieces[0]);
                map[i] = pieces[1];
            }
        }
        return map;
    }

    /// binsize in frames. 0 if not calculated
    uint32_t BinSize() const
    {
        return ReadAttribute<uint32_t>("BinSize");
    }

    /// The size of the bin in rows, read from the attributes.
    uint32_t BinRows() const { return ReadAttribute<uint32_t>("BinRows");}

    /// The size of the bin in columns, read from the attributes.
    uint32_t BinCols() const { return ReadAttribute<uint32_t>("BinCols");}

    /// The position of the first unit cell to be reduced, in X coordinate, read from the attributes
    uint32_t UnitCellOffsetX() const { return ReadAttribute<uint32_t>("UnitCellOffsetX");}

    /// The position of the first unit cell to be reduced, in Y coordinate, read from the attributes
    uint32_t UnitCellOffsetY() const { return ReadAttribute<uint32_t>("UnitCellOffsetY");}

    /// The algorithm used to create the dataset, read from the attributes.
    Reducer::Algorithm::Algorithm_t Algorithm() const { return Reducer::Algorithm::Algorithm_t::fromString(ReadAttribute<std::string>("Algorithm"));}

    /// The filter method used to select ZMWS from the source data set, read from the attributes.
    Reducer::Filter::Filter_t Filter() const { return Reducer::Filter::Filter_t::fromString(ReadAttribute<std::string>("Filter"));}

private:
    friend class ZmwReducedStatsFile;
};


/// A class to manage creation and readback of the reduced statistics file (rsts.h5)
/// To create an rsts file, a JSON configuration and a source sts.h5 file must be used.

class ZmwReducedStatsFile
{
public:
    /// Readonly constructor
    ZmwReducedStatsFile(const std::string& filename);

    /// Creator constructor
    ZmwReducedStatsFile(const std::string& filename,const ReducedStatsConfig& config);

    /// destructor
    ~ZmwReducedStatsFile();

    H5::Group& ScanData() { return scanData_; }

    /// Will reduce the indicated datasets (from the config.Outputs[].Input data sets)
    /// from the indicated input file, and create reduced data sets in the current
    /// object.
    /// \param inputFile - sts.h5 file full path
    /// \param config - the instructions on how to reduce the datasets.
    void Reduce(const ZmwStatsFile& inputFile,const ReducedStatsConfig& config);

    /// Gets the datasets object from this current file.
    /// \param dataSetPath - the absolute path to the dataset, starting a '/'
    ZmwReducedDataSet GetDataSet(const std::string& dataSetPath);

    /// debug feature enabler.
    /// \param flag - If true, then debug images (in NetPBM *.pgm format) are dropped in the current working
    /// directory.
    void SetImageOption(bool flag) { imageOption_ = flag; }

    /// Close the underlying file. This is automatically done with the destructor, but
    /// there may be cases where an explicit close is desired before destruction, so that the file
    /// can be read-back.
    void Close();
private:
    ZmwReducedDataSet CreateDataSet(const std::string& dataSetPath,
                                    H5::DataType& dt, H5::DataSpace& ds,
                                    H5::DSetCreatPropList& propList);
    void Create(const std::string& filename, const ReducedStatsConfig& config);
    void Reduce(const std::string& name, ZmwReducedDataSet& zrds, const ZmwStatDataSet& ds, Reducer::Binning& binning,
                Reducer::Algorithm& algo, Reducer::Filter& filter);
    std::pair<uint32_t,uint32_t> GetCoordinate(int index) const;

private:
    H5::H5File hdf5file_; ///< the H5 object
    H5::Group scanData_;
    const ZmwStatsFile* currentZmwStatsFile_ = nullptr;
    bool imageOption_ = false;

};

}}

#endif //PPA_ZMWREDUCEDSTATSFILE_H
