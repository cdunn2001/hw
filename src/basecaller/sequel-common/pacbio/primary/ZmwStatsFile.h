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

#ifndef SEQUELACQUISITION_ZMWSTATSFILE_H
#define SEQUELACQUISITION_ZMWSTATSFILE_H

#include <string>
#include <future>
#include <thread>
#include <functional>
#include <pacbio/utilities/SmartEnum.h>
#include <pacbio/ipc/ThreadSafeQueue.h>
#include <pacbio/primary/SequelHDF5.h>
#include <pacbio/logging/Logger.h>
#include <pacbio/primary/ZmwStatsFileData.h>

namespace PacBio {
namespace Primary {

enum ZmwStatsOption : int
{
    NONE = 0,
    QUIET = 1
};

bool ZmwStatsQuiet();
void ZmwStatsSetQuiet();
void ZmwStatsSetVerbose();


/// Describes a single H5 dataset, and a few attributes. All sts.h5 datasets have the same
/// 4 attributes. This object is also used to describe the structure of the datasets, so that
/// the underlying H5 objects can be constructed from them.

class ZmwStatDataSet : public PBDataSet
{
public:
    /// How much zlib compression to apply.  0 = none (fastest), 9 = max (slow), 3 = default
    static unsigned int ZLibCompressionLevel;

public:
    /// Constructor for 1D dataset
    /// \param name - the H5 name of the dataset
    /// \param dataType - the H5 data type
    /// \param units - the units for this dataset. Can be "", to be stored in an attribute
    /// \param description - the description of this dataset, to be stored in an attribute
    /// \param hqRegion - a boolean value to be stored in an attribute
    /// \param dim0 - a reference to a variable that holds the first dimension. Should be nH_ in all cases.
    /// \param isDiagnostic - optional diagnostic dataset

    ZmwStatDataSet(const char* name, H5::DataType dataType, const char* units,
                   const char* description, bool hqRegion, hsize_t& dim0,
                   bool isDiagnostic=false) :
            name_(name),
            dataType_(dataType),
            units_(units),
            description_(description),
            dimensions_(1),
            hqRegion_(hqRegion),
            temporal_(false),
            dim0_(dim0),
            dim1_(unused),
            dim2_(unused),
            isDiagnostic_(isDiagnostic)
    {

    }
    /// Constructor for 2D data
    /// see 1D data set for common parameters
    /// \param prefix If the prefix is "nMF_", this dataset is assumed to be temporal
    /// \param dim1 - a reference to a variable that holds the second dimension. Usually nA_. nF_ or nMF_.
    ZmwStatDataSet(const char* name, H5::DataType dataType, const char* units,
                   const char* description,  bool hqRegion, const char* prefix, hsize_t& dim0, hsize_t& dim1,
                   bool isDiagnostic=false) :
            name_(name),
            dataType_(dataType),
            units_(units),
            description_(description),
            dimensions_(2),
            hqRegion_(hqRegion),
            temporal_(strcmp("nMF_",prefix)==0),
            dim0_(dim0),
            dim1_(dim1),
            dim2_(unused),
            isDiagnostic_(isDiagnostic)
    {

    }

    /// Constructor for 3D data
    /// see 2D data set for common parameters
    /// \param dim2 - a reference to a variable that holds the second dimension. Usually nA_. nF_
    ZmwStatDataSet(const char* name, H5::DataType dataType, const char* units,
                   const char* description,  bool hqRegion, const char* prefix, hsize_t& dim0,  hsize_t& dim1, hsize_t& dim2,
                   bool isDiagnostic=false) :
            name_(name),
            dataType_(dataType),
            units_(units),
            description_(description),
            dimensions_(3),
            hqRegion_(hqRegion),
            temporal_(strcmp("nMF_",prefix)==0),
            dim0_(dim0),
            dim1_(dim1),
            dim2_(dim2),
            isDiagnostic_(isDiagnostic)
    {

    }

    ZmwStatDataSet(const H5::H5File& file, const std::string& name) :
            name_(""),
            dataType_(),
            units_(),
            description_(),
            dimensions_(),
            hqRegion_(),
            temporal_(),
            dim0_(unused),
            dim1_(unused),
            dim2_(unused),
            isDiagnostic_(false)
    {
        dataSet_ = file.openDataSet(name);
    }

    ZmwStatDataSet& operator=(const ZmwStatDataSet& source)
    {
        dataSet_ = source.dataSet_;
        return *this;
    }

    ZmwStatDataSet() :
        name_(""),
        dataType_(),
        units_(),
        description_(),
        dimensions_(),
        hqRegion_(),
        temporal_(),
        dim0_(unused),
        dim1_(unused),
        dim2_(unused),
        isDiagnostic_(false)
    {
    }

    /// create the underlying H5 object corresponding to this
    void Create(H5::Group& group, uint32_t binSize);

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
        if(mapstring !="")
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

    /// returns a H5 dataspace object that corresponds to this dataset
    H5::DataSpace GetDataSpace()
    {
        hsize_t startSpace[3];
        startSpace[0] = dim0_;
        startSpace[1] = dim1_;
        startSpace[2] = dim2_;

        return H5::DataSpace(dimensions_,startSpace);
    }


#if 0
        template<typename T>
    void Append(T value)
    {
        auto foffset = GetDims(dataSet_);
        auto fsize = foffset;
        auto msize = foffset;
        msize[0] = 1 ; // size in memory (1 element)
        fsize[0]++; // total size on disk
        dataSet_.extend(fsize.data());

        H5::DataSpace memspace(1, msize.data());
        H5::DataSpace filespace = dataSet_.getSpace();
        filespace.selectHyperslab(H5S_SELECT_SET, fsize.data(), foffset.data());
        dataSet_.write(&value, GetType<T>(), memspace, filespace);
    }
#endif

    /// Sets the value of the index'th ZMW of this dataset. Note that value may be just the
    /// first element of a contiguous memory array. For example std::vector[0] or boost::multi_array[0][0]
    template<typename T>
    void Set(hsize_t index, T& value, bool set=true)
    {
        if (set)
        {
            std::vector<hsize_t> foffset(dimensions_, 0);
            foffset[0] = index;
            std::vector<hsize_t> fsize(dimensions_);
            fsize[0] = 1;
            if (dimensions_ >= 2) fsize[1] = dim1_;
            if (dimensions_ >= 3) fsize[2] = dim2_;


            H5::DataSpace memspace(dimensions_, fsize.data());
            H5::DataSpace filespace = dataSet_.getSpace();
            filespace.selectHyperslab(H5S_SELECT_SET, fsize.data(), foffset.data());
            dataSet_.write(&value, GetType<T>(), memspace, filespace);
        }
    }

    template<typename T, typename Getter>
    void SetVector(std::vector<PacBio::Primary::ZmwStats>& buffer, Getter callback, bool add=true)
    {
        if (add)
        {
            std::vector<hsize_t> foffset(dimensions_, 0);
            foffset[0] = buffer[0].index_;
            std::vector<hsize_t> fsize(dimensions_);
            fsize[0] = buffer.size();
            hsize_t elementSize = 1;
            if (dimensions_ >= 2)
            {
                fsize[1] = dim1_;
                elementSize *= fsize[1];
            }
            if (dimensions_ >= 3)
            {
                fsize[2] = dim2_;
                elementSize *= fsize[2];
            }

            boost::multi_array<T, 2> contiguous(boost::extents[buffer.size()][elementSize]);

            // fill contiguous;
            for (uint32_t i = 0; i < buffer.size(); i++)
            {
                memcpy(&contiguous[i][0], callback(buffer[i]), sizeof(T) * elementSize);
            }


            H5::DataSpace memspace(dimensions_, fsize.data());
            H5::DataSpace filespace = dataSet_.getSpace();
            filespace.selectHyperslab(H5S_SELECT_SET, fsize.data(), foffset.data());
            dataSet_.write(contiguous.data(), GetType<T>(), memspace, filespace);
        }
    }

    /// Sets the value of the index'th ZMW of this dataset. Note that value may be just a pointer to the
    /// first element of a contiguous memory array. For example std::vector[0] or boost::multi_array[0][0]
    template<typename T>
    void Get(hsize_t index, T* value, bool get=true) const
    {
        if (get)
        {
            std::vector<hsize_t> foffset(dimensions_, 0);
            foffset[0] = index;
            std::vector<hsize_t> fsize(dimensions_);
            fsize[0] = 1;
            if (dimensions_ >= 2) fsize[1] = dim1_;
            if (dimensions_ >= 3) fsize[2] = dim2_;

            H5::DataSpace memspace(dimensions_, fsize.data());
            H5::DataSpace filespace = dataSet_.getSpace();
            filespace.selectHyperslab(H5S_SELECT_SET, fsize.data(), foffset.data());
            dataSet_.read(value, GetType<T>(), memspace, filespace);
        }
    }

    /// Returns a human readable representation of all the values of this dataset for a particular ZMW. If the dataset is scalar,
    /// then it is a simple numerical value. If the data is not a scalar, then the structure is flattened
    /// and all values are concatenated with '/' delimeters.
    std::string GetValue(hsize_t index) const;

    /// Returns a human readable representation of all the values of this dataset for a particular ZMW
    /// at a particular time. If the dataset is scalar,
    /// then it is a simple numerical value. If the data is not a scalar, then the structure is flattened
    /// and all values are concatenated with '/' delimeters.
    std::string GetValue(hsize_t index, hsize_t t) const;

    /// Returns the name of the dataset
    const char* Name() const { return name_; }

    // Returns the Units of the dataset
    const char* Units() const { return units_; }

    /// Returns the H5 DataType of the dataset
    const H5::DataType& GetDataType() const { return dataType_; }

    /// Returns true if this dataset is temporal, i.e. the 2nd index is dimensioned to nT
    bool IsTemporal() const { return temporal_; }

    /// Returns true if this dataset is diagnostic
    bool IsDiagnostic() const { return isDiagnostic_; }

private:
    const char* name_ = nullptr;
    const H5::DataType dataType_;
    const char* units_;
    const char* description_;
    int dimensions_;
    bool hqRegion_;
    bool temporal_;
    hsize_t& dim0_;
    hsize_t& dim1_;
    hsize_t& dim2_;
    static hsize_t unused;
    bool isDiagnostic_;
public:
    friend class ZmwStatGroup;
};

/// Represents a H5 group, and manages the group hierarchy by having children.
/// This C++ class is built at compile time, to create the underlying HDF5 group objects
/// at run time.
class ZmwStatGroup
{
public:
    /// Constructor
    ZmwStatGroup(const char* name, bool isDiagnostic=false)
    : name_(name)
    , isDiagnostic_(isDiagnostic)
    { }

    /// Open existing H5 group for read
    void Open(ZmwStatGroup* parent)
    {
        PBLOG_TRACE << "Open(parent) " << name_;
        group_ = parent->group_.openGroup(name_);
    }

    /// create a new H5 group
    void Create(ZmwStatGroup* parent, bool addDiagnostics)
    {
        PBLOG_TRACE << "Create(parent) " << name_;
        group_ = parent->group_.createGroup(name_);

        if (isDiagnostic_ && !addDiagnostics)
            parent->group_.unlink(name_);
    }

    /// Open all H5 datasets and H5 groups hierarchically from this group.
    void OpenAll(ZmwStatGroup* parent)
    {
        if (parent !=nullptr)
        {
            try
            {
                DisableDefaultErrorHandler noErrors;
                Open(parent);
            }
            catch(const H5::GroupIException& )
            {
                if (!ZmwStatsQuiet())
                {
                    PBLOG_WARN << "Could not open group " << this->Name() << " from " << parent->Name();
                }
            }
        }
        for(ZmwStatDataSet* dataset : datasets_)
        {
            try
            {
                DisableDefaultErrorHandler noErrors;
                dataset->dataSet_ = group_.openDataSet(dataset->Name());
            }
            catch(const H5::Exception&)
            {
                if (!ZmwStatsQuiet())
                {
                    PBLOG_WARN << "Could not open " << dataset->Name();
                }
            }
        }
        for(ZmwStatGroup* group : groups_)
        {
            group->OpenAll(this);
        }
    }

    /// Create all H5 datasets and H5 groups hierarchically from this group.
    void CreateAll(ZmwStatGroup* parent, uint32_t binSize, bool addDiagnostics)
    {
        if (parent != nullptr)
        {
            Create(parent, addDiagnostics);
        }
        for(ZmwStatDataSet* dataset : datasets_)
        {
            dataset->Create(group_, binSize);
        }

        for(ZmwStatGroup* group : groups_)
        {
            group->CreateAll(this, binSize, addDiagnostics);
        }
    }

    /// Recursively visits all datasets (via group hierarchy) and builds a vector of dataset
    /// names in the headers argument. It is a breadth-first walking of the tree.
    /// The type of datasets included depends on the temporal flag.
    void VisitAllHeaders(std::vector<std::string>& headers, bool temporal = false) const
    {
        for(ZmwStatDataSet* dataset : datasets_)
        {
            if (dataset->IsTemporal() == temporal)
                headers.push_back(std::string(dataset->Name()));
        }

        for(ZmwStatGroup* group : groups_)
        {
            group->VisitAllHeaders(headers,temporal);
        }
    }

    /// Recursively visits all datasets (via group hierarchy) and builds a vector of dataset
    /// values in the values argument, for a given ZMW. It is a breadth-first walking of the tree.
    /// Only non-temporal datasets are visited.
    void VisitAllValues(uint32_t index, std::vector<std::string>& values) const
    {
        for(ZmwStatDataSet* dataset : datasets_)
        {
            if (!dataset->IsTemporal() && !dataset->IsDiagnostic())
                values.push_back(dataset->GetValue(index));
        }

        for(ZmwStatGroup* group : groups_)
        {
            group->VisitAllValues(index,values);
        }
    }

    /// Recursively visits all datasets (via group hierarchy) and builds a vector of dataset
    /// values in the values argument, for  given ZMW at a given time. It is a breadth-first walking of the tree.
    /// Only non-temporal datasets are visited.
    void VisitAllTemporalValues(uint32_t index, std::vector<std::string>& headers, uint32_t t) const
    {
        for(ZmwStatDataSet* dataset : datasets_)
        {
            if (dataset->IsTemporal())
                headers.push_back(dataset->GetValue(index,t ));
        }

        for(ZmwStatGroup* group : groups_)
        {
            group->VisitAllTemporalValues(index,headers, t);
        }
    }

    /// Adds a ZmwStatGroup as a child of this group.
    void AddChildGroup(ZmwStatGroup* group)
    {
        groups_.push_back(group);
    }
    /// Adds a ZmwStatDataSet as a child of this group.
    void AddChildDataset(ZmwStatDataSet* dataset)
    {
        datasets_.push_back(dataset);
    }

    /// Creates an attribute in the group
    template<typename T>
    void CreateAttribute(const char* name, H5::DataType S, T value)
    {
        H5::DataSpace scalar;
        auto attr = group_.createAttribute(name, S , scalar);
        attr << value;
    }

    /// Reads a attribute from the group
    template<typename T>
    T ReadAttribute(const char* name) const
    {
        auto attr = group_.openAttribute(name);
        T s;
        attr >> s;
        return s;
    }

    const std::string Name() const { return name_; }

    H5::Group& Group() { return group_; }

    const H5::Group& ReadOnlyGroup() const { return group_; }

private:
    const char* name_;
    bool isDiagnostic_;
    std::vector<ZmwStatGroup*> groups_;
    std::vector<ZmwStatDataSet*> datasets_;
    H5::Group group_;
    friend class SequelHDF5ObjectManager;
};


/// This manages all of the ZmwStatGroup and ZmwStatDataset objects that
/// map to corresponding H5 objects in the H5 file. It provides a method of traversing
/// the hierarchy before the H5 file exists. The main use for this is creating
/// a new H5 file.
class SequelHDF5ObjectManager
{
public:
    /// Constructor
    SequelHDF5ObjectManager()
    {
        StartGroup(topGroup_);
    }

    /// Destructor
    ~SequelHDF5ObjectManager()
    {
        if (hdf5file_)
        {
            hdf5file_->close();
        }
    }

    void Flush()
    {
    }

    /// Hierarchically adds a new group to the current group
    void StartGroup(ZmwStatGroup& group)
    {
        if (currentGroup_ != nullptr)
        {
            currentGroup_->AddChildGroup(&group);
            groupStack_.push(currentGroup_);
        }
        currentGroup_ = &group;
    }
    /// Ends the current group and moves back up the hierarchy
    void EndGroup(ZmwStatGroup& /*group*/)
    {
        currentGroup_ = groupStack_.top();
        groupStack_.pop();
    }


    /// Adds a dataset to the current group
    void AddDataSet(ZmwStatDataSet& dataset)
    {
        if (currentGroup_ == nullptr) throw PBException("no group defined");
        currentGroup_->AddChildDataset(&dataset);
    }

    /// Walks the entire group/dataset tree and opens each  group and dataset H5 object.
    void OpenAll(const std::string& filename)
    {
        if (groupStack_.size() > 0) throw PBException("groups not corrected start/ended");

        hdf5file_ = std::make_unique<H5::H5File>();
        try
        {
            hdf5file_->openFile(filename.c_str(), H5F_ACC_RDONLY);
        }
        catch(...)
        {
            hdf5file_.reset();
            throw;
        }
        topGroup_.group_ = hdf5file_->openGroup("/");
        topGroup_.OpenAll(nullptr);
    }

    /// Walks the entire group/dataset tree and creates groups and datasets as H5 objects.
    void CreateAll(const std::string& filename, uint32_t numBins, bool addDiagnostics)
    {
        addDiagnostics_ = addDiagnostics;
        if (addDiagnostics_)
        {
            StartGroup(*diagGroup_);
            for (auto& ds : diagDatasets_)
            {
                AddDataSet(*ds);
            }
            EndGroup(*diagGroup_);
        }

        filename_ = filename;
        if (groupStack_.size() > 0) throw PBException("groups not corrected start/ended");

        hdf5file_.reset(new H5::H5File(filename.c_str(), H5F_ACC_TRUNC));
        topGroup_.group_ = hdf5file_->openGroup("/");
        topGroup_.CreateAll(nullptr, numBins, addDiagnostics_);
    }

    /// Returns a list of all ColumnHeaders, aka dataset names, for nontemporal datasets.
    std::vector<std::string> ColumnHeaders() const {
        std::vector<std::string> headers;
        topGroup_.VisitAllHeaders(headers, false);

        return headers;
    }
    /// Returns a list of all ColumnValues (numerical only) for al lnontemporal datasets,
    /// given a particular index.
    /// \param zmwIndex - the index of the ZMW to get values for.
    std::vector<std::string> ColumnValues(uint32_t zmwIndex) const {
        std::vector<std::string> values;
        topGroup_.VisitAllValues(zmwIndex, values);
        return values;
    }

    /// Returns a list of all column headers for temporal datasets (those that have a nT dimension)
    std::vector<std::string> TemporalColumnHeaders() const {
        std::vector<std::string> headers;
        topGroup_.VisitAllHeaders(headers, true);

        return headers;
    }
    ///Returns a list of all column values for temporal datasets (those that have a nT dimension)
    /// \param zmwIndex - the index of the ZMW to get values on
    /// \param t - the time index
    std::vector<std::string> TemporalColumnValues(uint32_t zmwIndex, uint32_t t) const {
        std::vector<std::string> values;
        topGroup_.VisitAllTemporalValues(zmwIndex, values, t);
        return values;
    }

    ZmwStatDataSet GetDataSet(const std::string& name) const
    {
        try
        {
            if (!hdf5file_) throw PBException("Can not create dataset on null HDF5file");
            ZmwStatDataSet dataset(*hdf5file_, name);
            return dataset;
        }
        catch (const H5::Exception& ex)
        {
            throw PBException("can't open dataset " + name +  " H5 Exception:" + ex.getDetailMsg());
        }
    }

    void SetDiagGroup(ZmwStatGroup& group)
    {
        diagGroup_ = &group;
    }

    void AddDiagDataset(ZmwStatDataSet& dataset)
    {
        diagDatasets_.push_back(&dataset);
    }

    /// Internal class used for assigning the group or dataset to manager
    /// This is a private class, even though it is exposed in the namespace for convenience.
    class Initer
    {
    public:
        Initer(ZmwStatGroup& group, SequelHDF5ObjectManager& manager)
        {
            manager.StartGroup(group);
        }
        Initer(ZmwStatDataSet& dataset, SequelHDF5ObjectManager& manager)
        {
            manager.AddDataSet(dataset);
        }
    };
    class IniterDiag
    {
    public:
        IniterDiag(ZmwStatGroup& group, SequelHDF5ObjectManager& manager)
        {
            manager.SetDiagGroup(group);
        }
        IniterDiag(ZmwStatDataSet& dataset, SequelHDF5ObjectManager& manager)
        {
            manager.AddDiagDataset(dataset);
        }
    };
    /// Internal class used for ending the group in the manager
    /// This is a private class, even though it is exposed in the namespace for convenience.
    class Closer
    {
    public:
        Closer(ZmwStatGroup& group, SequelHDF5ObjectManager& manager)
        {
            manager.EndGroup(group);
        }
    };

    class CloserDiag
    {
    public:
        CloserDiag(ZmwStatGroup& /*group*/, SequelHDF5ObjectManager& /*manager*/)
        {

        }
    };

    bool addDiagnostics_;
    std::string filename_; ///< the filename
    std::unique_ptr<H5::H5File> hdf5file_; ///< the H5 object
    ZmwStatGroup* currentGroup_{nullptr}; ///< this pointer changes as the ZmwStatFile is constructed, then it is never used again.
    std::stack<ZmwStatGroup*> groupStack_; ///< this stack allows the hierarchical traveral of the groups and is not used after construction of the ZmwStatFile
    ZmwStatGroup topGroup_{""}; ///< The top group node corresponding to "/" in the H5 file.
    ZmwStatGroup* diagGroup_{nullptr};
    std::vector<ZmwStatDataSet*> diagDatasets_;
};


/// This class abstracts the interface for reading and writing the sts.h5 file.
/// Construct it with just a filename for read access, and with the full dimensionality parameters for write accesss.
class ZmwStatsFile
{
public:

public:
    /// Read-only constructor. Filename must exist.
    ZmwStatsFile(const std::string& filename);

private:
    /// Write-only constructor. All values must be provided and can not be zero.
    ZmwStatsFile(const std::string& filename, uint32_t numHoles , uint32_t numAnalogs, uint32_t numFilters,
                 uint32_t numTimeslices, uint32_t binSize);

public:
    /// \param numHoles = the number of ZMWs (aka Holes)
    /// \param numAnalogs = the number of base analogs, usually 4 for normal biology
    /// \param numFilters = the number of detector channels or filters, 2 for Sequel (R/G) 4 for RS.
    /// \param timeInfo.first = number of frames
    /// \param timeInfo.second = binning size of frames
    struct NewFileConfig
    {
        uint32_t numHoles = 0;
        uint32_t numAnalogs = 0;
        uint32_t numFilters = 0;
        uint32_t numFrames = 0;
        uint32_t binSize = 0;
        bool     addDiagnostics = false;
        uint32_t mfBinSize = 0;
        uint32_t NumMFBlocks() const
        {
            return (numFrames + mfBinSize -1) / mfBinSize;
        }
        uint32_t LastMFBinFrames() const
        {
            uint32_t x = numFrames % mfBinSize;
            if (x == 0) x = mfBinSize;
            return x;
        }
        uint32_t NumTimeslices() const
        {
            return (numFrames + binSize -1) / binSize;
        }
        uint32_t LastBinFrames() const
        {
            uint32_t x = numFrames % binSize;
            if (x == 0) x = binSize;
            return x;
        }
    };
    /// Write-only constructor. All values must be provided and can not be zero.
    /// \param filename = the name of the new file to create
    /// \param config = a reference to a struct that contains all of the desired new file parameters.
    ZmwStatsFile(const std::string& filename, const NewFileConfig& config, bool dummyScanData=true);

    /// Destructor
    ~ZmwStatsFile();

    /// This constructs an empty ZmwStats struct for use in filling. The ZmwStats dimensions are declared at
    /// run time.
    ZmwStats GetZmwStatsTemplate() const;

    /// This reads the stats for one ZMW into a ZmwStats struct.
    /// \param index - the index of the ZMW to read. Must be zero based [0,nH)
    ZmwStats Get(uint32_t index) const;

    /// This writes the stats for one ZMW into a file
    /// \param index - the index of the ZMW to write. Must be zero based [0,nH)
    /// \param zmwStats - the stats for one ZMW
    void Set(hsize_t index, const ZmwStats& zmwStats);

    /// Writes /ScanData representation of experiment metadata.
    /// \param
    void WriteScanData(const Json::Value& experimentMetadata);

    uint32_t nH() const { return nH_; } ///< number of ZMWs
    uint32_t nA() const { return nA_; } ///< number of analogs
    uint32_t nF() const { return nF_; } ///< number of filters
    uint32_t nMF() const { return nMF_; } ///< number of MF time slices

    uint32_t NumFrames() const { return numFrames_;}
    uint32_t BinSize() const { return binSize_; }
    uint32_t mfBinSize() const { return mfBinSize_; }

    void CopyScanData(H5::Group& destScanData) const;

    std::string ChipLayoutName() const
    {
        std::string layoutName;
        auto LayoutName = objectManager_.topGroup_.ReadOnlyGroup()
            .openGroup("ScanData").openGroup("ChipInfo").openAttribute("LayoutName");
        LayoutName >> layoutName;
        return layoutName;
    }

    ZmwStatDataSet GetDataSet(const std::string& name) const
    {
        return objectManager_.GetDataSet(name);
    }
    /// See documentation for objectManager ColumnHeaders
    std::vector<std::string> ColumnHeaders() const { return objectManager_.ColumnHeaders(); }

    /// See documentation for objectManager ColumnValues
    std::vector<std::string> ColumnValues(uint32_t i) const { return objectManager_.ColumnValues(i); }

    /// See documentation for objectManager TemporalColumnHeaders
    std::vector<std::string> TemporalColumnHeaders() const { return objectManager_.TemporalColumnHeaders(); }

    /// See documentation for objectManager TemporalColumnValues
    std::vector<std::string> TemporalColumnValues(uint32_t zmw, uint32_t t) const { return objectManager_.TemporalColumnValues(zmw,t); }

    std::pair<uint32_t,uint32_t> GetCoordinate(int index) const;

    /// All of the stat datasets are defined in "ZmwStatsFileDefinition.h". Through the use of macros
    /// the list datasets can be used for creating H5 members, creating the ZmwStats struct members, and documentation.

    /// In this pass, each dataset is added as a member to ZmwStatsFile, and then added to the objectManager at construction time.

#define OVER_HQR true
#define NOT_OVER_HQR false
#define DECLARE_SEQUELH5_SUPPORT()        SequelHDF5ObjectManager objectManager_;
#define DECLARE_ZMWSTAT_ENUM(...)   SMART_ENUM(__VA_ARGS__);
#define DECLARE_ZMWSTAT_START_GROUP(name) \
         ZmwStatGroup name {#name}; \
         SequelHDF5ObjectManager::Initer _groupIniter_##name{name,objectManager_};
#define DECLARE_ZMWSTATDATASET_1D(prefix, name, type, units, description, hqr, dim0) \
         ZmwStatDataSet prefix##name{#name, type(), units, description, hqr, dim0 } ; \
         SequelHDF5ObjectManager::Initer _datasetIniter_##prefix##name{prefix##name,objectManager_};
#define DECLARE_ZMWSTATDATASET_2D(prefix, name, type, units, description, hqr, dim0, dim1) \
         ZmwStatDataSet prefix##name{#name, type(), units,description, hqr, #dim1, dim0, dim1} ; \
         SequelHDF5ObjectManager::Initer _datasetIniter_##prefix##name{prefix##name,objectManager_};
#define DECLARE_ZMWSTATDATASET_3D(prefix, name, type, units, description, hqr, dim0, dim1, dim2) \
         ZmwStatDataSet prefix##name{#name, type(), units ,description,hqr, #dim1, dim0,dim1,dim2} ; \
         SequelHDF5ObjectManager::Initer _datasetIniter_##prefix##name{prefix##name,objectManager_};
#define DECLARE_ZMWSTAT_END_GROUP(name)   \
         SequelHDF5ObjectManager::Closer _groupCloser_##name{name,objectManager_};

// These next set of macros declare diagnostic datasets to be added that are optional.

#define DECLARE_ZMWSTAT_START_GROUP_DIAGNOSTICS(name) \
        ZmwStatGroup name {#name, true}; \
        SequelHDF5ObjectManager::IniterDiag _groupIniter_##name{name,objectManager_};
#define DECLARE_ZMWSTATDATASET_1D_DIAGNOSTICS(prefix, name, type, units, description, hqr, dim0) \
         ZmwStatDataSet prefix##name{#name, type(), units, description, hqr, dim0, true} ; \
         SequelHDF5ObjectManager::IniterDiag _datasetIniter_##prefix##name{prefix##name,objectManager_};
#define DECLARE_ZMWSTATDATASET_2D_DIAGNOSTICS(prefix, name, type, units, description, hqr, dim0, dim1) \
         ZmwStatDataSet prefix##name{#name, type(), units,description, hqr, #dim1, dim0, dim1, true} ; \
         SequelHDF5ObjectManager::IniterDiag _datasetIniter_##prefix##name{prefix##name,objectManager_};
#define DECLARE_ZMWSTATDATASET_3D_DIAGNOSTICS(prefix, name, type, units, description, hqr, dim0, dim1, dim2) \
         ZmwStatDataSet prefix##name{#name, type(), units ,description,hqr, #dim1, dim0, dim1, dim2, true} ; \
         SequelHDF5ObjectManager::IniterDiag _datasetIniter_##prefix##name{prefix##name,objectManager_};
#define DECLARE_ZMWSTAT_END_GROUP_DIAGNOSTICS(name) \
         SequelHDF5ObjectManager::CloserDiag _groupCloser_##name{name,objectManager_};

#include "ZmwStatsFileDefinition.h"

#undef DECLARE_SEQUELH5_SUPPORT
#undef DECLARE_ZMWSTAT_ENUM
#undef DECLARE_ZMWSTAT_START_GROUP
#undef DECLARE_ZMWSTATDATASET_1D
#undef DECLARE_ZMWSTATDATASET_2D
#undef DECLARE_ZMWSTATDATASET_3D
#undef DECLARE_ZMWSTAT_END_GROUP
#undef DECLARE_ZMWSTAT_START_GROUP_DIAGNOSTICS
#undef DECLARE_ZMWSTATDATASET_1D_DIAGNOSTICS
#undef DECLARE_ZMWSTATDATASET_2D_DIAGNOSTICS
#undef DECLARE_ZMWSTATDATASET_3D_DIAGNOSTICS
#undef DECLARE_ZMWSTAT_END_GROUP_DIAGNOSTICS

private:
    void OpenForRead();
    void OpenForWrite(const NewFileConfig& config, bool dummyScanData);
    void Read(ZmwStatDataSet& ds);
    void Create(ZmwStatDataSet& ds);
    int WriterThread();

public:
    //experimental
    std::unique_ptr<ZmwStats> GetZmwStatsBuffer();
//    void SetBuffer(hsize_t index, std::unique_ptr<ZmwStats>&& zmw );
    void Buffer(bool flag=true) { isBuffered_ = flag; }
    bool IsBuffered() const {return isBuffered_;}
    void WriteBuffers(std::unique_ptr<std::vector<PacBio::Primary::ZmwStats> >&& bufferList);
    std::unique_ptr<std::vector<PacBio::Primary::ZmwStats> > GetZmwStatsBufferList()
    {
        return std::unique_ptr<std::vector<PacBio::Primary::ZmwStats>>(new std::vector<ZmwStats>);
    }

private:


private:
    std::string filename_;
    hsize_t nH_{0};
    hsize_t nA_{0};
    hsize_t nF_{0};
    hsize_t nMF_{0};
    hsize_t nAB_{0};
    hsize_t two_{2};
    hsize_t four_{4};
    hsize_t nFtriangle_{0};
    hsize_t numFrames_{0};
    hsize_t binSize_{0};
    hsize_t mfBinSize_{0};
    bool addDiagnostics_{false};
    mutable std::mutex mutex_;

    PacBio::IPC::ThreadSafeQueue<std::unique_ptr<std::vector<ZmwStats>>> workVectorQueue_;

    std::future<int> workerThread_;
    bool isBuffered_ = true;
    mutable std::vector<std::pair<uint32_t,uint32_t>> coordinates_;
};

}}
#endif //SEQUELACQUISITION_ZMWSTATSFILE_H
