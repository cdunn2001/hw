#ifndef SEQUELACQUISITION_SEQUELHDF5_H
#define SEQUELACQUISITION_SEQUELHDF5_H


// Copyright (c) 2016, Pacific Biosciences of California, Inc.
//
// All rights reserved.
//
// THIS SOFTWARE CONSTITUTES AND EMBODIES PACIFIC BIOSCIENCES' CONFIDENTIAL
// AND PROPRIETARY INFORMATION.
//
// Disclosure, redistribution and use of this software is subject to the
// terms and conditions of the applicable written agreement(s) between you
// and Pacific Biosciences, where "you" refers to you or your company or
// organization, as applicable.  Any other disclosure, redistribution or
// use is prohibited.
//
// THIS SOFTWARE IS PROVIDED BY PACIFIC BIOSCIENCES AND ITS CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL PACIFIC BIOSCIENCES OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
// OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
// ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// File Description:
//  C++ class wrapper to handle HDF5 objects for Sequel
//
// Programmer: Mark Lakata

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>
#include <string.h>

#include <atomic>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <memory>
#include <mutex>
#include <random>
#include <vector>
#include <list>

#include <boost/multi_array.hpp>
#include <boost/filesystem/path.hpp>

#include <json/json.h>

#include <H5public.h>

#include <pacbio/text/String.h>
#include <pacbio/Utilities.h>
#include <pacbio/utilities/ISO8601.h>

#include <pacbio/primary/HDF5cpp.h>

#include <pacbio/primary/SequelHDF5.h>

namespace PacBio {
namespace Primary {

#define uint8()      H5::PredType::STD_U8LE
#define int16()      H5::PredType::STD_I16LE
#define uint16()     H5::PredType::STD_U16LE
#define uint32()     H5::PredType::STD_U32LE
#define uint64()     H5::PredType::STD_U64LE
#define float32()    H5::PredType::IEEE_F32LE
#define float64()    H5::PredType::IEEE_F64LE
#define SeqH5string() H5::StrType(H5::PredType::C_S1, H5T_VARIABLE)

class SequelHDF5
{
public:
    static std::recursive_mutex& Mutex() { return mutex_; }

#define LOCK_HDF5()         std::lock_guard<decltype(SequelHDF5::Mutex())> lock(SequelHDF5::Mutex())
private:
    static std::recursive_mutex mutex_;
};

class SequelHDF5MembersStart
{
public:
    SequelHDF5MembersStart() { SequelHDF5::Mutex().lock(); }

    ~SequelHDF5MembersStart() { SequelHDF5::Mutex().unlock(); }
};

class SequelHDF5MembersEnd
{
public:
    SequelHDF5MembersEnd() { SequelHDF5::Mutex().unlock(); }

    ~SequelHDF5MembersEnd() { SequelHDF5::Mutex().lock(); }
};


template<typename _Get_TypeName>
const std::string& GetTypeName()
{
    static std::string name;

    if (name.empty())
    {
        const char* beginStr = "_Get_TypeName =";
        const size_t beginStrLen = 15; // Yes, I know...
        // But isn't it better than strlen()?

        size_t begin, length;
        name = __PRETTY_FUNCTION__;

        begin = name.find(beginStr) + beginStrLen + 1;
        length = name.find("]", begin) - begin;
        name = name.substr(begin, length);
    }

    return name;
}

#ifdef SUPPORT_MULTIDIMENSIONVECTOR
template<typename T>
  class MultidimensionVector
  {
  public:
      MultidimensionVector() {
          dims_[0] = 0;
          dims_[1] = 0;
          dims_[2] = 0;
          dims_[3] = 0;
          rank_ = 0;
      }
      explicit MultidimensionVector(int m) : MultidimensionVector(m,1) {rank_ = 1;}
      MultidimensionVector(int m, int n) : MultidimensionVector(m,n,1) {rank_ = 2;}
      MultidimensionVector(int m, int n, int o) : MultidimensionVector(m,n,o,1) {rank_ = 3;}
      MultidimensionVector(int m, int n, int o, int p)
      {
          rank_ = 4;
          dims_[0] = m;
          dims_[1] = n;
          dims_[2] = o;
          dims_[3] = p;
          values.resize(m*n*o*p);
      }
      MultidimensionVector(const std::vector<std::vector<T>>& inputVector)
      {
          dims_[0] = inputVector.size();
          dims_[1] = inputVector[0].size();
          dims_[2] = 1;
          dims_[3] = 1;
          rank_ = 2;
          values.resize(dims_[0]*dims_[1]);
          for(int i=0;i<inputVector.size();i++)
          {
              const std::vector<T>& innerVector = inputVector[i];
              for(int j=0;j<innerVector.size();j++)
              {
                  at(i,j) = innerVector[j];
              }
          }
      }

      size_t Size() const { return values.size(); }
      int Rank() const { return rank_;}
      T& Base() { return values[0]; }
      const T& Base() const { return values[0]; }
      void Resize(std::vector<hsize_t> dimensions)
      {
          if (dimensions.size() > 4)throw PBException("rank too high");
          dims_[0] = dimensions[0];
          dims_[1] = (dimensions.size()>1) ?dimensions[1] : 1;
          dims_[2] = (dimensions.size()>2) ?dimensions[2] : 1;
          dims_[3] = (dimensions.size()>3) ?dimensions[3] : 1;
          values.resize(dims_[0] * dims_[1] * dims_[2] * dims_[3]);
      }

      size_t Dimension(int rank) const
      {
          if (rank >= 4) throw PBException("rank too high");
          return dims_[rank];
      }

      T& at(int m)
      {
          return values[m];
      }
      T& at(int m, int n)
      {
          return values[m*dims_[1] + n];
      }
      T& at(int m, int n, int o)
      {
          return values[m*dims_[1]*dims_[2] + n*dims_[2] + o];
      }
      const T& at(int m) const
      {
          return values[m];
      }
      const T& at(int m, int n) const
      {
          return values[m*dims_[1] + n];
      }
      const T& at(int m, int n, int o) const
      {
          return values[m*dims_[1]*dims_[2] + n*dims_[2] + o];
      }
      template<typename U>
      void Import(int m, MultidimensionVector<U> v)
      {
          if (Rank() != v.Rank() +1)throw PBException("ohoh");
          if (Dimension(1) != v.Dimension(0)) throw PBException("ohoh");

          T* dst = &at(m,0,0);
          const U* src = &v.at(0,0);
          for(int i=0;i<v.Size();i++)
          {
              dst[i] = src[i];
          }
      }
  private:
      std::vector<T> values;
      int dims_[4];
      int rank_;
  };
#endif

template<typename T>
H5::DataType GetType()
{
    H5::DataType predType;


    if (typeid(T) == typeid(uint8_t))
    {
        predType = H5::PredType::NATIVE_UINT8;
    }
    else if (typeid(T) == typeid(int16_t))
    {
        predType = H5::PredType::NATIVE_INT16;
    }
    else if (typeid(T) == typeid(uint16_t))
    {
        predType = H5::PredType::NATIVE_UINT16;
    }
    else if (typeid(T) == typeid(int32_t))
    {
        predType = H5::PredType::NATIVE_INT32;
    }
    else if (typeid(T) == typeid(uint32_t))
    {
        predType = H5::PredType::NATIVE_UINT32;
    }
    else if (typeid(T) == typeid(uint64_t))
    {
        predType = H5::PredType::NATIVE_UINT64;
    }
    else if (typeid(T) == typeid(int64_t))
    {
        predType = H5::PredType::NATIVE_INT64;
    }
    else if (typeid(T) == typeid(float))
    {
        predType = H5::PredType::NATIVE_FLOAT;
    }
    else if (typeid(T) == typeid(double))
    {
        predType = H5::PredType::NATIVE_DOUBLE;
    }
    else if (typeid(T) == typeid(std::string))
    {
        predType = H5::PredType::NATIVE_DOUBLE; // wtf ?
    }
    else
    {
        throw PBException("Don't know how to convert \"" + GetTypeName<T>() + "\" to an HDF5 H5::PredType");
    }
    return predType;
}

inline H5::DataType GetDataType(const std::string& typeAnyCase)
{
    H5::DataType predType;
    std::string type = PacBio::Text::String::ToLower(typeAnyCase);
    if (type == "uint8")
    {
        predType = H5::PredType::STD_U8LE;
    }
    else if (type == "int16")
    {
        predType = H5::PredType::STD_I16LE;
    }
    else if (type == "uint16")
    {
        predType = H5::PredType::STD_U16LE;
    }
    else if (type == "int32")
    {
        predType = H5::PredType::STD_I64LE;
    }
    else if (type == "uint32")
    {
        predType = H5::PredType::STD_U32LE;
    }
    else if (type == "int64")
    {
        predType = H5::PredType::STD_I64LE;
    }
    else if (type == "uint64")
    {
        predType = H5::PredType::STD_U64LE;
    }
    else if (type == "float")
    {
        predType = H5::PredType::IEEE_F32LE;
    }
    else if (type == "double")
    {
        predType = H5::PredType::IEEE_F64LE;
    }
    else
    {
        throw PBException("Don't know how to convert \"" + type + "\" to an HDF5 H5::PredType");
    }
    return predType;
}


std::string GetName(const H5::DataSet& ds);

std::vector <hsize_t> GetDims(const H5::DataSet& ds);

size_t GetElements(const H5::DataSet& ds);

std::vector <hsize_t> GetDims(const H5::Attribute& attr);

size_t GetElements(const H5::Attribute& attr);

const H5::Attribute& operator>>(const H5::Attribute& attr, float& value);

const H5::Attribute& operator>>(const H5::Attribute& attr, double& value);

const H5::Attribute& operator>>(const H5::Attribute& attr, uint8_t& value);

const H5::Attribute& operator>>(const H5::Attribute& attr, uint16_t& value);

const H5::Attribute& operator>>(const H5::Attribute& attr, int32_t& value);

const H5::Attribute& operator>>(const H5::Attribute& attr, uint32_t& value);

const H5::Attribute& operator>>(const H5::Attribute& attr, uint64_t& value);

const H5::Attribute& operator>>(const H5::Attribute& attr, std::string& value);


template<typename T>
const H5::Attribute& operator>>(const H5::Attribute& attr, std::vector <T>& vector)
{
    LOCK_HDF5();
    vector.resize(GetElements(attr));
    attr.read(GetType<T>(), &vector[0]);
    return attr;
}

const H5::Attribute& operator>>(const H5::Attribute& attr, std::vector <std::string>& vector);

const H5::DataSet& operator>>(const H5::DataSet& ds, std::string& value);

const H5::DataSet& operator>>(const H5::DataSet& ds, std::vector<std::string>& array);


template<typename T>
const H5::DataSet& operator>>(const H5::DataSet& ds, T& value)
{
    LOCK_HDF5();
    ds.read(&value, GetType<T>());
    return ds;
}

template<typename T>
const H5::DataSet& operator>>(const H5::DataSet& ds, std::vector <T>& vector)
{
    LOCK_HDF5();
    vector.resize(GetElements(ds));
    ds.read(&vector[0], GetType<T>());
    return ds;
}

#ifdef SUPPORT_MULTIDIMENSIONVECTOR
template<typename T>
  const H5::DataSet& operator>>(const H5::DataSet& ds, MultidimensionVector<T>& value)
  {
      auto d = GetDims(ds);
      value.Resize(d);
      if (value.Size() != GetElements(ds) )
      {
          throw PBException("data set " + GetName(ds) + " and MultidimensionVector are not the same size. Data set has " +
                            std::to_string(GetElements(ds)) + " and MultidimensionVector has " +
                            std::to_string(value.Size()));
      }
      ds.read(&value.Base(), GetType<T>() );
      return ds;
  }
#endif

template<typename T, std::size_t NumDims>
const H5::DataSet& operator>>(const H5::DataSet& ds, boost::multi_array <T, NumDims>& value)
{
    LOCK_HDF5();
    auto d = GetDims(ds);
    if (d.size() != NumDims)
    {
        throw PBException("data set " + GetName(ds) + " has rank " + std::to_string(d.size()) +
                          " while multi_array has rank " + std::to_string(NumDims));
    }
    value.resize(d);
    if (value.num_elements() != GetElements(ds))
    {
        throw PBException("data set " + GetName(ds) + " and boost::multi_array are not the same size. Data set has " +
                          std::to_string(GetElements(ds)) + " and boost::multi_array has " +
                          std::to_string(value.num_elements()));
    }
    ds.read(value.data(), GetType<T>());
    return ds;
}


const H5::DataSet& operator>>(const H5::DataSet& ds, std::vector <std::string>& vector);


H5::Attribute& operator<<(H5::Attribute& attr, float value);

H5::Attribute& operator<<(H5::Attribute& attr, double value);

H5::Attribute& operator<<(H5::Attribute& attr, uint16_t value);

H5::Attribute& operator<<(H5::Attribute& attr, uint32_t value);

H5::Attribute& operator<<(H5::Attribute& attr, int value);

H5::Attribute& operator<<(H5::Attribute& attr, const uint64_t value);

H5::Attribute& operator<<(H5::Attribute& attr, const unsigned long long value);

H5::Attribute& operator<<(H5::Attribute& attr, const std::string& value);

H5::Attribute& operator<<(H5::Attribute& attr, const std::pair <uint32_t, uint32_t>& value);

template<typename T>
H5::Attribute& operator<<(H5::Attribute& attr, const std::vector <T>& value)
{
    LOCK_HDF5();
    if (value.size() != GetElements(attr))
        throw PBException("attribute " + attr.getName() + " vector are not the same size");
    attr.write(GetType<T>(), &value[0]);
    return attr;
}

H5::Attribute& operator<<(H5::Attribute& attr, const std::vector <std::string>& value);

H5::DataSet& operator<<(H5::DataSet& ds, const std::string& value);

inline H5::DataSet& operator<<(H5::DataSet& ds, const char* value)
{
    const std::string s(value);
    return ds << s;
}

template<typename T>
H5::DataSet& operator<<(H5::DataSet& ds, const T& value)
{
    LOCK_HDF5();
    ds.write(&value, GetType<T>());
    return ds;
}

template<typename T>
H5::DataSet& operator<<(H5::DataSet& ds, const std::vector <T>& value)
{
    LOCK_HDF5();
    if (value.size() != GetElements(ds))
    {
        throw PBException("data set " + GetName(ds) + " and vector are not the same size. Data set has " +
                          std::to_string(GetElements(ds)) + " and vector has " +
                          std::to_string(value.size()));
    }
    ds.write(&value[0], GetType<T>());
    return ds;
}

#ifdef SUPPORT_MULTIDIMENSIONVECTOR
template<typename T>
  H5::DataSet& operator<<(H5::DataSet& ds, const MultidimensionVector<T>& value)
  {
      if (value.Size() != GetElements(ds) )
      {
          throw PBException("data set " + GetName(ds) + " and MultidimensionVector are not the same size. Data set has " +
                            std::to_string(GetElements(ds)) + " and MultidimensionVector has " +
                            std::to_string(value.Size()));
      }
      ds.write(&value.Base(), GetType<T>() );
      return ds;
  }
#endif

template<typename T, std::size_t NumDims>
H5::DataSet& operator<<(H5::DataSet& ds, const boost::multi_array <T, NumDims>& value)
{
    LOCK_HDF5();
    auto d = GetDims(ds);
    if (d.size() != NumDims)
    {
        throw PBException("data set " + GetName(ds) + " has rank " + std::to_string(d.size()) +
                          " while boost::multi_array has rank " + std::to_string(NumDims));
    }
    if (value.num_elements() != GetElements(ds))
    {
        throw PBException("data set " + GetName(ds) + " and boost::multi_array are not the same size. Data set has " +
                          std::to_string(GetElements(ds)) + " and boost::multi_array has " +
                          std::to_string(value.num_elements()));
    }
    ds.write(value.data(), GetType<T>());
    return ds;
}


std::ostream& operator<<(std::ostream& s, const H5::Attribute& attr);

std::ostream& operator<<(std::ostream& s, const H5::DataSet& ds);

Json::Value& operator<<(Json::Value& s, const H5::Attribute& attr);

Json::Value& operator<<(Json::Value& s, const H5::DataSet& ds);

class DisableDefaultErrorHandler
{
public:
    DisableDefaultErrorHandler()
    {
        PushErrorHandler();
    }

    ~DisableDefaultErrorHandler()
    {
        PopErrorHandler();
    }

private:
    std::list<std::pair<H5E_auto_t, void*> > errorHandlerStack;
private:
    void PushErrorHandler()
    {
        LOCK_HDF5();
        H5E_auto_t func;
        void* data;
        herr_t e = H5Eget_auto(H5E_DEFAULT, &func, &data);
        if (e != 0) throw PBException("H5Eget_auto failed");
        e = H5Eset_auto(H5E_DEFAULT, NULL, NULL);
        if (e != 0) throw PBException("H5Eset_auto failed");
        errorHandlerStack.push_back(std::pair<H5E_auto_t, void*>(func, data));
    }

    void PopErrorHandler()
    {
        if (errorHandlerStack.size() > 0)
        {
            LOCK_HDF5();
            auto x = errorHandlerStack.back();
            herr_t e = H5Eset_auto(H5E_DEFAULT, x.first, x.second);
            if (e != 0) throw PBException("H5Eset_auto failed");
            errorHandlerStack.pop_back();
        }
    }
};


class PBFile
{
public:
    static void MakePath(H5::H5File& hdf5file, const std::string& path)
    {
        boost::filesystem::path p = path;
        boost::filesystem::path q;
        for (auto z : p)
        {
            q = q / z;
            if (!PathExists(hdf5file, q.string()))
            {
                hdf5file.createGroup(q.string());
            }
        }
        if (! PathExists(hdf5file,path))
        {
            throw PBException("Could not create " + path);
        }
    }

    static bool PathExists( H5::H5File& hdf5file, const std::string& path)
    {
        DisableDefaultErrorHandler ddeh;

        H5G_stat_t stat;
        herr_t s = H5Gget_objinfo(hdf5file.getLocId(), path.c_str(), false, &stat);
        return s == 0;
    }
};

class PBDataSpace
{
public:
    static std::pair<std::vector<hsize_t>,std::vector<hsize_t>> GetDimensions(const H5::DataSpace& ds)
    {
        std::vector<hsize_t> dims(ds.getSimpleExtentNdims());
        std::vector<hsize_t> maxdims(ds.getSimpleExtentNdims());
        ds.getSimpleExtentDims(dims.data(), maxdims.data());
        return std::make_pair(dims, maxdims);
    }

    static H5::DataSpace Create(std::vector<hsize_t> dims)
    {
        H5::DataSpace ds(dims.size(), dims.data());
        return ds;
    }
};

class PBDataSet
{
public:
    H5::DataSet& DataSet() { return dataSet_; }
    const H5::DataSet& DataSet() const { return dataSet_; }
    std::vector <hsize_t> GetDims() const { return PacBio::Primary::GetDims(dataSet_);}

public:
    /// Reads a attribute from the current dataset
    template<typename T>
    T ReadAttribute(const char* name) const
    {
        auto attr = dataSet_.openAttribute(name);
        T s;
        attr >> s;
        return s;
    }

    /// Creates an attribute in the current dataset
    template<typename T>
    void CreateAttribute(const char* name, H5::DataType S, T value)
    {
        H5::DataSpace scalar;
        auto attr = dataSet_.createAttribute(name, S , scalar);
        attr << value;
    }

    template<typename T>
    static void CopyAttribute(const PBDataSet& source, PBDataSet& destination, const char* name )
    {
        T temporary;

        auto attr = source.dataSet_.openAttribute(name);
        attr >> temporary;

        H5::DataType s = attr.getDataType();
        H5::DataSpace scalar;
        auto attrWrite = destination.dataSet_.createAttribute(name, s , scalar);
        attrWrite << temporary;
    }

    bool Exists() const { auto id = dataSet_.getId(); return id !=0; }

protected:
    H5::DataSet dataSet_; /// the actual H5 dataset object

};

/// Class that has static members that tries to make the HDF5 library more accessible.
class PBHDF5
{
public:

    /// Increase the first dimension of the dataset by one.
    /// \returns the index of the new record of the dataset. This index can be used
    /// as the offset for the hyperslab.
    static hsize_t AddTo0Dim(H5::DataSet& ds)
    {
        auto dims = GetDims(ds);
        hsize_t oldOffset = dims[0];
        dims[0]++;
        ds.extend(dims.data());
        return oldOffset;
    }

    /// Used for datasets that are 2 dimensional. The data is viewed as a 1 dimensional array
    /// of 1 dimensional arrays or records.  To add a new record (of size dim0), create
    /// an instance of this class, fill it with the rec[] method, and call Append(ds,record).
    template<typename T>
    class Record1D
    {
    public:
        Record1D(int dim0) :
                vec_(dim0),
                dim_(dim0) {}

        T& operator[](int i) { return vec_[i]; }

        const void* CData() const { return vec_.data(); }

        void* Data() { return vec_.data(); }

        hsize_t Size() const { return dim_; }

    private:
        std::vector<T> vec_;
        int dim_;
    };

    /// Used for datasets that are 3 dimensional. The data is viewed as a 1 dimensional array
    /// of 2 dimensional arrays or records.  To add a new record (of size dim0xdim1), create
    /// an instance of this class, fill it with the rec(i,j) method,
    /// and call Append(ds,record).
    template<typename T>
    class Record2D
    {
    public:
        Record2D(int dim0, int dim1) :
                vec_(dim0 * dim1),
                dim0_(dim0),
                dim1_(dim1) {}

        T& operator()(int i, int j) { return vec_[i * dim1_ + j]; }

        const void* CData() const { return vec_.data(); }

        void* Data() { return vec_.data(); }

        hsize_t Size0() const { return dim0_; }

        hsize_t Size1() const { return dim1_; }

    private:
        std::vector<T> vec_;
        int dim0_;
        int dim1_;
    };

    /// appends a string value to a variable string dataset
    static void Append(H5::DataSet& ds, const std::string& value)
    {
        hsize_t pos = AddTo0Dim(ds);
        H5::DataSpace fspace1 = ds.getSpace();
        hsize_t hs_size[1];
        hs_size[0] = 1;
        H5::DataSpace memFrame(1, hs_size);

        hsize_t offset[1];
        offset[0] = pos;

        fspace1.selectHyperslab(H5S_SELECT_SET, hs_size, offset);
        H5::StrType datatype(H5::PredType::C_S1, H5T_VARIABLE);
        std::string timestamp = value;
        std::vector<const char*> pointers(1);
        pointers[0] = timestamp.c_str();
        ds.write(pointers.data(), datatype, memFrame,fspace1);
    }


    /// appends a 1 dim record to a dataset. See Record1D for more help.
    template<typename T>
    static void Append(H5::DataSet& ds, const Record1D<T>& record)
    {
        hsize_t pos = AddTo0Dim(ds);
        H5::DataSpace fspace1 = ds.getSpace();
        hsize_t hs_size[2];
        hs_size[0] = 1;
        hs_size[1] = record.Size();
        H5::DataSpace memFrame(2, hs_size);

        hsize_t offset[2];
        offset[0] = pos;
        offset[1] = 0;

        fspace1.selectHyperslab(H5S_SELECT_SET, hs_size, offset);
        ds.write(record.CData(), GetType<T>(), memFrame, fspace1);
    }

    /// gets a 1 dim string at index.
    static void Get(H5::DataSet& ds, uint32_t index, std::string& record)
    {
#if 1
        // dumb way. Read entire vector, then pull out the desired one. Works.
        std::vector<std::string> values;
        ds >> values;
        record = values.at(index);
#else
        // this way doesn't work :(

//      LOCK_HDF5();

        hsize_t hs_size[1];
        hs_size[0] = 1;
        H5::DataSpace memFrame(1, hs_size);

        hsize_t offset[1];
        offset[0] = index;

        H5::DataSpace fspace1 = ds.getSpace();
        fspace1.selectHyperslab(H5S_SELECT_SET, hs_size, offset);

        H5::StrType stringTypeX(H5::PredType::C_S1, H5T_VARIABLE);
        // first load the string as a custom struct, because empty strings are not null-terminated.
        struct
        {
            uint32_t ptr;
            uint32_t other[3];
        } var;
        ds.read(&var, stringTypeX, memFrame, fspace1);

        if (var.ptr == 0)
        {
            // this was an empty string
            record = "";
        }
        else
        {
            // this was not an empty string, read it again into H5 string object
            H5std_string s;
            ds.read(s, stringTypeX, memFrame, fspace1);
            record = s;
        }
#endif
    }


    /// Gets a 1-dim record at index from dataset. See Record1D for usage.
    template<typename T>
    static void Get(H5::DataSet& ds, uint32_t index, Record1D<T>& record)
    {
        H5::DataSpace fspace1 = ds.getSpace();
        hsize_t hs_size[2];
        hs_size[0] = 1;
        hs_size[1] = record.Size();
        H5::DataSpace memFrame(2, hs_size);

        hsize_t offset[3];
        offset[0] = index;
        offset[1] = 0;
        offset[2] = 0;

        fspace1.selectHyperslab(H5S_SELECT_SET, hs_size, offset);
        ds.read(record.Data(), GetType<T>(), memFrame, fspace1);
    }


    /// appends a 2-dim record to the dataset. See Record2D for usage.
    template<typename T>
    static void Append(H5::DataSet& ds, const Record2D<T>& record)
    {
        hsize_t pos = AddTo0Dim(ds);
        hsize_t hs_size[3];
        hs_size[0] = 1;
        hs_size[1] = record.Size0();
        hs_size[2] = record.Size1();
        H5::DataSpace memFrame(3, hs_size);

        hsize_t offset[3];
        offset[0] = pos;
        offset[1] = 0;
        offset[2] = 0;

        H5::DataSpace fspace1 = ds.getSpace();
        fspace1.selectHyperslab(H5S_SELECT_SET, hs_size, offset);
        ds.write(record.CData(), GetType<T>(), memFrame, fspace1);
    }

    template<typename T>
    static void Get(H5::DataSet& ds, uint32_t index, Record2D<T>& record)
    {
        hsize_t hs_size[3];
        hs_size[0] = 1;
        hs_size[1] = record.Size0();
        hs_size[2] = record.Size1();
        H5::DataSpace memFrame(3, hs_size);

        hsize_t offset[3];
        offset[0] = index;
        offset[1] = 0;
        offset[2] = 0;

        H5::DataSpace fspace1 = ds.getSpace();
        fspace1.selectHyperslab(H5S_SELECT_SET, hs_size, offset);
        ds.read(record.Data(), GetType<T>(), memFrame, fspace1);
    }
};

}}

#endif //SEQUELACQUISITION_SEQUELHDF5_H
