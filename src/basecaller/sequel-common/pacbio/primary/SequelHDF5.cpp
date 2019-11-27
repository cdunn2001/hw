// Copyright (c) 2014, Pacific Biosciences of California, Inc.
//
// All rights reserved.
//
// THIS SOFTWARE CONSTITUTES AND EMBODIES PACIFIC BIOSCIENCES<92> CONFIDENTIAL
// AND PROPRIETARY INFORMATION.
//
// Disclosure, redistribution and use of this software is subject to the
// terms and conditions of the applicable written agreement(s) between you
// and Pacific Biosciences, where <93>you<94> refers to you or your company or
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
//  SequelMovie functions that could not be put into the class definitions
//  and thus could not be part of the *.h file.
//
// Programmer: Mark Lakata

/************************************************************

 This file is intended for use with HDF5 Library version 1.8

 ************************************************************/

#include <pacbio/primary/SequelHDF5.h>

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <sstream>

#include <pacbio/logging/Logger.h>
#include <pacbio/text/String.h>
#include <pacbio/process/ProcessBase.h>

#ifdef WIN32
#else
#include <unistd.h>

#endif

#include <fstream>

using namespace std;

namespace PacBio {
namespace Primary {

  std::recursive_mutex SequelHDF5::mutex_;

  const H5::Attribute& operator>>(const H5::Attribute& attr, float& value)
  {
      LOCK_HDF5();
      attr.read(H5::PredType::NATIVE_FLOAT, &value);
      return attr;
  }

  const H5::Attribute& operator>>(const H5::Attribute& attr, double& value)
  {
      LOCK_HDF5();
      attr.read(H5::PredType::NATIVE_DOUBLE, &value);
      return attr;
  }

  const H5::Attribute& operator>>(const H5::Attribute& attr, uint32_t& value)
  {
      LOCK_HDF5();
      attr.read(H5::PredType::NATIVE_UINT32, &value);
      return attr;
  }

  const H5::Attribute& operator>>(const H5::Attribute& attr, int32_t& value)
  {
      LOCK_HDF5();
      attr.read(H5::PredType::NATIVE_INT32, &value);
      return attr;
  }

  const H5::Attribute& operator>>(const H5::Attribute& attr, uint8_t& value)
  {
        LOCK_HDF5();
        attr.read(H5::PredType::NATIVE_UINT8, &value);
        return attr;
  }

  const H5::Attribute& operator>>(const H5::Attribute& attr, uint16_t& value)
  {
      LOCK_HDF5();
      attr.read(H5::PredType::NATIVE_UINT16, &value);
      return attr;
  }

  const H5::Attribute& operator>>(const H5::Attribute& attr, uint64_t& value)
  {
      LOCK_HDF5();
      attr.read(H5::PredType::NATIVE_UINT64, &value);
      return attr;
  }
  const H5::Attribute& operator>>(const H5::Attribute& attr, int64_t& value)
  {
      LOCK_HDF5();
      attr.read(H5::PredType::NATIVE_INT64, &value);
      return attr;
  }


  const H5::Attribute& operator>>(const H5::Attribute& attr, std::string& value)
  {
      LOCK_HDF5();

      H5::StrType stringTypeX(H5::PredType::C_S1, H5T_VARIABLE);
      stringTypeX.setCset( H5T_CSET_ASCII);

      struct
      {
          uint64_t ptr;
          uint64_t other;
      } var;

      attr.read(stringTypeX, &var);
      if (var.ptr == 0)
      {
          value = "";
      }
      else
      {
          free((void*) var.ptr);
          H5std_string s;
          attr.read(stringTypeX, s);
          value = s;
      }

      return attr;
  }

  const H5::Attribute& operator>>(const H5::Attribute& attr0, std::vector<std::string>& array)
  {
      LOCK_HDF5();
      H5E_auto2_t func;
      void* client_data;
      H5::Exception::getAutoPrint(func, &client_data);
      H5::Exception::dontPrint();

      try
      {
          hid_t attr = attr0.getId();

          hid_t atype = H5Aget_type(attr);
          hid_t aspace = H5Aget_space(attr);
          int rank = H5Sget_simple_extent_ndims(aspace);
          if (rank != 1) throw PBException("Attribute " + attr0.getName() + " is not a string array");

          hsize_t sdim[1];
          herr_t ret = H5Sget_simple_extent_dims(aspace, sdim, NULL);
          if (ret < 0)
          {
              throw PBException("H5Sget_simple_extent_dims failed");
          }
          size_t size = H5Tget_size (atype);
          if (size != sizeof(void*))
          {
              throw PBException("Internal inconsistency. Expected pointer size element");
          }

          // HDF5 only understands vector of char* :-(
          std::vector<char*> arr_c_str(sdim[0]);

//          H5::StrType stringType(H5::PredType::C_S1, H5T_VARIABLE);
          attr0.read(SeqH5string(), arr_c_str.data());
          array.resize(sdim[0]);
          for(hsize_t i=0;i<sdim[0];i++)
          {
              // std::cout << i << "=" << arr_c_str[i] << std::endl;
              array[i] = arr_c_str[i];
              free(arr_c_str[i]);
          }

      }
      catch (H5::Exception& err)
      {
          H5::Exception::setAutoPrint(func, client_data);
          throw std::runtime_error(string("HDF5 Error in " )
                                    + err.getFuncName()
                                    + ": "
                                    + err.getDetailMsg());


      }
      H5::Exception::setAutoPrint(func, client_data);
      return attr0;
  }

  const H5::DataSet& operator>>(const H5::DataSet& ds, std::string& value)
  {
      LOCK_HDF5();
      H5::StrType stringTypeX(H5::PredType::C_S1, H5T_VARIABLE);
      struct
      {
          uint32_t ptr;
          uint32_t other[3];
      } var;
      ds.read(&var, stringTypeX);
#if 0
	std::cout << "oper>> dataset" << var.ptr << std::endl;
#endif

      if (var.ptr == 0)
      {
          value = "";
      }
      else
      {
          H5std_string s;
          ds.read(s, stringTypeX);

          value = s;
      }
      return ds;
  }

const H5::DataSet& operator>>(const H5::DataSet& ds, std::vector<std::string>& array)
{
      // couldn't figure out how to get the C++ interface to work, so
      // it is easier to dip to the C interface to get this to work.
/*
* Get dataspace and allocate memory for read buffer.
*/
    herr_t       space = H5Dget_space(ds.getId());
    hsize_t     dims[10] = {0};
    int ndims = H5Sget_simple_extent_dims(space, dims, NULL);
    if (ndims != 1)
    {
        throw PBException("operator>> array can't be used on dataset with ndims=" + std::to_string(ndims));
    }
    char** rdata = (char**) malloc(dims[0] * sizeof(char*));

/*
 * Create the memory datatype.
 */
    hid_t  memtype = H5Tcopy(H5T_C_S1);
    herr_t status = H5Tset_size(memtype, H5T_VARIABLE);
    if (status < 0) throw PBException("H5Tset_size failed");

/*
 * Read the data.
 */
    status = H5Dread(ds.getId(), memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, rdata);
    if (status < 0) throw PBException("H5Dread failed");

/*
 * Output the data to the screen.
 */
    array.resize(dims[0]);
    for (hsize_t i = 0; i < dims[0]; i++)
    {
        // std::cout << "["<<i<<"]: "<<rdata[i] << std::endl;
        array[i] = rdata[i];
    }

/*
 * Close and release resources.  Note that H5Dvlen_reclaim works
 * for variable-length strings as well as variable-length arrays.
 * Also note that we must still free the array of pointers stored
 * in rdata, as H5Tvlen_reclaim only frees the data these point to.
 */
    status = H5Dvlen_reclaim(memtype, space, H5P_DEFAULT, rdata);
    if (status < 0) throw PBException("H5Dvlen_reclaim failed");
    free(rdata);

    return ds;
}


  H5::Attribute& operator<<(H5::Attribute& attr, float value)
  {
      LOCK_HDF5();
      attr.write(H5::PredType::NATIVE_FLOAT, &value);
      return attr;
  }

  H5::Attribute& operator<<(H5::Attribute& attr, double value)
  {
      LOCK_HDF5();
      attr.write(H5::PredType::NATIVE_DOUBLE, &value);
      return attr;
  }

  H5::Attribute& operator<<(H5::Attribute& attr, uint16_t value)
  {
      LOCK_HDF5();
      attr.write(H5::PredType::NATIVE_UINT16, &value);
      return attr;
  }

  H5::Attribute& operator<<(H5::Attribute& attr, uint32_t value)
  {
      LOCK_HDF5();
      attr.write(H5::PredType::NATIVE_UINT32, &value);
      return attr;
  }

  H5::Attribute& operator<<(H5::Attribute& attr, int value)
  {
      LOCK_HDF5();
      attr.write(H5::PredType::NATIVE_INT, &value);
      return attr;
  }

  H5::Attribute& operator<<(H5::Attribute& attr, const uint64_t value)
  {
      LOCK_HDF5();
      attr.write(H5::PredType::NATIVE_UINT64, &value);
      return attr;
  }

#ifndef WIN32

  H5::Attribute& operator<<(H5::Attribute& attr, const unsigned long long value)
  {
      LOCK_HDF5();
      attr.write(H5::PredType::NATIVE_UINT64, &value);
      return attr;
  }

#endif

  H5::Attribute& operator<<(H5::Attribute& attr, const std::string& value)
  {
      LOCK_HDF5();
   //   H5::StrType stringType(H5::PredType::C_S1, H5T_VARIABLE);

#if 0
      const char* s = value.c_str();
      H5Awrite(attr.getId(), stringType.getId(), &s);
      std::cout << "wrote:" << s << std::endl;
      char* buffer;
      H5Aread(attr.getId(), stringType.getId(), &buffer);
      std::cout << "read:" << buffer << std::endl;
      free(buffer);
#else
      attr.write(SeqH5string(), value);
#endif
      return attr;
  }

  H5::Attribute& operator<<(H5::Attribute& attr, const std::pair<uint32_t, uint32_t>& value)
  {
      LOCK_HDF5();
      attr.write(H5::PredType::NATIVE_UINT32, &value);
      return attr;
  }

  H5::Attribute& operator<<(H5::Attribute& attr, const std::vector<std::string>& strings)
  {
      LOCK_HDF5();
      H5E_auto2_t func;
      void* client_data;
      H5::Exception::getAutoPrint(func, &client_data);
      H5::Exception::dontPrint();

      try
      {
          // HDF5 only understands vector of char* :-(
          std::vector<const char*> arr_c_str;
          for (unsigned ii = 0; ii < strings.size(); ++ii)
          {
              arr_c_str.push_back(strings[ii].c_str());
          }

          //
          //  one dimension
          //
          hsize_t     str_dimsf[1] {arr_c_str.size()};
          H5::DataSpace   dataspace(1, str_dimsf);

          // Variable length string
          H5::StrType datatype(H5::PredType::C_S1, H5T_VARIABLE);
          attr.write(datatype, arr_c_str.data());
      }
      catch (H5::Exception& err)
      {
          H5::Exception::setAutoPrint(func, client_data);
          throw PBException(std::string("Can't write vector<string> to ") + attr.getName()
                                   + err.getFuncName()
                                   + ": "
                                   + err.getDetailMsg());


      }
      H5::Exception::setAutoPrint(func, client_data);
      return attr;
  }


  H5::DataSet& operator<<(H5::DataSet& ds, const std::string& value)
  {
      LOCK_HDF5();
//      H5::StrType stringType(H5::PredType::C_S1, H5T_VARIABLE);
      ds.write(value, SeqH5string());
      return ds;
  }


  std::ostream& operator<<(std::ostream& s, const H5::Attribute& attr)
  {
      LOCK_HDF5();
      if (attr.getId() == 0)
      {
          if (s.precision() >= 100)
          {
              s << "<not loaded>" << std::endl;
          }
      }
      else
      {
          if (s.precision() >= 100)
          {
              std::string n;
              n = attr.getName();
              s << n << ":\t";
          }
          switch (attr.getTypeClass())
          {
          case H5T_STRING:
          {
              std::string t;
              attr >> t;
              if (s.precision() >= 100)
              {
                  s << '"' << t << '"';
              }
              else
              {
                  s << t;
              }
          }
              break;
          case H5T_INTEGER:
          {
              if (attr.getIntType().getSign() == H5T_SGN_NONE)
              {
                  uint64_t value;
                  attr >> value;
                  if (s.precision() >= 100)
                  {
                      s << value << std::hex << " (0x" << value << ")" << std::dec;
                  }
                  else
                  {
                      s << value;
                  }
              }
              else if (attr.getIntType().getSign() == H5T_SGN_2)
              {
                  int64_t value;
                  attr >> value;
                  if (s.precision() >= 100)
                  {
                      s << value << std::hex << " (0x" << value << ")" << std::dec;
                  }
                  else
                  {
                      s << value;
                  }
              }
              else
              {
                  throw PBException("Weird integer type");
              }
          }
              break;
          case H5T_FLOAT:
          {
              double value;
              attr >> value;
              s << value;
          }
              break;
          default:
              s << "(not implemented)";
              break;
          }

          if (s.precision() >= 100)
          {
              s << std::endl;
          }
      }
      return s;
  }

Json::Value& operator<<(Json::Value& s, const H5::Attribute& attr)
{
    LOCK_HDF5();
    if (attr.getId() == 0)
    {
        s.setComment(std::string("// attribute skipped because not present"),Json::CommentPlacement::commentBefore);
    }
    else
    {
        std::string n;
        n = attr.getName();

        PBLOG_DEBUG << "creating JSON for attribute : " << n << " type:" << attr.getTypeClass();

        switch (attr.getTypeClass())
        {
        case H5T_STRING:
        {
            std::string t;
            attr >> t;
            s[n] = t;
        }
            break;
        case H5T_INTEGER:
        {
            if (attr.getIntType().getSign() == H5T_SGN_NONE)
            {
                uint64_t value;
                attr.read(H5::PredType::NATIVE_UINT64, &value);
                s[n] = value;
            }
            else if (attr.getIntType().getSign() == H5T_SGN_2)
            {
                int64_t value;
                attr.read(H5::PredType::NATIVE_INT64, &value);
                s[n] = value;
            }
            else
            {
                throw PBException("Weird integer type");
            }
        }
            break;
        case H5T_FLOAT:
        {
            double value;
            attr >> value;
            s[n] = value;
        }
            break;
        default:
            PBLOG_WARN << "HDF5 type for " << n << " not supported";
            s[n] = Json::Value::null;
            break;
        }

        PBLOG_DEBUG << "done with JSON for attribute : " << n << " got:" << s[n];

    }
    return s;
}

Json::Value& operator<<(Json::Value& , const H5::DataSet& )
{
    throw PBException("not implemented");
}

std::string GetName(const H5::DataSet& ds)
  {
      LOCK_HDF5();
      auto len = H5Iget_name(ds.getId(), NULL, 0);
      std::vector<char> buffer(len);
      H5Iget_name(ds.getId(), buffer.data(), len + 1);
      std::string n(buffer.data());
      return n;
  }


  std::vector<hsize_t> GetDims(const H5::DataSet& ds)
  {
      LOCK_HDF5();
      std::vector<hsize_t> dims;
      auto framesSpace = ds.getSpace();
      int rank = framesSpace.getSimpleExtentNdims();
      dims.resize(rank);

      framesSpace.getSimpleExtentDims(&dims[0]);
      return dims;
  }

  size_t GetElements(const H5::DataSet& ds)
  {
      LOCK_HDF5();
      auto v = GetDims(ds);
      size_t s = 1;
      for (auto&& x : v)
      {
          s *= x;
      }
      return s;
  }

  std::vector<hsize_t> GetDims(const H5::Attribute& attr)
  {
      LOCK_HDF5();
      std::vector<hsize_t> dims;
      auto framesSpace = attr.getSpace();
      int rank = framesSpace.getSimpleExtentNdims();
      dims.resize(rank);

      framesSpace.getSimpleExtentDims(&dims[0]);
      return dims;
  }

  size_t GetElements(const H5::Attribute& attr)
  {
      LOCK_HDF5();
      auto v = GetDims(attr);
      size_t s = 1;
      for (auto&& x : v)
      {
          s *= x;
      }
      return s;
  }

  std::ostream& operator<<(std::ostream& s, const H5::DataSet& ds)
  {
      LOCK_HDF5();
      if (ds.getId() == 0)
      {
          if (s.precision() >= 100)
          {
              s << "<not loaded>" << std::endl;
          }
      }
      else
      {
          if (s.precision() >= 100)
          {
              std::string n = GetName(ds);

              s << n << ":\t";
          }
          switch (ds.getTypeClass())
          {
          case H5T_STRING:
          {
              std::string t;
              ds >> t;
              if (s.precision() >= 100)
              {
                  s << '"' << t << '"';
              }
              else
              {
                  s << t;
              }
          }
              break;
#if 0
        case H5T_INTEGER:
            {
                uint64_t value;
                ds >> value;
                s << value << std::hex << " (0x" << value << ")" << std::dec;
            }
            break;
        case H5T_FLOAT:
            {
                double value;
                ds >> value;
                s << value ;
            }
            break;
#endif
          default:
              s << "(not implemented)";
              break;
          }

          s << std::endl;
      }
      return s;
  }

}} // namespace
