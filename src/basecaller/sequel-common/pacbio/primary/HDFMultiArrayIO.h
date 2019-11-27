#ifndef Sequel_Basecaller_Common_HDFMultiArrayIO_H_
#define Sequel_Basecaller_Common_HDFMultiArrayIO_H_

// Copyright (c) 2010-2015, Pacific Biosciences of California, Inc.
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
// Description:
/// \file HDFMultiArrayIO.h
/// \brief Defines class HDFMultiArrayIO.

#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <boost/multi_array.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include "HDF5cpp.h"
#include "HDF5_type_traits.h"

namespace PacBio {
namespace Primary {

// TODO: Some functions don't involve Boost MultiArray. Might want to move them to a difference class/file/etc.

/// HDFMultiArrayIO represents an HDF5 file and provides methods to read data
/// from and write data to the file using Boost multiarrays.
/// \see http://www.hdfgroup.org/HDF5/
/// \see http://www.boost.org/doc/libs/1_55_0/libs/multi_array/doc/index.html
class HDFMultiArrayIO
{
public:     // Enumeration
    enum AccessMode {
        ReadOnly    = 0u,
        ReadWrite   = 1u,
        WriteOver   = 2u,   // Clobbers pre-existing file.
        Create      = 4u    // Fails if file already exists.
    };

public:		// Static "factory" methods
    /// \brief Creates a read-only handle to a file.
    /// \detail The caller owns the object (i.e., is responsible to delete it
    /// when it's done with it).
    static std::unique_ptr<const HDFMultiArrayIO> CreateReader(const std::string fileName)
    { return std::unique_ptr<const HDFMultiArrayIO>(new HDFMultiArrayIO (fileName, ReadOnly)); }

    static std::unique_ptr<HDFMultiArrayIO> CreateWriter(const std::string fileName)
    { return std::unique_ptr<HDFMultiArrayIO>(new HDFMultiArrayIO (fileName, Create)); }

public:		// Structors
    /// Open HDF5 file file reading or writing according to fileMode.
    HDFMultiArrayIO (const std::string& fileName, AccessMode mode);

	// No copying.

	~HDFMultiArrayIO(void);

public:		// Read-only operations
	std::string FileName() const
	{ return fileName_; }

	// TODO: Eliminate attribute option in Dimensionality, Shape, and Read.

    /// The dimensionality (a.k.a. rank) of the specified data set or attribute.
    /// If no attribute name is specified, the dimensionality of the data set is
    /// returned. \a datasetName should be "absolute path" to the data set.
    int Dimensionality(
            const std::string& datasetName,
            const std::string& attrName = std::string()
            ) const;

    /// Returns the sizes of all dimensions of specified data set or attribute.
    /// If no attribute name is specified, the shape of the data set is returned.
    /// \a datasetName should be "absolute path" to the data set.
    /// Size of returned vector is equal to dimensionality of the data set.
    std::vector<int> Shape (
            const std::string& datasetName,
            const std::string& attrName = std::string()
            ) const;

    /// Reads the entire specified data set or attribute and return the data in
    /// a multi_array.
    /// If no attribute name is specified, the data set is returned.
    /// \a datasetName should be "absolute path" to the data set.
    template<typename ElementType, size_t NumDims>
    boost::multi_array<ElementType, NumDims>
    Read(const std::string& datasetName, const std::string& attrName = std::string()) const;

    /// Reads a data set of strings.
    std::vector<std::string> ReadStringDataset(const std::string& datasetName) const;

public:		// Read attributes.
	/// \brief Read a scalar (single element) attribute of a group or data set.
    /// \details To read a group attribute, omit the \a datasetName argument.
	template <typename T>
	T ReadScalarAttribute(
		const std::string& attributeName,
		const std::string& groupName,
		const std::string& datasetName = std::string()) const;

    /// \brief Read a 1D array attribute of a group or data set.
    /// \details Scalar attributes are returned as one-element vectors.
    /// To read a group attribute, omit the \a datasetName argument.
	template <typename T>
	std::vector<T> ReadVectorAttribute(
		const std::string& attributeName,
		const std::string& groupName,
		const std::string& datasetName = std::string()) const;

    template<typename ElementType, size_t NumDims>
	boost::multi_array<ElementType, NumDims>
	ReadChunkedData(const std::string& datasetName,
                    const unsigned long long* offset,
					std::vector<int> count) const;

    template<typename ElementType, size_t NumDims>
	void
	ReadChunkedData(const std::string& datasetName,
					ElementType* dataPtr,
					const unsigned long long* offset,
					std::vector<int> count) const;


public:		// Write operations

    template <typename ElementType, size_t NumDims>
    void Write (const std::string& datasetName,
                const boost::multi_array<ElementType, NumDims>& data);

    template <typename ElementType>
    void Write (const std::string& datasetName,
                const std::vector<ElementType>& data);
    
    template <typename T>
    void WriteScalarAttribute(
            const T& attributeValue,
            const std::string& attributeName,
            const std::string& groupName,
            const std::string& datasetName = std::string());

    void WriteStringAttribute(
            const std::string& attributeValue,
            const std::string& attributeName,
            const std::string& groupName,
            const std::string& datasetName = std::string());

    void CreateGroup(const std::string& groupName);


private:  // Utility methods
    std::unique_ptr<H5::H5Object> getH5Object(const std::string& groupName,
                                              const std::string& datasetName = std::string()) const;

    std::string getH5ObjectName(const std::string& groupName,
                                const std::string& datasetName = std::string()) const;

private:    // Static data
    // TODO: This mutex should probably be "more global."
    static std::mutex h5FileMutex_;

private:	// Data
	std::string fileName_;
    std::unique_ptr<H5::H5File> h5File_;
};




//////////////////////////////////////////////////////////////////////////
//
// Inline implementation of method templates
//

template<typename ElementType, size_t NumDims>
inline boost::multi_array <ElementType, NumDims> 
HDFMultiArrayIO::Read (const std::string& datasetName, const std::string& attrName) const
{
	using boost::numeric_cast;

	BOOST_STATIC_ASSERT (HDF5_type_traits<ElementType>::isSpecialized);

    H5::DataSet dataset = h5File_->openDataSet(datasetName);
	if (attrName.empty())
	{
		H5::DataSpace dataspace = dataset.getSpace();
		int numDims = dataspace.getSimpleExtentNdims();
		if (numDims != NumDims)
		{
			const char* msg = "Unexpected dimensionality of HDF5 data set.";
			throw std::string(msg);
		}
        auto numDimsU = numeric_cast<size_t>(numDims);
        std::vector <hsize_t> hShape (numeric_cast<size_t>(numDimsU));
		dataspace.getSimpleExtentDims ( &hShape[0] );
        std::vector<size_t> shape(numDimsU);
		std::transform(hShape.begin(), hShape.end(), shape.begin(), numeric_cast<size_t, hsize_t>);
		boost::multi_array <ElementType, NumDims> ds (shape);
		H5::DataSpace memspace(ds.num_dimensions(), &hShape[0]);
		dataset.read(ds.data(), HDF5_type_traits<ElementType>::H5_data_type(), memspace, dataspace);
		return ds;
	}
	else
	{
		// Unfortunately, the abstract interface shared by H5::DataSet and 
		// H5::Attribute does not include the read method.
		// It's easiest to just have a parallel block that uses an attribute 
		// instead of the dataset.

		H5::Attribute attr = dataset.openAttribute(attrName.c_str());
		H5::DataSpace dataspace = attr.getSpace();
		int numDims = dataspace.getSimpleExtentNdims();
		if (numDims != NumDims)
		{
			const char* msg = "Unexpected dimensionality of HDF5 data set.";
			throw std::string(msg);
		}
        auto numDimsU = numeric_cast<size_t>(numDims);
        std::vector <hsize_t> hShape (numDimsU);
		dataspace.getSimpleExtentDims ( &hShape[0] );
        std::vector<size_t> shape(numDimsU);
		std::transform(hShape.begin(), hShape.end(), shape.begin(), numeric_cast<size_t, hsize_t>);
		boost::multi_array <ElementType, NumDims> ds (shape);
		attr.read(HDF5_type_traits<ElementType>::H5_data_type(), ds.data());
		return ds;
	}
}

// avm
/*
 *      USEAGE :
*       auto h5File = HDFMultiArrayIO::CreateReader(traceFileName);
		unsigned long long offset[3] = {0,0,0};
		auto oo = h5File->ReadChunkedData<short,3>("/TraceData/Traces",offset);
		auto ptr = oo.data();
*
*/
template<typename ElementType, size_t NumDims>
inline
boost::multi_array <ElementType, NumDims>
HDFMultiArrayIO::ReadChunkedData(const std::string &datasetName,
                                 const unsigned long long* offset,
                                 std::vector<int> count) const
{
	using boost::numeric_cast;

	BOOST_STATIC_ASSERT (HDF5_type_traits<ElementType>::isSpecialized);


    H5::DataSet dataset = h5File_->openDataSet(datasetName);
	H5::DataSpace dataspace = dataset.getSpace();

	// get num dims
	int numDims = dataspace.getSimpleExtentNdims();

	// get total dim values
	hsize_t dimValue[numDims];
	dataspace.getSimpleExtentDims( dimValue, NULL);

	std::vector <unsigned long long> shape;
	for (int ii = 0; ii < numDims; ++ii)
	{
		shape.push_back(dimValue[ii]);
	}

	// get Chunk dims from file
	hsize_t chunkDims[numDims];

	H5::DSetCreatPropList cparms = dataset.getCreatePlist();
	cparms.getChunk(numDims,chunkDims);

	hsize_t chunkCount[3];
	for(int ii=0; ii < count.size(); ++ii)
		chunkCount[ii] = count[ii];
	// define memory space to read dataset
	//H5::DataSpace mspace(numDims,chunkDims);
	H5::DataSpace mspace(numDims,chunkCount);

	/*ideally selectHyperslab( H5S_SELECT_SET, count, offset );
	  but in this case we assume that count == chunkDim. count is the dim of data that one wants to read.*/
	dataspace.selectHyperslab( H5S_SELECT_SET, chunkCount, offset );

	std::vector<int> chunkDimsVec;
	std::vector<int> chunkDimsVec2;
	for(int ii=0; ii<numDims; ++ii)
	{
		chunkDimsVec.push_back(chunkDims[ii]);
		chunkDimsVec2.push_back(count[ii]);
	}

	boost::multi_array <ElementType, NumDims> ds (chunkDimsVec2);
    //boost::multi_array <ElementType, NumDims> ds (chunkDimsVec);
	dataset.read( ds.data(),
				  HDF5_type_traits<ElementType>::H5_data_type(),
				  mspace,
				  dataspace );

    /*dataset.read( outputPtr,
				  HDF5_type_traits<ElementType>::H5_data_type(),
				  mspace,
				  dataspace );*/

	std:: cout << "done " <<std ::endl;
    return ds;
}

///////////////////////////////////////////////////

/*  avm
 *  USEAGE :
 *  string trcFileLoc = "/home/UNIXHOME/amaster/avmDocs/testtrace.trc.h5";
	auto h5File = HDFMultiArrayIO::CreateReader(trcFileLoc);

	unsigned long long offset[3] = {0,0,0};

	auto oo = h5File->ReadChunkedData<short,3>("/TraceData/Traces",offset);
	auto ptr = oo.data();
 */
template<typename ElementType, size_t NumDims>
inline
void
HDFMultiArrayIO::ReadChunkedData(const std::string &datasetName,
								 ElementType* dataPtr,
								 const unsigned long long* offset,
								 std::vector<int> count) const
{
	using boost::numeric_cast;

	BOOST_STATIC_ASSERT (HDF5_type_traits<ElementType>::isSpecialized);

	H5::DataSet dataset = h5File_->openDataSet(datasetName);
	H5::DataSpace dataspace = dataset.getSpace();

	// get num dims
	int numDims = dataspace.getSimpleExtentNdims();

	// get total dim values
	hsize_t dimValue[numDims];
	dataspace.getSimpleExtentDims( dimValue, NULL);

	// get Chunk dims from file
	hsize_t chunkDims[numDims];

	H5::DSetCreatPropList cparms = dataset.getCreatePlist();
	cparms.getChunk(numDims,chunkDims);

	hsize_t chunkCount[3];
	for(int ii=0; ii < count.size(); ++ii)
		chunkCount[ii] = count[ii];

	// define memory space to read dataset
	H5::DataSpace mspace(numDims,chunkCount);
	//H5::DataSpace mspace(numDims,chunkDims);

	/*ideally selectHyperslab( H5S_SELECT_SET, count, offset );
	  but in this case we assume that count == chunkDim. count is the dim of data that one wants to read.*/
	//dataspace.selectHyperslab( H5S_SELECT_SET, chunkDims, offset );
	dataspace.selectHyperslab( H5S_SELECT_SET, chunkCount, offset );

	/*dataset.read( dataPtr,
				  HDF5_type_traits<ElementType>::H5_data_type(),
				  mspace,
				  dataspace );*/
	dataset.read( dataPtr,
				  HDF5_type_traits<ElementType>::H5_data_type(),
				  mspace,
				  dataspace );

}


/////////////////////////////////////////////////////


template <typename ElementType, size_t NumDims>
inline void
HDFMultiArrayIO::Write (const std::string& datasetName,
                        const boost::multi_array<ElementType, NumDims>& data)
{
    BOOST_STATIC_ASSERT (HDF5_type_traits<ElementType>::isSpecialized);

    H5::DataType dtype = HDF5_type_traits<ElementType>::H5_data_type();
    std::vector<hsize_t> dims(data.shape(), data.shape() + NumDims);
    H5::DataSpace dspace(NumDims, dims.data());

    H5::DataSet dataset = h5File_->createDataSet(datasetName, dtype, dspace);

    dataset.write(data.data(), dtype);
}


template <typename ElementType>
inline void
HDFMultiArrayIO::Write (const std::string& datasetName,
                        const std::vector<ElementType>& data)
{
    BOOST_STATIC_ASSERT (HDF5_type_traits<ElementType>::isSpecialized);

    H5::DataType dtype = HDF5_type_traits<ElementType>::H5_data_type();
    std::vector<hsize_t> dims { data.size() };
    H5::DataSpace dspace(1, dims.data());

    H5::DataSet dataset = h5File_->createDataSet(datasetName, dtype, dspace);

    dataset.write(data.data(), dtype);
}


/// Specialization for writing vector<string>.
template <>
inline void
HDFMultiArrayIO::Write<std::string>(const std::string& datasetName,
                                    const std::vector<std::string>& data)
{
    if (data.empty()) return;

    const hsize_t n = data.size();
    const H5::StrType dtype (0, H5T_VARIABLE);
    const H5::DataSpace dspace (1, &n);

    std::vector<const char*> cstr(n);
    for (size_t i = 0; i < n; ++i)
    {
        cstr[i] = data[i].c_str();
    }

    H5::DataSet dataset = h5File_->createDataSet(datasetName, dtype, dspace);
    dataset.write(cstr.data(), dtype, dspace);
}


}} // PacBio::Primary

#endif	// Sequel_Basecaller_Common_HDFMultiArrayIO_H_
