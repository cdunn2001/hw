
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
//
//	HDF5 I/O with boost::multi_array
//
//	A class that wraps an H5::H5File with some convenient operations 
//	using Boost MultiArray.
//
//	For more information about the HDF5 file format, see ...
//	http://www.hdfgroup.org/HDF5/
//
//	For more information about Boost MultiArray, see ...
//	http://www.boost.org/doc/libs


#include "HDFMultiArrayIO.h"

#include <boost/numeric/conversion/cast.hpp>
#include <memory>

namespace PacBio {
namespace Primary {

using std::lock_guard;
using std::mutex;
using std::string;
using std::vector;

using boost::numeric_cast;

std::mutex HDFMultiArrayIO::h5FileMutex_ {};

// HDF5 exceptions cause horrible messages to be printed---
// even when the exceptions are being caught and are "normal".
// We don't want that.  Use this, RAII style.
class H5Silencer {
private:
    H5E_auto2_t func;
    void* client_data;

public:
    H5Silencer()
    {
        H5::Exception::getAutoPrint(func, &client_data);
        H5::Exception::dontPrint();
    }

    ~H5Silencer()
    {
        H5::Exception::setAutoPrint(func, client_data);
    }
};


//
// Helper routines

// Returns a group or dataset as a unique_ptr to H5Object.
std::unique_ptr<H5::H5Object>
HDFMultiArrayIO::getH5Object(const std::string& groupName,
                             const std::string& datasetName) const
{
    std::unique_ptr<H5::H5Object> obj;
    string objName = groupName;
    if (!datasetName.empty())
    {
        objName += "/" + datasetName;
        try {
            obj.reset(new H5::DataSet(h5File_->openDataSet(objName)));
        }
        catch (const H5::FileIException& ex)
        {
            std::ostringstream msg;
            msg << "Failed to open data set \"" << objName
                << "\" in file \"" << FileName()
                << "\" : " << ex.getDetailMsg();
            throw msg.str();
        }
    }
    else
    {
        try {
            obj.reset(new H5::Group(h5File_->openGroup(objName)));
        }
        catch (const H5::FileIException& ex)
        {
            std::ostringstream msg;
            msg << "Failed to open group \"" << objName
                << "\" in file \"" << FileName()
                << "\" : " << ex.getDetailMsg();
            throw msg.str();
        }
    }

    return obj;
}


std::string
HDFMultiArrayIO::getH5ObjectName(const std::string& groupName,
                                 const std::string& datasetName) const
{
    string objName = groupName;
    if (!datasetName.empty())
        objName += "/" + datasetName;
    return objName;
}


HDFMultiArrayIO::HDFMultiArrayIO (const std::string& fileName, AccessMode mode)
	: fileName_		(fileName)
{
    // Used implicitly to release mutex when lock goes out of scope.
    lock_guard<mutex> lock(h5FileMutex_);

	try
	{
        h5File_.reset( new H5::H5File (fileName, mode) );
	}
	catch (H5::Exception ex)
	{
		std::ostringstream msg;
		msg << ex.getDetailMsg() << ".";
		msg << "  Failed to open file, " << fileName << ",";
        msg << " with access mode " << mode << ".";
		throw H5::Exception(ex.getFuncName(), msg.str());
	}
}

HDFMultiArrayIO::~HDFMultiArrayIO(void)
{
    lock_guard<mutex> lock(h5FileMutex_);
    h5File_.reset();
}

int HDFMultiArrayIO::Dimensionality (const std::string& datasetName, const std::string& attrName) const
{
    H5::DataSet dataset = h5File_->openDataSet(datasetName);
	if (attrName.empty())
	{
		H5::DataSpace dataspace = dataset.getSpace();
		return dataspace.getSimpleExtentNdims();
	}
	else
	{
		H5::Attribute attr = dataset.openAttribute (attrName.c_str());
		H5::DataSpace dataspace = attr.getSpace();
		return dataspace.getSimpleExtentNdims();
	}
}


std::vector<int> HDFMultiArrayIO::Shape (const std::string& datasetName, const std::string& attrName) const
{
    H5::DataSet dataset = h5File_->openDataSet(datasetName);
	H5::DataSpace dataspace;
	if (attrName.empty())
	{
		dataspace = dataset.getSpace();
	}
	else
	{
		H5::Attribute attr = dataset.openAttribute (attrName.c_str());
		dataspace = attr.getSpace();
	}

	int numDims = dataspace.getSimpleExtentNdims();
    std::vector <hsize_t> hShape (numeric_cast<size_t>(numDims));
	dataspace.getSimpleExtentDims ( &hShape[0] );
	std::vector <int> shape (hShape.size());
	for (size_t i = 0; i < shape.size(); ++i)
	{
		assert (i < hShape.size());
        shape[i] = numeric_cast<int> (hShape[i]);
	}
	return shape;
}

std::vector<std::string>
HDFMultiArrayIO::ReadStringDataset(const std::string& datasetName) const
{
    H5::DataSet dataset = h5File_->openDataSet(datasetName);
    H5::DataSpace dataspace = dataset.getSpace();
    H5::DataSpace memspace(dataspace);

    // Dimensions of the data set.
    vector<int> dims = Shape(datasetName, "");

    vector<char*> res(dims[0]);

    H5::StrType datatype(0, H5T_VARIABLE);

    dataset.read(&res.front(), datatype, memspace, dataspace);

    vector<string> result(dims[0]);
    for (int ii=0; ii < dims[0]; ++ii)
    {
        result[ii] = string(res[ii]);
    }

    return result;
}


template <typename T>
T HDFMultiArrayIO::ReadScalarAttribute(
		const std::string& attributeName,
		const std::string& groupName,
		const std::string& datasetName) const
{
	BOOST_STATIC_ASSERT (HDF5_type_traits<T>::isSpecialized);

	std::vector<T> result = ReadVectorAttribute<T>(attributeName, groupName, datasetName);

	if (result.size() != 1)
	{
		std::string objName = groupName;
		if (!datasetName.empty()) objName += "/" + datasetName;
		std::ostringstream msg;
		msg << "Expected scalar, found vector attribute " << attributeName
			<< " of object " << objName
			<< " in file " << FileName() << ".";
		throw msg.str();
	}

	return result[0];
}


// Explicit template instantiations.
template int HDFMultiArrayIO::ReadScalarAttribute<int>(
	const std::string& attributeName,
	const std::string& groupName,
	const std::string& datasetName) const;

template float HDFMultiArrayIO::ReadScalarAttribute<float>(
	const std::string& attributeName,
	const std::string& groupName,
	const std::string& datasetName) const;


// Template specialization for std::string
template <>
string HDFMultiArrayIO::ReadScalarAttribute<string>(
		const string& attributeName,
		const string& groupName,
		const string& datasetName) const
{
    auto obj = getH5Object(groupName, datasetName);
    string objName = getH5ObjectName(groupName, datasetName);
	H5::Attribute attr;

	try {attr = obj->openAttribute(attributeName);}
	catch (const H5::AttributeIException& ex)
	{
		std::ostringstream msg;
		msg << "Failed to open attribute \"" << attributeName
            << "\" of object \"" << objName
			<< "\" in file \"" << FileName()
			<< "\" : " << ex.getDetailMsg();
		throw msg.str();
	}

	H5::DataSpace dataspace = attr.getSpace();
	int numDims = dataspace.getSimpleExtentNdims();

	if (numDims != 0)
	{
		std::ostringstream msg;
		msg << "Bad dimensionality of attribute " << attributeName
			<< " of object " << objName
			<< " in file " << FileName() << ".";
		throw msg.str();
	}

	assert (numDims == 0);

	// Read the attribute data.
	string result;
	attr.read(H5::StrType(0, H5T_VARIABLE), result);

	return result;
}


template <typename T>
vector<T> HDFMultiArrayIO::ReadVectorAttribute(
	const string& attributeName,
	const string& groupName,
	const string& datasetName) const
{
	BOOST_STATIC_ASSERT (HDF5_type_traits<T>::isSpecialized);

    auto obj = getH5Object(groupName, datasetName);
    std::string objName = getH5ObjectName(groupName, datasetName);
	H5::Attribute attr;

	try {attr = obj->openAttribute(attributeName);}
	catch (const H5::AttributeIException& ex)
	{
		std::ostringstream msg;
		msg << "Failed to open attribute \"" << attributeName
			<< "\" of object \"" << objName
			<< "\" in file \"" << FileName()
			<< "\" : " << ex.getDetailMsg();
		throw msg.str();
	}

	H5::DataSpace dataspace = attr.getSpace();
	int numDims = dataspace.getSimpleExtentNdims();

	if (numDims < 0 || numDims > 1)
	{
		std::ostringstream msg;
		msg << "Bad dimensionality of attribute " << attributeName
			<< " of object " << objName
			<< " in file " << FileName() << ".";
		throw msg.str();
	}

	assert (numDims == 0 || numDims == 1);

	// Get the size of the attribute.
	size_t size;
	if (numDims == 0)
	{
		// The attribute is actually a scalar.
		size = 1;
	}
	else
	{
		// The attribute is indeed a vector (i.e., 1D array).
		hsize_t hsize;
		dataspace.getSimpleExtentDims(&hsize);
        size = numeric_cast<size_t>(hsize);
	}

	// Allocate the memory.
	vector<T> result(size);

	// Read the attribute data.
	attr.read(HDF5_type_traits<T>::H5_data_type(), &result[0]);

	return result;
}

// Explicit template instantiations.
template vector<float> HDFMultiArrayIO::ReadVectorAttribute<float>(
    const string& attributeName,
    const string& groupName,
    const string& datasetName) const;


template <class T>
void HDFMultiArrayIO::WriteScalarAttribute(
        const T& attributeValue,
        const string& attributeName,
        const string& groupName,
        const string& datasetName)
{
    BOOST_STATIC_ASSERT (HDF5_type_traits<T>::isSpecialized);

    auto obj = getH5Object(groupName, datasetName);

    H5::DataSpace scalarDspace;
    H5::DataType dtype(HDF5_type_traits<T>::H5_data_type());

    try {
        obj->createAttribute(attributeName, dtype, scalarDspace);
        H5::Attribute attr = obj->openAttribute(attributeName);
        attr.write(dtype, &attributeValue);
    }
    catch (const H5::Exception& ex)
    {
        std::string objName = groupName + "/" + datasetName;
        std::ostringstream msg;
        msg << "Failed to create attribute \"" << attributeName
            << "\" of object \"" << objName
            << "\" in file \"" << FileName()
            << "\" : " << ex.getDetailMsg();
        throw msg.str();
    }
}


// Explicit template instantiations.
template
void HDFMultiArrayIO::WriteScalarAttribute<int>
       (const int& attributeValue,
        const string& attributeName,
        const string& groupName,
        const string& datasetName);

template
void HDFMultiArrayIO::WriteScalarAttribute<uint32_t>
       (const uint32_t& attributeValue,
        const string& attributeName,
        const string& groupName,
        const string& datasetName);


void HDFMultiArrayIO::WriteStringAttribute(
        const string& attributeValue,
        const string& attributeName,
        const string& groupName,
        const string& datasetName)
{
    auto obj = getH5Object(groupName, datasetName);

    H5::DataSpace scalarDspace;
    H5::StrType dtype(H5::PredType::C_S1, 256);

    try {
        auto attr = obj->createAttribute(attributeName, dtype, scalarDspace);
        attr.write(dtype, attributeValue.c_str());
    }
    catch (const H5::Exception& ex)
    {
        std::string objName = groupName + "/" + datasetName;
        std::ostringstream msg;
        msg << "Failed to create attribute \"" << attributeName
            << "\" of object \"" << objName
            << "\" in file \"" << FileName()
            << "\" : " << ex.getDetailMsg();
        throw msg.str();
    }
}


void HDFMultiArrayIO::CreateGroup(const std::string& groupName)
{
    H5Silencer stfu;

    // if the group already exists, just return.
    try {
        h5File_->openGroup(groupName);
        return;
    }
    catch (const H5::Exception&) {}

    try {
        h5File_->createGroup(groupName);
    }
    catch (const H5::Exception& ex)
    {
        std::ostringstream msg;
        msg << "Failed to create group \"" << groupName
            << "\" in file \"" << FileName()
            << "\" : " << ex.getDetailMsg();
        throw msg.str();
    }
}



}} // PacBio::Primary
