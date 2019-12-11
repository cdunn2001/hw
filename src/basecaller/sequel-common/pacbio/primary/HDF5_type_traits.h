#ifndef Crystal_Tools_HDF5_type_traits_H_
#define Crystal_Tools_HDF5_type_traits_H_

#include <cstdint>

#include "HDF5cpp.h"

// This traits structure template must be specialized for any type 
// supported by HDFMultiArrayIO::Read and HDFMultiArrayIO::Write.
template <typename T>
struct HDF5_type_traits
{
	static const bool isSpecialized = false;
	static const H5::PredType& H5_data_type();
};

//
// Specializations
//

// For each specialization, H5_data_type is defined in HDF5_type_traits.cpp.

template <>
struct HDF5_type_traits<bool>
{
	static const bool isSpecialized = true;
    static const H5::PredType& H5_data_type(); // Defined in HDF5_type_traits.cpp.
};


// Unsigned integer types

template <>
struct HDF5_type_traits<std::uint16_t>
{
	static const bool isSpecialized = true;
    static const H5::PredType& H5_data_type();
};

template <>
struct HDF5_type_traits<std::uint32_t>
{
    static const bool isSpecialized = true;
    static const H5::PredType& H5_data_type();
};

template <>
struct HDF5_type_traits<unsigned char>
{
	static const bool isSpecialized = true;
    static const H5::PredType& H5_data_type();
};


// Signed integer types

template <>
struct HDF5_type_traits<short>
{
	static const bool isSpecialized = true;
    static const H5::PredType& H5_data_type();
};

template <>
struct HDF5_type_traits<int>
{
	static const bool isSpecialized = true;
    static const H5::PredType& H5_data_type();
};


// Floating-point types

template <>
struct HDF5_type_traits<float>
{
	static const bool isSpecialized = true;
    static const H5::PredType& H5_data_type();
};

template <>
struct HDF5_type_traits<double>
{
    static const bool isSpecialized = true;
    static const H5::PredType& H5_data_type();
};

#endif	// Crystal_Tools_HDF5_type_traits_H_
