
#include "HDF5_type_traits.h"

//////////////////////////////////////////////////////////////////////////
//
//	Type traits to support read and write method templates.
//

const H5::PredType& HDF5_type_traits<bool>::H5_data_type() { return H5::PredType::NATIVE_HBOOL; }

// Unsigned types
const H5::PredType& HDF5_type_traits<unsigned char>::H5_data_type() { return H5::PredType::NATIVE_UINT8;}
const H5::PredType& HDF5_type_traits<std::uint16_t>::H5_data_type () { return H5::PredType::NATIVE_UINT16;}
const H5::PredType& HDF5_type_traits<std::uint32_t>::H5_data_type () { return  H5::PredType::NATIVE_UINT32;}

// Signed types
const H5::PredType& HDF5_type_traits<short>::H5_data_type () { return  H5::PredType::NATIVE_SHORT;}
const H5::PredType& HDF5_type_traits<int>::H5_data_type () { return  H5::PredType::NATIVE_INT;}
const H5::PredType& HDF5_type_traits<float>::H5_data_type () { return  H5::PredType::NATIVE_FLOAT;}
const H5::PredType& HDF5_type_traits<double>::H5_data_type () { return  H5::PredType::NATIVE_DOUBLE;}
