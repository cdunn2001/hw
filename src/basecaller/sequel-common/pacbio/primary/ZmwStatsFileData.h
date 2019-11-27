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
// Defines the API for reading and writing the sts.h5 file.
//
// This file can automatically generates the list of members. To see a rendered version of this file,
// do
//
//     make rstdocs
//     cat ZmwStats.rst
//
// in the build directory.


#ifndef SEQUELACQUISITION_ZMWSTATS_H
#define SEQUELACQUISITION_ZMWSTATS_H

#include <type_traits>

#include "boost/multi_array.hpp"

#include <pacbio/utilities/SmartEnum.h>
#include <pacbio/primary/SequelHDF5.h>
#include <pacbio/logging/Logger.h>

namespace PacBio {
namespace Primary {

// Adding a bit of template infrastructure, to enable the automatic population
// of default "not present" values into the stats file.
namespace ZmwStatsHelper {

template <typename T>
struct DefaultIntegral
{
    static_assert(std::is_integral<T>::value, "DefaultIntegral required integral type");
    static_assert(std::is_unsigned<T>::value, "DefaultIntegral only handles unsigned");
    static constexpr T value = std::numeric_limits<T>::max();
};
template <typename T>
struct DefaultFloatingPoint
{
    static_assert(std::is_floating_point<T>::value, "DefaultFloatingPoint requires floating point type");
    static constexpr T value = std::numeric_limits<T>::quiet_NaN();
};

template <typename T>
struct DefaultVal
{
    static constexpr T value = std::conditional<
        std::is_integral<T>::value,
        DefaultIntegral<T>,
        DefaultFloatingPoint<T>
        >::type::value;
};
template <typename T> constexpr T DefaultVal<T>::value;

}

/// A structure used to pass numerical values in and out of the ZmwStatsFile.
/// The structure of this struct is flat, unlike the underlying H5 file which is put into
/// groups. Because of this, the option of a "prefix" argument is used to prepend a string to the
/// name of the dataset to distinguish members of the same name. For example, there is nontemporal
/// Baseline and a temporal Baseline. The former has no prefix, while the latter has a "VsT" prefix.
struct ZmwStats
{
#define float32_t float
#define DECLARE_SEQUELH5_SUPPORT()
#define DECLARE_ZMWSTAT_START_GROUP(name)
#define two_ 2
#define four_ 4
#define nAB_ (nA_+1)
#define nFtriangle_ (nF_ + nF_*(nF_+1)/2)
#define DECLARE_ZMWSTAT_END_GROUP(name)
#define DECLARE_ZMWSTAT_ENUM(...) SMART_ENUM(__VA_ARGS__);
#define DECLARE_ZMWSTAT_START_GROUP_DIAGNOSTICS(name)
#define DECLARE_ZMWSTAT_END_GROUP_DIAGNOSTICS(name)

    uint32_t nA_;
    uint32_t nF_;
    uint32_t nMF_;
    uint32_t index_;

#define DECLARE_ZMWSTATDATASET_1D(prefix, name, type, units, description, hqr, dim0)             type ##_t prefix##name;
#define DECLARE_ZMWSTATDATASET_2D(prefix, name, type, units, description, hqr, dim0, dim1)       boost::multi_array<type ##_t,1>  prefix##name;
#define DECLARE_ZMWSTATDATASET_3D(prefix, name, type, units, description, hqr, dim0, dim1, dim2) boost::multi_array<type ##_t,2>  prefix##name;
#define DECLARE_ZMWSTATDATASET_1D_DIAGNOSTICS(prefix, name, type, units, description, hqr, dim0)             DECLARE_ZMWSTATDATASET_1D(prefix, name, type, units, description, hqr, dim0)
#define DECLARE_ZMWSTATDATASET_2D_DIAGNOSTICS(prefix, name, type, units, description, hqr, dim0, dim1)       DECLARE_ZMWSTATDATASET_2D(prefix, name, type, units, description, hqr, dim0, dim1)
#define DECLARE_ZMWSTATDATASET_3D_DIAGNOSTICS(prefix, name, type, units, description, hqr, dim0, dim1, dim2) DECLARE_ZMWSTATDATASET_3D(prefix, name, type, units, description, hqr, dim0, dim1, dim2)

#include "ZmwStatsFileDefinition.h"

#undef DECLARE_ZMWSTAT_ENUM
#undef DECLARE_ZMWSTATDATASET_1D
#undef DECLARE_ZMWSTATDATASET_2D
#undef DECLARE_ZMWSTATDATASET_3D
#undef DECLARE_ZMWSTATDATASET_1D_DIAGNOSTICS
#undef DECLARE_ZMWSTATDATASET_2D_DIAGNOSTICS
#undef DECLARE_ZMWSTATDATASET_3D_DIAGNOSTICS

#define DECLARE_ZMWSTAT_ENUM(...)
#define DECLARE_ZMWSTATDATASET_1D(prefix, name, type, units, description, hqr, dim0)             // , prefix##name{0}
#define DECLARE_ZMWSTATDATASET_2D(prefix, name, type, units, description, hqr, dim0, dim1)       , prefix##name(boost::extents[dim1])
#define DECLARE_ZMWSTATDATASET_3D(prefix, name, type, units, description, hqr, dim0, dim1, dim2) , prefix##name(boost::extents[dim1][dim2])
#define DECLARE_ZMWSTATDATASET_1D_DIAGNOSTICS(prefix, name, type, units, description, hqr, dim0)             // DECLARE_ZMWSTATDATASET_1D(prefix, name, type, units, description, hqr, dim0)
#define DECLARE_ZMWSTATDATASET_2D_DIAGNOSTICS(prefix, name, type, units, description, hqr, dim0, dim1)       DECLARE_ZMWSTATDATASET_2D(prefix, name, type, units, description, hqr, dim0, dim1)
#define DECLARE_ZMWSTATDATASET_3D_DIAGNOSTICS(prefix, name, type, units, description, hqr, dim0, dim1, dim2) DECLARE_ZMWSTATDATASET_3D(prefix, name, type, units, description, hqr, dim0, dim1, dim2)

    ZmwStats(uint32_t nA, uint32_t nF, uint32_t nMF)
     : nA_(nA)
     , nF_(nF)
     , nMF_(nMF)

#include "ZmwStatsFileDefinition.h"

    {
        Init();
    }

#undef DECLARE_ZMWSTATDATASET_1D
#undef DECLARE_ZMWSTATDATASET_2D
#undef DECLARE_ZMWSTATDATASET_3D
#undef DECLARE_ZMWSTATDATASET_1D_DIAGNOSTICS
#undef DECLARE_ZMWSTATDATASET_2D_DIAGNOSTICS
#undef DECLARE_ZMWSTATDATASET_3D_DIAGNOSTICS

#define DECLARE_ZMWSTATDATASET_1D(prefix, name, type, units, description, hqr, dim0)             prefix##name = ZmwStatsHelper::DefaultVal<type ##_t>::value;
#define DECLARE_ZMWSTATDATASET_2D(prefix, name, type, units, description, hqr, dim0, dim1)       std::fill_n(prefix##name.data(), prefix##name.num_elements(), ZmwStatsHelper::DefaultVal<type ##_t>::value);
#define DECLARE_ZMWSTATDATASET_3D(prefix, name, type, units, description, hqr, dim0, dim1, dim2) std::fill_n(prefix##name.data(), prefix##name.num_elements(), ZmwStatsHelper::DefaultVal<type ##_t>::value);
#define DECLARE_ZMWSTATDATASET_1D_DIAGNOSTICS(prefix, name, type, units, description, hqr, dim0)             DECLARE_ZMWSTATDATASET_1D(prefix, name, type, units, description, hqr, dim0)
#define DECLARE_ZMWSTATDATASET_2D_DIAGNOSTICS(prefix, name, type, units, description, hqr, dim0, dim1)       DECLARE_ZMWSTATDATASET_2D(prefix, name, type, units, description, hqr, dim0, dim1)
#define DECLARE_ZMWSTATDATASET_3D_DIAGNOSTICS(prefix, name, type, units, description, hqr, dim0, dim1, dim2) DECLARE_ZMWSTATDATASET_3D(prefix, name, type, units, description, hqr, dim0, dim1, dim2)

    /// Initializes all data to zero. In principle, this is not needed because
    /// all values should be overwritten by application code, or loaded by ZmwStatsFile.
    void Init()
    {
#include "ZmwStatsFileDefinition.h"
    }



#undef nAB_
#undef two_
#undef four_
#undef nFtriangle_
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
};

}}

#endif //SEQUELACQUISITION_ZMWSTATS_H
