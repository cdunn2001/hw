#ifndef mongo_dataTypes_configs_MovieConfig_H_
#define mongo_dataTypes_configs_MovieConfig_H_

// Copyright (c) 2019-2020, Pacific Biosciences of California, Inc.
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
//  Description:
//  Defines class MovieConfig.

#include <array>

#include <pacbio/datasource/PacketLayout.h>

#include <common/MongoConstants.h>
#include <dataTypes/AnalogMode.h>

namespace PacBio {
namespace Mongo {
namespace Data {

/// Represents configuration of the instrument, chemistry, and data collection
/// for a particular movie.
//
// TODO this structure is currently in flux.  There are tentative plans
// to alter things so that it is generated directly by the DataSource
// instance.
struct MovieConfig
{
public:
    float frameRate;
    float photoelectronSensitivity;
    float refSnr;

    /// Convention is to order analogs by decreasing relative amplitude.
    std::array<AnalogMode, numAnalogs> analogs;

    // The pedestal value applied upstream before any
    // batch is generated.  This must be subtracted off
    // all batches that come through, in order to get the
    // true trace value
    int16_t pedestal = 0;
    // Batches come through as a variant because it's not
    // known until rumtime what the format ultimately will
    // be.  This field gives us a sneak peak though at what
    // to expect, in case that matters to any particular
    // implementation
    using EncodingFormat = DataSource::PacketLayout::EncodingFormat;
    EncodingFormat encoding = EncodingFormat::INT16;

    // TODO: Will likely need additional members.
};

/// Creates an instance with somewhat arbitrary value.
/// Convenient for some unit tests.
MovieConfig MockMovieConfig();

}}}     // namespace PacBio::Mongo::Data

#endif // mongo_dataTypes_configs_MovieConfig_H_
