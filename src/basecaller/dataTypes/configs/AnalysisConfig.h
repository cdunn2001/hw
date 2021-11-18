// Copyright (c) 2021, Pacific Biosciences of California, Inc.
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

#ifndef mongo_dataTypes_configs_AnalysisConfig_H_
#define mongo_dataTypes_configs_AnalysisConfig_H_

#include <pacbio/datasource/MovieInfo.h>
#include <pacbio/datasource/PacketLayout.h>

namespace PacBio::Mongo::Data
{

/// Represents the full analysis configuration for analyzing a
/// particular movie and is used by the basecaller as the
/// main structure for obtaining any configuration needed for
/// analysis.
struct AnalysisConfig
{
    PacBio::DataSource::MovieInfo movieInfo;

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
};

// Creates arbitrary configuration mostly for testing purposes.
AnalysisConfig MockAnalysisConfig();

} // namespace PacBio::Mongo::Data

#endif // mongo_dataTypes_configs_AnalysisConfig_H_
