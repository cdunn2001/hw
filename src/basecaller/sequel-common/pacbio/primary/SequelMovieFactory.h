// Copyright (c) 2015, Pacific Biosciences of California, Inc.
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
/// \brief  factory functions to generate various movie-type files for reading
///         and writing. Supports *.mov.h5, *.trc.h5, *.crc

#ifndef _SEQUEL_MOVIE_FACTORY_H_
#define _SEQUEL_MOVIE_FACTORY_H_

#include <pacbio/primary/SequelMovie.h>
#include <pacbio/primary/SequelTraceFile.h>
#include <pacbio/primary/SequelMovieFileHDF5.h>
#include <pacbio/primary/SequelMovieConfig.h>

namespace PacBio {
namespace Primary {


class SequelMovieFactory
{
public:

    /// Determines the file type based on naming convention of filename.
    static SequelMovieFileBase::SequelMovieType MovieType(const std::string& filename);

    /// Create a output file based on the file extension. Currently supported extensions
    /// are .h5, .trc.h5 and .crc.
    static std::unique_ptr<SequelMovieFileBase> CreateOutput(const SequelMovieConfig& config);

    /// Create an input file based on the file extension. Currently supported extensions
    /// are .h5 and .trc.h5 .
    static std::unique_ptr<SequelMovieFileBase> CreateInput(const std::string& filename);

    static size_t EstimatedSize(const std::string& filename, const SequelROI& roi, uint64_t numFrames);
};


}} // end of namespace

#endif // _SEQUEL_MOVIE_FACTORY_H_
