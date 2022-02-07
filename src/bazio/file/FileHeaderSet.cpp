// Copyright (c) 2022, Pacific Biosciences of California, Inc.
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

#include <bazio/Sanity.h>
#include <bazio/SmartMemory.h>

#include "FileHeaderSet.h"

using namespace PacBio::Primary;

namespace PacBio::BazIO
{

FileHeaderSet::FileHeaderSet(const std::vector<std::pair<std::string,std::shared_ptr<std::FILE>>>& files)
{
    for (const auto& file : files)
    {
        auto fp = file.second.get();
        const auto& fileName = file.first;

        // Seek for SANITY block and last position of the file header
        uint64_t headerSize = Sanity::FindAndVerify(fp);

        // Wrap header into smrt pointer
        auto header = SmartMemory::AllocMemPtr<char>(headerSize + 1);

        // Set position indicator to beginning
        std::rewind(fp);

        // Read file header
        size_t result = std::fread(header.get(), 1, headerSize, fp);
        if (result != headerSize)
            throw std::runtime_error("Cannot read file header: " + fileName + "!");

        FileHeader fh(header.get(), headerSize);

        // Check if BAZ version is correct
        if (!(fh.BazMajorVersion() == 2 && fh.BazMinorVersion() == 0))
        {
            PBLOG_ERROR << "Incorrect BAZ version provided. Need version 2.0.x, provided version is "
                        << fh.BazVersion();
            exit(EXIT_FAILURE);
        }

        if (fh.ZmwNumbers().empty())
        {
            PBLOG_ERROR << "No ZMW numbers found in baz file " << fileName;
            exit(EXIT_FAILURE);
        }

        if (!fh.Complete())
        {
            PBLOG_ERROR << "Trying to read unfinished baz file " << fileName;
            exit(EXIT_FAILURE);
        }

        if (fh.Truncated())
            PBLOG_INFO << "Converting truncated file " << fileName;

        // Check sanity dword
        if (!Sanity::ReadAndVerify(fp))
        {
            PBLOG_ERROR << "Corrupt file. Cannot read SANITY DWORD after FILE_HEADER";
            exit(EXIT_FAILURE);
        }

        fhs_.insert(std::move(fh));
    }

    if (!std::equal(fhs_.begin(), fhs_.end(), fhs_.begin(),
                   [&](const FileHeader& a, const FileHeader& b)
                   { return IsConsistent(a, b); }))
    {
        throw PBException("Inconsistent file headers detected, check files!");
    }

    if (!DistinctHoleNumbers())
    {
        throw PBException("ZMW hole numbers are not distinct, check files!");
    }

    for (const auto& fh : fhs_)
    {
        // Adjust index since each BAZ file stores it sequentially starting at 0.
        for (const auto& kv : fh.ZmwInformation().ZmwNumbersToIndex())
            zmwNumbersToIndex_[kv.first] = totalNumZmws_ + kv.second;

        zmwNumbers_.insert(std::end(zmwNumbers_),
                           fh.ZmwNumbers().begin(), fh.ZmwNumbers().end());

        zmwFeatures_.insert(std::end(zmwFeatures_),
                            fh.ZmwUnitFeatures().begin(), fh.ZmwUnitFeatures().end());

        maxNumZmws_.push_back(fh.MaxNumZMWs());
        numSuperChunks_.push_back(fh.NumSuperChunks());

        totalNumZmws_ += fh.MaxNumZMWs();
    }

    movieName_ = PacBio::Text::String::Split(fhs_.begin()->MovieName(), '.')[0];
}

bool FileHeaderSet::DistinctHoleNumbers() const
{
    std::set<uint32_t> holeNumbers;
    size_t totalNumbers = 0;
    for (const auto& fh : fhs_)
    {
        holeNumbers.insert(fh.ZmwNumbers().begin(), fh.ZmwNumbers().end());
        totalNumbers += fh.ZmwNumbers().size();
    }

    if (holeNumbers.size() != totalNumbers)
    {
        return false;
    }

    return true;
}

bool FileHeaderSet::IsConsistent(const FileHeader& a, const FileHeader& b) const
{
    if (a.BazLongVersion() != b.BazLongVersion())
    {
        PBLOG_ERROR << "FileHeader BAZ versions mismatch for files "
                    << a.MovieName() << " and " << b.MovieName() << "!";
        return false;
    }

    if (a.BaseCallerVersion() != b.BaseCallerVersion())
    {
        PBLOG_ERROR << "FileHeader BaseCaller versions mismatch for files "
                    << a.MovieName() << " and " << b.MovieName() << "!";
        return false;
    }

    if (a.ZmwInformation().HoleFeatureMap().size() != b.ZmwInformation().HoleFeatureMap().size()
        || !std::equal(a.ZmwInformation().HoleFeatureMap().begin(),
                       a.ZmwInformation().HoleFeatureMap().end(), b.ZmwInformation().HoleFeatureMap().begin(),
                       [](const std::pair<std::string,uint32_t>& a, const std::pair<std::string,uint32_t>& b)
                       { return a.first == b.first && a.second == b.second; }))
    {
        PBLOG_ERROR << "FileHeader Hole Feature map mismatch for files "
                    << a.MovieName() << " and " << b.MovieName() << "!";
        return false;
    }

    if (a.ZmwInformation().HoleTypesMap().size() != b.ZmwInformation().HoleTypesMap().size()
        || !std::equal(a.ZmwInformation().HoleTypesMap().begin(),
                       a.ZmwInformation().HoleTypesMap().end(), b.ZmwInformation().HoleTypesMap().begin(),
                       [](const std::pair<std::string,uint32_t>& a, const std::pair<std::string,uint32_t>& b)
                       { return a.first == b.first && a.second == b.second; }))
    {
        PBLOG_ERROR << "FileHeader Hole Types map mismatch for files "
                    << a.MovieName() << " and " << b.MovieName() << "!";
        return false;
    }

    // TODO: This needs to be updated to support BAZ files with different number of super chunks.
    if (a.NumSuperChunks() != b.NumSuperChunks())
    {
        PBLOG_ERROR << "FileHeader number of super chunk mismatch for files "
                    << a.MovieName() << " and " << b.MovieName() << "!";
        return false;
    }

    if (a.Internal() != b.Internal())
    {
        PBLOG_ERROR << "FileHeader internal mismatch for files "
                    << a.MovieName() << " and " << b.MovieName() << "!";
        return false;
    }

    if (a.PacketByteSize() != b.PacketByteSize())
    {
        PBLOG_ERROR << "FileHeader packet byte size mismatch for files "
                    << a.MovieName() << " and " << b.MovieName() << "!";
        return false;
    }

    if (a.MetricByteSize() != b.MetricByteSize())
    {
        PBLOG_ERROR << "FileHeader metric byte size mismatch for files "
                    << a.MovieName() << " and " << b.MovieName() << "!";
        return false;
    }

    if (PacBio::Text::String::Split(a.MovieName(), '.').front() !=
        PacBio::Text::String::Split(b.MovieName(), '.').front())
    {
        PBLOG_ERROR << "FileHeader movie name mismatch for files "
                    << a.MovieName() << " and " << b.MovieName() << "!";
        return false;
    }

    // TODO:
    // Check packet/metric fields?
    // Check basemap and relative amplitudes?
    // Check metric frames?
    // Check frameRate?

    return true;
}

} // namespace PacBio::BazIO

