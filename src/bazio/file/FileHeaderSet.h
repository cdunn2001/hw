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

#ifndef PACBIO_BAZIO_FILE_FILE_HEADER_SET_H
#define PACBIO_BAZIO_FILE_FILE_HEADER_SET_H

#include <set>
#include <unordered_set>

#include <pacbio/text/String.h>

#include <bazio/MetricField.h>
#include <bazio/file/FileHeader.h>

namespace PacBio::BazIO
{

class FileHeaderSet
{
public:
    FileHeaderSet(const std::vector<std::pair<std::string,std::unique_ptr<std::FILE>>>& files);

    // Default constructor
    FileHeaderSet() = delete;

    // Move constructor
    FileHeaderSet(FileHeaderSet&&) = default;

    // Copy constructor
    FileHeaderSet(const FileHeaderSet&) = default;

    // Move assignment operator
    FileHeaderSet& operator=(FileHeaderSet&&) = default;

    // Copy assignment operator
    FileHeaderSet& operator=(const FileHeaderSet&) = delete;

    // Destructor
    ~FileHeaderSet() = default;

public:
    const auto& FileHeaders() const
    { return fhs_; }

    // Returns the maximum number of ZMWs for each BAZ file.
    const std::vector<size_t>& MaxNumZmws() const
    { return maxNumZmws_; }

    // Returns the total number of ZMWs across all BAZ files.
    uint32_t TotalNumZmws() const
    { return totalNumZmws_; }

    const std::vector<uint32_t>& NumSuperChunks() const
    { return numSuperChunks_; }

    uint32_t MetricByteSize() const
    { return fhs_.begin()->MetricByteSize();; }

    uint32_t ZmwNumberToIndex(const uint32_t holeNumber) const
    { return zmwNumbersToIndex_.at(holeNumber); }

    uint32_t ZmwIndexToNumber(const uint32_t index) const
    { return zmwNumbers_.at(index); }

    const std::vector <uint32_t>& ZmwNumbers() const
    { return zmwNumbers_; }

    uint32_t ZmwFeatures(uint32_t zmwIndex) const
    { return zmwFeatures_.at(zmwIndex); }

    const std::vector<Primary::MetricField>& MetricFields() const
    { return fhs_.begin()->MetricFields(); }

    const std::vector<BazIO::FieldParams<BazIO::PacketFieldName>>& PacketFields() const
    { return fhs_.begin()->PacketFields(); }

    const std::vector<BazIO::GroupParams<BazIO::PacketFieldName>>& PacketGroups() const
    { return fhs_.begin()->PacketGroups(); }

    uint32_t MetricFrames() const
    { return fhs_.begin()->MetricFrames(); }

    double FrameRateHz() const
    { return fhs_.begin()->FrameRateHz(); }

    const std::vector<float>& RelativeAmplitudes() const
    { return fhs_.begin()->RelativeAmplitudes(); }

    const std::string& BaseMap() const
    { return fhs_.begin()->BaseMap(); }

    const std::string& MovieName() const
    { return movieName_; }

    const Json::Value ExperimentMetadata() const
    { return fhs_.begin()->ExperimentMetadata(); }

    size_t NumHeaders() const
    { return fhs_.size(); }

    std::string BaseCallerVersion() const
    { return fhs_.begin()->BaseCallerVersion(); }

    std::string BazVersion() const
    { return fhs_.begin()->BazVersion(); }

    std::string BazWriterVersion() const
    { return fhs_.begin()->BazWriterVersion(); }

    float MovieTimeInHrs() const
    { return fhs_.begin()->MovieTimeInHrs(); }

    bool Internal() const
    { return fhs_.begin()->Internal(); }

    bool HasPacketField(BazIO::PacketFieldName fieldName) const
    { return fhs_.begin()->HasPacketField(fieldName); }

    bool IsZmwNumberRejected(uint32_t zmwNumber) const
    {
        return std::any_of(fhs_.begin(), fhs_.end(),
                           [&](const auto& fh) { return fh.IsZmwNumberRejected(zmwNumber); });
    }

    float MovieLengthFrames() const
    { return fhs_.begin()->MovieLengthFrames(); }

private:
    struct FileHeaderHash
    {
        std::size_t operator()(const FileHeader& fh) const
        {
            // Hole numbers should be unique for each BAZ file so just take the first one.
            return fh.ZmwNumbers().front();
        }
    };

    struct FileHeaderEqual
    {
        bool operator()(const FileHeader& lhs, const FileHeader& rhs) const
        {
            std::set<uint32_t> lhsZmwNumbers(lhs.ZmwNumbers().begin(), lhs.ZmwNumbers().end());
            std::set<uint32_t> rhsZmwNumbers(rhs.ZmwNumbers().begin(), rhs.ZmwNumbers().end());
            std::vector<uint32_t> diff;
            std::set_difference(lhsZmwNumbers.begin(), lhsZmwNumbers.end(),
                                rhsZmwNumbers.begin(), rhsZmwNumbers.end(),
                                std::back_inserter(diff));
            return (PacBio::Text::String::Split(lhs.MovieName(), '.').front() ==
                    PacBio::Text::String::Split(rhs.MovieName(), '.').front()) &&
                   diff.empty();
        }
    };

    bool DistinctHoleNumbers() const;
    bool IsConsistent(const FileHeader& a, const FileHeader& b) const;

private:
    std::vector<size_t> maxNumZmws_;
    uint32_t totalNumZmws_ = 0;
    std::vector<uint32_t> numSuperChunks_;
    std::map<uint32_t,uint32_t> zmwNumbersToIndex_;
    std::vector<uint32_t> zmwNumbers_;
    std::vector<uint32_t> zmwFeatures_;
    std::string movieName_;
    std::unordered_set<FileHeader, FileHeaderHash, FileHeaderEqual> fhs_;
};

} // namespace PacBio::BazIO

#endif // PACBIO_BAZIO_FILE_FILE_HEADER_SET_H
