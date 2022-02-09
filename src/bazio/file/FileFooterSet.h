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

#ifndef PACBIO_BAZIO_FILE_FILE_FOOTER_SET_H
#define PACBIO_BAZIO_FILE_FILE_FOOTER_SET_H

#include <map>
#include <vector>

namespace PacBio::BazIO
{

class FileFooterSet
{
public:
    FileFooterSet(const std::map<uint32_t, std::vector<uint32_t>>&& truncationMap)
    : truncationMap_(truncationMap)
    { }

    // Default constructor
    FileFooterSet() = delete;

    // Move constructor
    FileFooterSet(FileFooterSet&&) = default;

    // Copy constructor
    FileFooterSet(const FileFooterSet&) = default;

    // Move assignment operator
    FileFooterSet& operator=(FileFooterSet&&) = default;

    // Copy assignment operator
    FileFooterSet& operator=(const FileFooterSet&) = delete;

    // Destructor
    ~FileFooterSet() = default;
    
public:

    bool IsZmwNumberTruncated(uint32_t zmwNumber) const
    { return truncationMap_.find(zmwNumber) != truncationMap_.cend(); }

private:
    std::map<uint32_t, std::vector<uint32_t>> truncationMap_;
};

} // namespace PacBio::BazIO

#endif // PACBIO_BAZIO_FILE_FILE_FOOTER_SET_H
