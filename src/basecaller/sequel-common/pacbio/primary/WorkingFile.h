// Copyright (c) 2017, Pacific Biosciences of California, Inc.
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
/// \brief definition of a object that manages a filename. While the file is being written, the file name is
/// contains the extension ".tmp" and when the file is closed, the file is renamed without the ".tmp" extension.



#ifndef SEQUELACQUISITION_SEQUELWORKINGFILE_H
#define SEQUELACQUISITION_SEQUELWORKINGFILE_H

#include <string>
#include <cstdio>
#include <pacbio/POSIX.h>

class WorkingFile
{
public:
    void SetupWorkingFilename(const std::string& filename)
    {
        openForWrite_ = true;

        if (PacBio::Text::String::StartsWith(filename, "/dev"))
        {
            workingFilename_ = filename;
            finalFilename_ = filename;
        }
        else
        {
            workingFilename_ = filename + ".tmp";
            finalFilename_ = filename;
            if (PacBio::POSIX::IsDirectory(finalFilename_))
            {
                finalFilename_ = "";
                throw PBException("Can't write to " + finalFilename_);
            }
            if (PacBio::POSIX::IsFile(finalFilename_))
            {
                PacBio::POSIX::Unlink(finalFilename_);
                if (PacBio::POSIX::IsFile(finalFilename_))
                {
                    std::string message = "Can't remove " + finalFilename_ + " before writing";
                    finalFilename_ = "";
                    throw PBException(message);
                }
            }
        }
    }

    void CleanupWorkingFilename()
    {
        if (workingFilename_ != finalFilename_ && finalFilename_ != "")
        {
            if (PacBio::POSIX::IsFile(workingFilename_))
            {
                if (std::rename(workingFilename_.c_str(), finalFilename_.c_str()))
                {
                    throw PBExceptionErrno("rename(" + workingFilename_ + "," + finalFilename_ + ") failed");
                }
            }
        }
    }

    bool IsOpenForWrite() const { return openForWrite_; }

    operator const char* () const
    {
        return workingFilename_.c_str();
    }
    const std::string& FinalFilename() const {  return finalFilename_; }

private:
    std::string workingFilename_;
    std::string finalFilename_;
    bool openForWrite_{false};
};

#endif //SEQUELACQUISITION_SEQUELWORKINGFILE_H
