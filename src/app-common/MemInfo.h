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
// File Description:
/// \brief  C++ interface to /sys/devices/system/node/*/meminfo
//
// Programmer: Mark Lakata

#ifndef PA_UTILITIES_MEMINFO_H
#define PA_UTILITIES_MEMINFO_H

#include <fstream>
#include <string>
#include <regex>

#include <pacbio/logging/Logger.h>
#include <pacbio/PBException.h>


namespace PacBio {
namespace Utilities {

class NodeMemInfo 
{
public:
    NodeMemInfo(int node)
        : numaNode_(node)
        , filename_("/sys/devices/system/node/node" + std::to_string(numaNode_)
                    + "/meminfo")
    {
    }

    size_t HugePages_Total() const
    {
        return std::stoul(GrepFile(filename_,
            R"(Node \d+ HugePages_Total:\s+(\d+))"));
    }

    size_t HugePages_Free() const
    {
        return std::stoul(GrepFile(filename_,
            R"(Node \d+ HugePages_Free:\s+(\d+))"));
    }

private:
    const int numaNode_;
    const std::string filename_;
    // returns the string contents of the first parenthesized group inside the regex
    inline static std::string GrepFile(const std::string& filename, const std::string& res)
    {
        try
        {
            std::regex re(res);
            std::ifstream file(filename);
            std::string line;
            while(getline(file,line))
            {
                PBLOG_DEBUG << "line:" << line;
                std::smatch m;
                if (std::regex_search(line, m, re))
                {
                    if (m.size() != 2) throw PBException("Regular expression /"
                        +res + "/ needs a single set of parens.");
                    PBLOG_DEBUG << "! match:" << m[1];
                    return m[1];
                }
            }
        }
        catch(...)
        {
            PBLOG_ERROR << "GrepFile: regex:" << res << " filename:" << filename;
            throw;
        }
        throw PBException("Could not find pattern " + res + " within " +
            filename);
    }

};

}} // namespace

#endif
