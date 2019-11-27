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
/// \brief  handler that manages creating and deleting temporary directories


#include "TemporaryDirectory.h"
#include <pacbio/text/String.h>
#include <pacbio/logging/Logger.h>
#include <pacbio/POSIX.h>
#include <pacbio/utilities/Path.h>

namespace PacBio {
namespace Primary  {

  TemporaryDirectory::TemporaryDirectory(std::string root  )
          : keep_(false)
  {
      boost::filesystem::path t;

      if (root !="" && !POSIX::IsDirectoryWritable(root))
      {
          PBLOG_WARN << "TemporaryDirectory " << root << " is not writeable, using a system temp directory instead";
          root = "";
      }
      if (root == "")
      {
          t = boost::filesystem::temp_directory_path();
      } else {
          t = boost::filesystem::path(root);
      }
      path_ = t / boost::filesystem::unique_path();
      PacBio::Utilities::Path::CreateDirectories(path_);
      PBLOG_DEBUG << "Created temporary directory " << path_;
  }

  TemporaryDirectory::~TemporaryDirectory() noexcept
  {
      try
      {
          if (!keep_)
          {
              PBLOG_DEBUG << "Deleting temporary directory " << path_;
              remove_all(path_);
          }
          else
          {
              PBLOG_INFO << "NOT deleting temporary directory " << path_;
          }
      }
      catch(const std::exception& ex)
      {
          PBLOG_ERROR << "TemporaryDirectory::~TemporaryDirectory() caught exception: " << ex.what();
      }
      catch(...)
      {
          std::cerr << "Uncaught exception caught in ~TemporaryDirectory " << PacBio::GetBackTrace(5);
          PBLOG_FATAL << "Uncaught exception caught in ~TemporaryDirectory " << PacBio::GetBackTrace(5);
          PacBio::Logging::PBLogger::Flush();
          std::terminate();
      }
  }


  boost::filesystem::path TemporaryDirectory::GenerateTempFileName(const std::string& base)
  {
      std::string random = PacBio::Text::String::RandomString(32);
      auto s = random + base;
      auto p = path_ / s;
      PBLOG_DEBUG << "Created temporary file path " << p;
      return p;
  }

}}
