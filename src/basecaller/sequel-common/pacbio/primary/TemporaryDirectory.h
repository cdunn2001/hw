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

#ifndef SEQUELPRIMARYANALYSIS_TEMPORARYDIRECTORY_H
#define SEQUELPRIMARYANALYSIS_TEMPORARYDIRECTORY_H

#include <boost/filesystem.hpp>

namespace PacBio
{
 namespace Primary
 {
  /// A class that creates temporary file names, and deletes them when the object is destroyed, unless the test fails.
  ///
  class TemporaryDirectory
  {
  private:
      bool keep_;
      boost::filesystem::path path_;
  public:
      TemporaryDirectory(std::string root = "");
      ~TemporaryDirectory() noexcept;
      void Keep()
      {
          keep_ = true;
      }

      /// Generate a filename and save it so that it can be deleted later.
      ///
      boost::filesystem::path GenerateTempFileName(const std::string& base);
  };



 }
}

#endif //SEQUELPRIMARYANALYSIS_TEMPORARYDIRECTORY_H
