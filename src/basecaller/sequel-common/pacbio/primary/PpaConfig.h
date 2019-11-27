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
//  Description:
/// \brief  Configuration object definition used for ppad command line configuration.
///
#ifndef SEQUELACQUISITION_PPACONFIG_H
#define SEQUELACQUISITION_PPACONFIG_H

#include <pacbio/process/ConfigurationBase.h>

#include <pacbio/primary/ZmwReducedStatsFile.h>

namespace PacBio {
namespace Primary {

/// This is the top level configuration object for PPA (ppad executable)
/// BinDir and RstsDestination are legacy parameters that initially were specified
/// on the command line but now can be specified in this configuration object.
/// The ReducedStatsConfig is a JSON subobject that is used to configure the ppa-reducestats executable,
/// i.e. the contents are used as input to the --config argument of that executable.
class PpaConfig : public PacBio::Process::ConfigurationObject
{
    CONF_OBJ_SUPPORT_COPY(PpaConfig)
    ADD_PARAMETER(std::string, BinDir, "");
    ADD_PARAMETER(std::string, Baz2BamArgs, "");
    ADD_PARAMETER(std::string, RstsDestinationPrefix, "pbi@icc:/data/pacbio/context/collection");
};

}}

#endif //SEQUELACQUISITION_PPACONFIG_H
