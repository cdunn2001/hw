// Copyright (c) 2014-2020, Pacific Biosciences of California, Inc.
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted (subject to the limitations in the
// disclaimer below) provided that the following conditions are met:
//
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
//  * Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
//  * Neither the name of Pacific Biosciences nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE
// GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY PACIFIC
// BIOSCIENCES AND ITS CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL PACIFIC BIOSCIENCES OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
// USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
// OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGE.

#pragma once

#include <pbbam/DataSet.h>
#include <pbbam/RunMetadata.h>

#include <pacbio/process/ConfigurationBase.h>

#include <pacbio/primary/HQRFMethod.h>
#include <bazio/FastaUtilities.h>

#include <postprimary/bam/Platform.h>

#include "UserParameters.h"

namespace PacBio {
namespace Primary {
namespace Postprimary {

class PpaInputFilterConfig : public PacBio::Process::ConfigurationObject
{
public:
    // SNR cutoff for the HQ-region
    ADD_PARAMETER_NO_DEFAULT(float, minSnr);
};

class PpaAdapterFindingConfig : public PacBio::Process::ConfigurationObject
{
public:
    // Disables adapter finding, note that if this flag is
    // set to true, the adapter sequences are populated below
    // but adapter finding is not run.
    ADD_PARAMETER(bool, disableAdapterFinding, false);
    ADD_PARAMETER_NO_DEFAULT(std::string, leftAdapter);
    ADD_PARAMETER_NO_DEFAULT(std::string, rightAdapter);
};

//
// Control filtering options that control aspects of the control filtering algorithm
//
class PpaControlFilterConfig : public PacBio::Process::ConfigurationObject
{
public:
    // Disables control filtering, note that if this flag is
    // set to true, the control and control adapter sequences are
    // populated below if found but control filtering is not run.
    ADD_PARAMETER(bool, disableControlFiltering, true);
    ADD_PARAMETER_NO_DEFAULT(std::string, leftAdapter);
    ADD_PARAMETER_NO_DEFAULT(std::string, rightAdapter);
    ADD_PARAMETER_NO_DEFAULT(std::string, sequence);
};


class PpaHQRFConfig : public PacBio::Process::ConfigurationObject
{
public:
    // Customer-facing public enum whose string values are used in
    // the run metadata as well as command-line usage for selecting
    // which HQRF model to use.
    ADD_ENUM(PacBio::Primary::HqrfPublicMethod, method,PacBio::Primary::HqrfPublicMethod::DEFAULT);
};

class PpaOutputFilterConfig : public PacBio::Process::ConfigurationObject
{
public:
    // ZMW output stride for the BAM file
    ADD_PARAMETER(uint32_t, zmwOutputStride, 1);
};

class PpaAlgoConfig : public PacBio::Process::ConfigurationObject
{
    CONF_OBJ_SUPPORT_COPY(PpaAlgoConfig)
public:
    ADD_OBJECT(PpaInputFilterConfig, inputFilter);

    ADD_OBJECT(PpaAdapterFindingConfig, adapterFinding);

    ADD_OBJECT(PpaControlFilterConfig, controlFilter);

    ADD_OBJECT(PpaHQRFConfig, hqrf);

    ADD_OBJECT(PpaOutputFilterConfig, outputFilter);

public:
    void SetSequelDefaults()
    {
        inputFilter.minSnr.SetDefault(3.75f);
    }

    void SetSpiderDefaults()
    {
        inputFilter.minSnr.SetDefault(2.0f);
    }

public:
    void SetPlatformDefaults(const Platform& platform)
    {
        switch (platform)
        {
            case Platform::SEQUEL:
                SetSequelDefaults();
                break;
            case Platform::SEQUELII:
                SetSpiderDefaults();
                break;
            default:
                throw std::runtime_error("Only SEQUEL and SEQUELII platforms supported for PPA algo configuration");
                break;
        }
    }

    void Populate(const PacBio::BAM::CollectionMetadata& cmd)
    {
        SetSnrCut(cmd);
        SetAdapterSequences(cmd);
        SetControls(cmd);
        SetHQRF(cmd);
        SetZmwOutputStride(cmd);
    }

    void Populate(const PacBio::BAM::DataSet& ds)
    {
        SetSnrCut(ds);
        SetAdapterSequences(ds);
        SetControls(ds);
        SetHQRF(ds);
        SetZmwOutputStride(ds);
    }

    void Populate(const UserParameters* user)
    {
        SetSnrCut(user);
        SetAdapterSequences(user);
        SetControls(user);
        SetHQRF(user);
        SetZmwOutputStride(user);
    }

private:
    void SetSnrCut(const PacBio::BAM::DataSet& ds);
    void SetAdapterSequences(const PacBio::BAM::DataSet& ds);
    void SetControls(const PacBio::BAM::DataSet& ds);
    void SetHQRF(const PacBio::BAM::DataSet& ds);
    void SetZmwOutputStride(const PacBio::BAM::DataSet& ds);

    void SetSnrCut(const PacBio::BAM::CollectionMetadata& cmd);
    void SetAdapterSequences(const PacBio::BAM::CollectionMetadata& cmd);
    void SetControls(const PacBio::BAM::CollectionMetadata& cmd);
    void SetHQRF(const PacBio::BAM::CollectionMetadata& cmd);
    void SetZmwOutputStride(const PacBio::BAM::CollectionMetadata& cmd);

    void SetSnrCut(const UserParameters* user);
    void SetAdapterSequences(const UserParameters* user);
    void SetControls(const UserParameters* user);
    void SetHQRF(const UserParameters* user);
    void SetZmwOutputStride(const UserParameters* user);
};

}}} // PacBio::Primary::Postprimary


