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

#include <pacbio/logging/Logger.h>


#include "PpaAlgoConfig.h"

using namespace PacBio::Primary::Postprimary;

void PpaAlgoConfig::SetSnrCut(const PacBio::BAM::CollectionMetadata& cmd)
{
    if (cmd.HasAutomationParameters() && cmd.AutomationParameters().HasSNRCut())
    {
        inputFilter.minSnr = static_cast<float>(cmd.AutomationParameters().SNRCut());
        PBLOG_DEBUG << "Setting minSnr using collection metadata to = " << inputFilter.minSnr;
    }
    else
    {
        PBLOG_WARN << "minSnr not found in collection metadata";
    }
}

void PpaAlgoConfig::SetSnrCut(const PacBio::BAM::DataSet& ds)
{
    if (ds.Metadata().CollectionMetadata().HasAutomationParameters() &&
        ds.Metadata().CollectionMetadata().AutomationParameters().HasSNRCut())
    {
        inputFilter.minSnr = static_cast<float>(ds.Metadata().CollectionMetadata().AutomationParameters().SNRCut());
        PBLOG_DEBUG << "Setting minSnr using collection metadata to = " << inputFilter.minSnr;
    }
    else
    {
        PBLOG_WARN << "minSnr not found in collection metadata";
    }
}

void PpaAlgoConfig::SetSnrCut(const UserParameters* user)
{
    if (!std::isnan(user->minSnr))
    {
        inputFilter.minSnr = user->minSnr;
        PBLOG_INFO << "Overriding using command-line minSnr = " << inputFilter.minSnr;
    }
}

void PpaAlgoConfig::SetAdapterSequences(const PacBio::BAM::CollectionMetadata& cmd)
{
    if (cmd.HasTemplatePrepKit())
    {
        if (cmd.TemplatePrepKit().HasLeftAdaptorSequence())
        {
            adapterFinding.leftAdapter = cmd.TemplatePrepKit().LeftAdaptorSequence();
            PBLOG_DEBUG << "Setting left adapter for adapter finding using collection metadata to = "
                        << adapterFinding.leftAdapter;
        }
        else
        {
            const std::string errMsg = "Left adapter for template prep kit not found in collection metadata";
            PBLOG_ERROR << errMsg;
            throw std::runtime_error(errMsg);
        }
        if (cmd.TemplatePrepKit().HasRightAdaptorSequence())
        {
            adapterFinding.rightAdapter = cmd.TemplatePrepKit().RightAdaptorSequence();
            PBLOG_DEBUG << "Setting right adapter for adapter finding using collection metadata to = "
                        << adapterFinding.rightAdapter;
        }
        else
        {
            const std::string errMsg = "Right adapter for template prep kit not found in collection metadata";
            PBLOG_ERROR << errMsg;
            throw std::runtime_error(errMsg);
        }
    }
    else
    {
        const std::string errMsg = "Template prep kit not found in collection metadata";
        PBLOG_ERROR << errMsg;
        throw std::runtime_error(errMsg);
    }
}

void PpaAlgoConfig::SetAdapterSequences(const PacBio::BAM::DataSet& ds)
{
    if (ds.Metadata().CollectionMetadata().HasTemplatePrepKit())
    {
        if (ds.Metadata().CollectionMetadata().TemplatePrepKit().HasLeftAdaptorSequence())
        {
            adapterFinding.leftAdapter = ds.Metadata().CollectionMetadata().TemplatePrepKit().LeftAdaptorSequence();
            PBLOG_DEBUG << "Setting left adapter for adapter finding using collection metadata to = "
                        << adapterFinding.leftAdapter;
        }
        else
        {
            const std::string errMsg = "Left adapter for template prep kit not found in collection metadata";
            PBLOG_ERROR << errMsg;
            throw std::runtime_error(errMsg);
        }
        if (ds.Metadata().CollectionMetadata().TemplatePrepKit().HasRightAdaptorSequence())
        {
            adapterFinding.rightAdapter = ds.Metadata().CollectionMetadata().TemplatePrepKit().RightAdaptorSequence();
            PBLOG_DEBUG << "Setting right adapter for adapter finding using collection metadata to = "
                        << adapterFinding.rightAdapter;
        }
        else
        {
            const std::string errMsg = "Right adapter for template prep kit not found in collection metadata";
            PBLOG_ERROR << errMsg;
            throw std::runtime_error(errMsg);
        }
    }
    else
    {
        const std::string errMsg = "Template prep kit not found in collection metadata";
        PBLOG_ERROR << errMsg;
        throw std::runtime_error(errMsg);
    }
}

void PpaAlgoConfig::SetAdapterSequences(const UserParameters* user)
{
    if (!user->adaptersFilePath.empty())
    {
        adapterFinding.disableAdapterFinding = false;
        PBLOG_DEBUG << "Adapter finding enabled since user specified adapter file path";

        std::vector<FastaEntry> adapterList = FastaUtilities::ParseSingleFastaFile(user->adaptersFilePath);
        if (adapterList.empty())
            throw std::runtime_error("Adapter file is empty");
        if (adapterList.size() > 2)
            throw std::runtime_error("Adapter file contains more than two sequences");
        // Check for uniqueness
        if (adapterList.size() == 2
            && adapterList.at(0).sequence == adapterList.at(1).sequence)
            throw std::runtime_error("Adapter file contains identical sequences. Sequences have to be unique.");
        for (const auto& adapter : adapterList)
            if (adapter.sequence.size() > 100)
                PBLOG_WARN << "Adapter sequence is longer than 100 bp! "
                           << "Please make sure you are using the correct adapter fasta file";

        adapterFinding.leftAdapter = adapterList.at(0).sequence;
        if (adapterList.size() == 1)
            adapterFinding.rightAdapter = adapterList.at(0).sequence;
        else
            adapterFinding.rightAdapter = adapterList.at(1).sequence;

        PBLOG_DEBUG << "Overriding and setting left adapter for adapter finding using command-line to = "
                    << adapterFinding.leftAdapter;
        PBLOG_DEBUG << "Overriding and setting right adapter for adapter finding using command-line to = "
                    << adapterFinding.rightAdapter;
    }

    if (user->disableAdapterFinding)
    {
        PBLOG_DEBUG << "Adapter finding disabled on command-line, will not run";
        adapterFinding.disableAdapterFinding = true;
    }
}

void PpaAlgoConfig::SetControls(const PacBio::BAM::CollectionMetadata& cmd)
{
    if (cmd.HasControlKit())
    {
        if (cmd.ControlKit().HasLeftAdapter() &&
            cmd.ControlKit().HasRightAdapter() &&
            cmd.ControlKit().HasSequence())
        {
            controlFilter.disableControlFiltering = false;
            PBLOG_DEBUG << "Control filtering enabled due to control kit found in collection metadata";

            controlFilter.leftAdapter = cmd.ControlKit().LeftAdapter();
            controlFilter.rightAdapter = cmd.ControlKit().RightAdapter();
            controlFilter.sequence = cmd.ControlKit().Sequence();
            PBLOG_DEBUG << "Setting control adapters and sequence using collection metadata";
        }
        else
        {
            const std::string errMsg = "Missing sequences from control kit";
            PBLOG_ERROR << errMsg;
            throw std::runtime_error(errMsg);
        }
    }
}

void PpaAlgoConfig::SetControls(const PacBio::BAM::DataSet& ds)
{
    if (ds.Metadata().CollectionMetadata().HasControlKit())
    {
        if (ds.Metadata().CollectionMetadata().ControlKit().HasLeftAdapter() &&
            ds.Metadata().CollectionMetadata().ControlKit().HasRightAdapter() &&
            ds.Metadata().CollectionMetadata().ControlKit().HasSequence())
        {
            controlFilter.disableControlFiltering = false;
            PBLOG_DEBUG << "Control filtering enabled due to control kit found in collection metadata";

            controlFilter.leftAdapter = ds.Metadata().CollectionMetadata().ControlKit().LeftAdapter();
            controlFilter.rightAdapter = ds.Metadata().CollectionMetadata().ControlKit().RightAdapter();
            controlFilter.sequence = ds.Metadata().CollectionMetadata().ControlKit().Sequence();
            PBLOG_DEBUG << "Setting control adapters and sequence using collection metadata";
        }
        else
        {
            const std::string errMsg = "Missing sequences from control kit";
            PBLOG_ERROR << errMsg;
            throw std::runtime_error(errMsg);
        }
    }
}

void PpaAlgoConfig::SetControls(const UserParameters* user)
{
    if(!user->controlFilePath.empty() && !user->controlAdaptersFilePath.empty())
    {
        controlFilter.disableControlFiltering = false;
        PBLOG_DEBUG << "Control filtering enabled since user specified control and control adapter file paths";

        std::vector<FastaEntry> controlList = FastaUtilities::ParseSingleFastaFile(user->controlFilePath);
        if (controlList.empty())
            throw std::runtime_error("Control list is empty");

        controlFilter.sequence = controlList.at(0).sequence;

        std::vector<FastaEntry> controlAdapterList = FastaUtilities::ParseSingleFastaFile(user->controlAdaptersFilePath);
        if (controlAdapterList.empty())
            throw std::runtime_error("Control adapter file is empty");
        if (controlAdapterList.size() > 2)
            throw std::runtime_error("Control adapter file contains more than two sequences");
        // Check for uniqueness
        if (controlAdapterList.size() == 2
            && controlAdapterList.at(0).sequence == controlAdapterList.at(1).sequence)
            throw std::runtime_error("Control adapter file contains identical sequences. Sequences have to be unique.");
        for (const auto& adapter : controlAdapterList)
            if (adapter.sequence.size() > 100)
                PBLOG_WARN << "Control adapter sequence is longer than 100 bp! "
                           << "Please make sure you are using the correct adapter fasta file";

        controlFilter.leftAdapter = controlAdapterList.at(0).sequence;
        if (controlAdapterList.size() == 1)
            controlFilter.rightAdapter = controlAdapterList.at(0).sequence;
        else
            controlFilter.rightAdapter = controlAdapterList.at(1).sequence;

        PBLOG_DEBUG << "Overriding control adapters and sequences using command-line";
    }

    if (user->disableControlFiltering)
    {
        PBLOG_DEBUG << "Control filtering disabled on command-line, will not run";
        controlFilter.disableControlFiltering = true;
    }
}

void PpaAlgoConfig::SetHQRF(const PacBio::BAM::CollectionMetadata& cmd)
{
    if (cmd.HasAutomationParameters() && cmd.AutomationParameters().HasHQRFMethod())
    {
        hqrf.method = PacBio::Primary::HqrfPublicMethod::fromString(cmd.AutomationParameters().HQRFMethod());
        PBLOG_DEBUG << "Setting hqrf method using collection metadata to = " << hqrf.method().toString();
    }
    else
    {
        PBLOG_WARN << "HQRFMethod not found in collection metadata";
    }
}

void PpaAlgoConfig::SetHQRF(const PacBio::BAM::DataSet& ds)
{
    if (ds.Metadata().CollectionMetadata().HasAutomationParameters() &&
        ds.Metadata().CollectionMetadata().AutomationParameters().HasHQRFMethod())
    {
        hqrf.method = PacBio::Primary::HqrfPublicMethod::fromString(
                ds.Metadata().CollectionMetadata().AutomationParameters().HQRFMethod());
        PBLOG_DEBUG << "Setting hqrf method using collection metadata to = " << hqrf.method().toString();
    }
    else
    {
        PBLOG_WARN << "HQRFMethod not found in collection metadata";
    }
}

void PpaAlgoConfig::SetHQRF(const UserParameters* user)
{
    if (!user->hqrfMethod.empty())
    {
        hqrf.method = PacBio::Primary::HqrfPublicMethod::fromString(user->hqrfMethod);
        PBLOG_DEBUG << "Overriding hqrf method using command-line to = " << hqrf.method().toString();
    }
}
