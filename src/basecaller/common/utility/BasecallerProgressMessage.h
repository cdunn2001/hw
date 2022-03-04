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

#ifndef PACBIO_UTILITY_BASECALLER_PROGRESS_MESSAGE_H
#define PACBIO_UTILITY_BASECALLER_PROGRESS_MESSAGE_H

#include <common/utility/ProgressMessage.h>

namespace PacBio::Utility
{

SMART_ENUM(SmrtBasecallerStages,
        Start,
        CreateSource,
        CreateRepacker,
        CreateTraceSaver,
        CreateBasecaller,
        CreatePrelimHQFilter,
        CreateBazSaver,
        SourceReady,
        Analyze,
        FlushOutput);

using SmrtBasecallerProgressMessage = PacBio::Utility::ProgressMessage<SmrtBasecallerStages>;

SmrtBasecallerProgressMessage::Table stages = {
        { "Start",                      { 0, 0, 1} },
        { "CreateSource",       { 0, 1, 20  } },
        { "CreateRepacker",     { 0, 2, 5   } },
        { "CreateTraceSaver",   { 0, 3, 10  } },
        { "CreateBasecaller",   { 0, 4, 10  } },
        { "CreatePrelimHQFilter",   { 0, 5, 10  } },
        { "CreateBazSaver",       { 0, 6, 20  } },
        { "SourceReady",       { 1, 7, 20  } },
        { "Analyze",       { 1, 8, 20  } },
        { "FlushOutput",       { 1, 9, 20  } }
};

} // namespace PacBio::Utility

#endif // PACBIO_UTILITY_BASECALLER_PROGRESS_MESSAGE_H
