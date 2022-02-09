// Copyright (c) 2021, Pacific Biosciences of California, Inc.
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
///  \brief Platform dependent initializations.
//
// Programmer: Mark Lakata

#include "PaWsConfig.h"
#include <assert.h>

namespace PacBio {
namespace Primary {
namespace PaWs {

using namespace PacBio::Sensor;

// static size_t GiB = 0x4000'0000ULL;

void Sequel2DefaultConfig(PaWsConfig* config)
{
    assert(config);
    config->socketIds = std::vector<std::string>{"1"};
}

void KestrelConfig(PaWsConfig* config)
{
    assert(config);
    config->socketIds = std::vector<std::string>{"1", "2", "3", "4"};
}

void FactoryConfig(PaWsConfig* config)
{
    assert(config);
    switch(config->platform)
    {
        case Platform::Sequel2Lvl1:
        case Platform::Sequel2Lvl2:
            Sequel2DefaultConfig(config); 
            break;
        case Platform::Kestrel:
            KestrelConfig(config); 
            break;
        case Platform::Mongo:
        default:
        PBLOG_WARN << "Can't do a factory reset for platform:" << config->platform.toString();
    }
}

}}} // namespace
