// Copyright (c) 2015-2018, Pacific Biosciences of California, Inc.
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

#include "HQRegionFinder.h"
#include "HQRegionFinderModels.h"
#include "HQRegionFinderParams.h"
#include "BlockHQRegionFinder.h"
#include "StaticHQRegionFinder.h"

namespace PacBio {
namespace Primary {
namespace Postprimary {

// Evaluates a polynomial with a list of coefficients by descending degree
// (e.g. y = ax^2 + bx + c)
float evaluatePolynomial(const std::vector<float>& coeff, float x)
{
    float y = coeff[0];
    for (unsigned int i = 1; i < coeff.size(); i++)
        y = y * x + coeff[i];
    return y;
}

RegionLabel HQRegionFinder::FindAndAnnotateHQRegion(
        const BlockLevelMetrics& metrics,
        const EventData &zmw) const
{
    size_t pulseBegin;
    size_t pulseEnd;
    std::tie(pulseBegin, pulseEnd) = FindHQRegion(metrics, zmw);
    assert(pulseBegin <= zmw.NumEvents());
    assert(pulseEnd <= zmw.NumEvents());

    const auto& isBase = zmw.IsBase();
    size_t baseBegin = std::count(isBase.begin(), isBase.begin() + pulseBegin, true);
    size_t baseEnd = std::count(isBase.begin(), isBase.begin() + pulseEnd, true);

    RegionLabel hqRegion(baseBegin, baseEnd, 100, RegionLabelType::HQREGION);

    if (baseBegin == 0 && baseEnd == 0)
    {
        // Need to make sure an empty HQ region is annotated correctly in pulses
        // as well.  (The 0th base might be still be the nth pulse)
        hqRegion.pulseBegin = 0;
        hqRegion.pulseEnd = 0;
    }
    else
    {
        // Save the pulse index (which are immutable) for the HQ-region. This
        // is so we can calculate metrics over the original HQ-region before
        // the burst filter. The burst filter can overlap the boundaries
        // of the HQ-region and potentially shift left-ward the base indices
        // (if the HQ-region base indices get turned into pulses) so
        // we retain the original pulse indices which will not change regardless if bases
        // are overturned into pulses.
        hqRegion.pulseBegin = (zmw.NumBases() != 0) ? pulseBegin : 0;
        hqRegion.pulseEnd = (zmw.NumBases() != 0) ? pulseEnd : 0;
    }
    return hqRegion;
}

std::unique_ptr<HQRegionFinder> HQRegionFinderFactory(
        const UserParameters& user,
        const std::shared_ptr<PpaAlgoConfig>& ppaAlgoConfig,
        double frameRateHz)
{
    HQRFMethod hqrfMethod = CoeffLookup(ppaAlgoConfig);

    auto frameRate = static_cast<float>(frameRateHz);
    auto snrThresh = ppaAlgoConfig->inputFilter.minSnr;
    auto ignoreBazAL = user.ignoreBazAL;

    // Change to make_unique and direct return after c++14?
    std::unique_ptr<HQRegionFinder> ret;
    if (user.hqrf && user.fakeHQ.empty())
    {
        switch (hqrfMethod)
        {
            case HQRFMethod::SPIDER_CRF_HMM:
            {
                PBLOG_INFO << "SPIDER_CRF_HMM configured";
                ret.reset(new BlockHQRegionFinder<HQRFMethod::SPIDER_CRF_HMM>(frameRate, snrThresh, ignoreBazAL));
                break;
            }
            case HQRFMethod::SEQUEL_CRF_HMM:
            {
                PBLOG_INFO << "SEQUEL_CRF_HMM configured";
                ret.reset(new BlockHQRegionFinder<HQRFMethod::SEQUEL_CRF_HMM>(frameRate, snrThresh, ignoreBazAL));
                break;
            }
            case HQRFMethod::TRAINED_CART_HMM:
            {
                PBLOG_INFO << "TRAINED_CART_HMM configured";
                ret.reset(new BlockHQRegionFinder<HQRFMethod::TRAINED_CART_HMM>(frameRate, snrThresh, ignoreBazAL));
                break;
            }
            case HQRFMethod::ZOFFSET_CRF_HMM:
            {
                PBLOG_INFO << "ZOFFSET_CRF_HMM configured";
                ret.reset(new BlockHQRegionFinder<HQRFMethod::ZOFFSET_CRF_HMM>(frameRate, snrThresh, ignoreBazAL));
                break;
            }
            case HQRFMethod::TRAINED_CART_CART:
            {
                PBLOG_INFO << "TRAINED_CART_CART configured";
                ret.reset(new BlockHQRegionFinder<HQRFMethod::TRAINED_CART_CART>(frameRate, snrThresh, ignoreBazAL));
                break;
            }
            case HQRFMethod::BAZ_HMM:
            {
                PBLOG_INFO << "BAZ_HMM configured";
                ret.reset(new BlockHQRegionFinder<HQRFMethod::BAZ_HMM>(frameRate, snrThresh, ignoreBazAL));
                break;
            }
            case HQRFMethod::ZOFFSET_CART_HMM:
            {
                PBLOG_INFO << "ZOFFSET_CART_HMM configured";
                ret.reset(new BlockHQRegionFinder<HQRFMethod::ZOFFSET_CART_HMM>(frameRate, snrThresh, ignoreBazAL));
                break;
            }
        }
    } else {
        PBLOG_INFO << "Using fakeHQ";
        ret.reset(new StaticHQRegionFinder(user.fakeHQ));
    }
    return ret;
}

}}} // ::PacBio::Primary::Postprimary

