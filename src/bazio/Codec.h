// Copyright (c) 2014-2016, Pacific Biosciences of California, Inc.
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

#include <algorithm>
#include <cassert>
#include <cmath>

#include <x86intrin.h>

namespace PacBio {
namespace Primary {

class Codec {
public:
    Codec()
    {
        static constexpr int B = 2;
        static constexpr int t = 6;

        double T = std::pow(B, t);

        int next = 0;
        double grain;
        const int end = static_cast<int>(256/T);
        for (int i = 0; i < end; ++i) {
            grain = std::pow(B, i);
            std::vector<uint16_t> nextOnes;
            for (double j = 0; j < T; ++j)
                nextOnes.push_back(j*grain + next);
            next = static_cast<int>(nextOnes.back() + grain);
            framepoints.insert(framepoints.end(), nextOnes.cbegin(), nextOnes.cend());
        }
        assert(framepoints.size()-1 <= UINT8_MAX);

        const uint16_t maxElement = (*max_element(framepoints.cbegin(), framepoints.cend()));
        frameToCode.assign(maxElement+1, 0);

        const int fpEnd = framepoints.size() - 1;
        uint8_t i = 0;
        uint16_t fl = 0;
        uint16_t fu = 0;
        for (; i < fpEnd; ++i) {
            fl = framepoints[i];
            fu = framepoints[i+1];
            if (fu > fl+1) {
                const int middle = (fl+fu)/2;
                for (int f = fl; f < middle; ++f)
                    frameToCode[f] = i;
                for (int f = middle; f < fu; ++f)
                    frameToCode[f] = static_cast<uint8_t>(i+1);
            } else
                frameToCode[fl] = i;
        }

        // this next line differs from the python implementation (there, it's "i+1")
        // our C++ for loop has incremented our index counter one more time than the indexes from python enumerate(...)
        frameToCode[fu] = i;
        maxFramepoint = fu;
    }

    uint16_t CodeToFrame(const uint8_t code) const
    {
        assert(framepoints.size() == 256);
        return framepoints[code];
    }

    uint8_t FrameToCode(const uint16_t frame) const
    {
        assert(frameToCode.size() >= maxFramepoint);
        return frameToCode[std::min(maxFramepoint, frame)];
    }

    uint8_t DownsampleFrame(const uint16_t frame) const
    {
        static constexpr uint16_t encodedMax = 255;
        return static_cast<uint8_t>(std::min(framepoints[FrameToCode(frame)], encodedMax));
    }

    // Some work was done trying to see if a bit-twiddle implementation could
    // be faster than the lookup tables currently in use.  They did not win out
    // when used in the basewriter code paths, *but* they were not a ton slower.
    // There is a slight chance basewriter will be re-written to be more
    // vectorizable friendly, in which case the below can serve as a reference
    // implementation for that.  I think the odds of this functionlity getting
    // vectorized are low, but were that to happen I wouldn't want to have to
    // re-invent the code below, so I'm preserving it in this `Experimental`
    // struct.  Unit tests exist to make sure this implementation produces
    // the same results as the lookup table version.
    struct Experimental
    {
        static uint16_t CodeToFrame(const uint8_t code)
        {
            const auto group = code / 64;
            const auto rem = code % 64;
            auto tmp = 1 << group;
            auto tmp2 = (tmp - 1) << 6;
            return static_cast<uint16_t>(tmp2 + rem * tmp);
        }

        static uint8_t FrameToCode(const uint16_t frame)
        {
            if (frame > 952) return 255;
            auto group = _bit_scan_reverse((frame + 64) >> 6); // 0,1,2,3
            auto power = 1 << group; // 1, 2, 4, 8
            auto base = (power - 1) << 6;
            return static_cast<uint8_t>(64 * group + ((frame - base + (power >> 1)) >> group));
        }

        static uint8_t DownsampleFrame(const uint16_t frame)
        {
            auto dist = 1 << _bit_scan_reverse((frame + 64) >> 6); // 1,2,4,8
            auto rem = frame & (dist - 1);
            auto adjust = rem >= (dist >> 1) ? dist - rem - 1/dist : -rem;
            return static_cast<uint8_t>(std::min(frame + adjust, 255));
        }
    };
private:
    std::vector<uint16_t> framepoints;
    std::vector<uint8_t> frameToCode;
    uint16_t maxFramepoint;
};

}}
