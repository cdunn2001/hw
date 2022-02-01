//
//  Copyright (c) 2011-2015, Pacific Biosciences of California, Inc.
//
//  All rights reserved.
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of Pacific Biosciences nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
//  NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
//  THIS LICENSE.  THIS SOFTWARE IS PROVIDED BY PACIFIC BIOSCIENCES AND ITS
//  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//  PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL PACIFIC BIOSCIENCES OR
//  ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
//  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
//  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
//  BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
//  IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
//  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.
//

#ifndef PULSELOOKBACK_HPP
#define PULSELOOKBACK_HPP

#include <cassert>
#include <deque>
#include <pacbio/smrtdata/Basecall.h>

namespace PacBio {
namespace Primary {

/// Class to maintain a fixed-size lookback in the pulses emitted in a ZMW
/// Enables indexing using negative numbers to look at the -1 pulse, -2 pulse, etc.
template <int LookbackSize>
class PulseLookback
{
public:
    PulseLookback() 
        : size_(0)
        , data_{}
    {}
    
    ~PulseLookback() {}
      
    void PushBack(const SmrtData::Basecall& pulse)
    {
        if (size_ < LookbackSize) {
            size_++;
        }
        // Just making sure a developer sees this if for some reason we change
        // the lookback.  I assume any value someone might choose will be
        // fine, but a large LookbackSize may have bad performance characteristics
        // so I want a developer to have to explicitly change/remove this assert!
        static_assert(LookbackSize == 2, "Unsupported Lookback size.  See comments");

        // Will do excess work while first populating, but whatever, that's
        // transient
        for (size_t i = LookbackSize-1; i > 0; --i)
        {
            data_[i] = std::move(data_[i-1]);
        }
        data_[0] = pulse;
    }

    const SmrtData::Basecall& GetPreviousPulse(int offset) const
    {
        // Storage order is going backwards in time, with offset=0 being our
        // first element
        return data_[offset];
    }
    
    /// The current size_, which may be less than LookbackSize, for example at the
    /// beginning of trace processing.
    size_t Size() const
    {
        return size_;
    }
    
private:
    size_t size_;
    std::array<SmrtData::Basecall, LookbackSize> data_;
};


}} // ::PacBio::Primary


#endif // PULSELOOKBACK_HPP
