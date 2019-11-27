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
/// \brief A class that calculates statistics on a variable, such as a rate calculation


#ifndef SEQUEL_STATISTIC_H
#define SEQUEL_STATISTIC_H

#include <chrono>
#include <string>
#include <pacbio/text/String.h>

/// class for calculating the rate of a "linearly" increasing integer value.
class Statistic
{
public:
    uint64_t value{0};
    mutable uint64_t valuePrevious{0};
    mutable std::chrono::system_clock::time_point t0;
    Statistic() : value{0} {}
    Statistic(uint64_t v) : value{v} {}
    Statistic& operator++() { ++value; return *this;}
    Statistic& operator++(int ) { value ++; return *this;}
    Statistic& operator+=(uint64_t x) { value += x; return *this;}
    Statistic& operator=(uint64_t x) { value = x; return *this;}
    operator uint64_t() const { return value;}
    operator std::string() const { return ToString();}
    double Rate() const
    {
        using FpSeconds = std::chrono::duration<double, std::chrono::seconds::period>;
        static_assert(std::chrono::treat_as_floating_point<FpSeconds::rep>::value,
                      "Rep required to be floating point");
        auto t = std::chrono::system_clock::now();
        double deltaTime = FpSeconds(t - t0).count();
        double r;
        if (deltaTime !=0)
        {
            r = (value - valuePrevious) / deltaTime;
            valuePrevious = value;
            t0 = t;
        }
        else
        {
            r = 0.0;
        }
        return r;
    }
    std::string ToString() const
    {
        std::string s = PacBio::Text::String::Format("%ld (%.3g/sec)",value,Rate());
        return s;
    }
};



#endif //SEQUEL_STATISTIC_H
