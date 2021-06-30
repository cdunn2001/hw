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

#ifndef PACBIO_COMMON_UTILITY_OVERLOAD_H
#define PACBIO_COMMON_UTILITY_OVERLOAD_H

#include <utility>

namespace PacBio {
namespace Utility {

template <typename... Fs>
struct Overload;

template <typename F, typename...Fs>
struct Overload<F, Fs...> : F, Overload<Fs...>
{
    Overload(F f, Fs... fs) : F(std::move(f)), Overload<Fs...>(std::move(fs)...) {}
    using F::operator();
    using Overload<Fs...>::operator();
};

template <typename F>
struct Overload<F> : F
{
    Overload(F f) : F(std::move(f)) {}
    using F::operator();
};

template <typename... Fs>
auto make_overload(Fs... fs)
{
    return Overload<Fs...>(fs...);
}

// Note: this is nicely simpler in C++17...
// template<class... Ts> struct Overload : Ts... { using Ts::operator()...; };
// template<class... Ts> Overload(Ts...) -> Overload<Ts...>;



}}

#endif //PACBIO_COMMON_UTILITY_OVERLOAD_H
