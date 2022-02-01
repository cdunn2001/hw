// Copyright (c) 2018, Pacific Biosciences of California, Inc.
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

#ifndef PACBIO_POSTPRIMARY_SAFEOPTIONAL_H
#define PACBIO_POSTPRIMARY_SAFEOPTIONAL_H

#include <boost/optional.hpp>

namespace PacBio {
namespace Primary {
namespace Postprimary {

// Wrapper for the boost::optional class, that does not implement relational
// operators or allow implicit construction.  The raw boost::optional type
// does, which allows something like the following easy typo bug to actually
// compile and run, which we'd like to avoid since we'll likely be comparing
// the optional frequently with native types:
//
// int someMagicNumber1 = 12;
// int someMagicNumber2 = 24;
// boost optional<int> opt = SomeComputation();
// if (opt) // We're making sure the optional is not "null"
// {
//   Apply the `*` operator to get at the underlying value
//   if (*opt < someMagicNumber1)
//      /* Do some work */
// }
//
// Oops, we didn't apply the `*` operator.  Maybe we just forgot to, and maybe
// we forgot `opt` is an optional and is maybe empty. Now someMagicNumber2 just
// got implicitly converted to an optional<int>, if opt is empty, then this
// will evaluate to true, instead of causing a compiler error that makes us
// explicitly handle the optionalness of `opt`
// if (opt > someMagicNumber2)
//   /* Do something maybe unexpected! */
// }
template <typename T>
class PBOptional
{
public:
    PBOptional() = default;
    PBOptional(const PBOptional&) = default;
    PBOptional(PBOptional&&) = default;
    PBOptional& operator=(const PBOptional&) = default;
    PBOptional& operator=(PBOptional&&) = default;

    PBOptional& operator=(const T& data) { this->data_ = data; return *this; }
    PBOptional& operator=(T&& data) { this->data_ = std::move(data); return *this; }

    explicit operator bool() const { return (bool)data_; }

    const T* operator->() const { return &(*data_); }
    const T& operator*() const { return *data_; }
    T* operator->() { return &(*data_); }

    T get_or(const T& def) const { return data_ ? *data_ : def; }
    T& DefaultInit() { data_ = T{}; return *data_; }
    void Reset() { data_ = boost::none; }

protected:
    boost::optional<T> data_;
};

}}}

#endif // PACBIO_POSTPRIMARY_SAFEOPTIONAL_H
