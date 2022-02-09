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

// Programmer: Armin Töpfer

#pragma once

#include <errno.h>
#include <memory>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <pacbio/PBException.h>

namespace PacBio {
namespace Primary {

/// \brief Appends content of src vector to dst vector using move semantics.
/// \param[in] src Input vector that will be empty after execution
/// \param[in,out] dest Output vector that will be appended to
template <typename T>
static void MoveAppend(std::vector<T>& src, std::vector<T>& dst) noexcept
{
    if (dst.empty())
    {
        dst = std::move(src);
    }
    else
    {
        dst.reserve(dst.size() + src.size());
        std::move(src.begin(), src.end(), std::back_inserter(dst));
        src.clear();
    }
}
/// \brief Appends content of src vector to dst vector using move semantics.
/// \param[in] src Input vector via perfect forwarding
/// \param[in,out] dest Output vector that will be appended to
template <typename T>
static void MoveAppend(std::vector<T>&& src, std::vector<T>& dst) noexcept
{
    if (dst.empty())
    {
        dst = std::move(src);
    }
    else
    {
        dst.reserve(dst.size() + src.size());
        std::move(src.begin(), src.end(), std::back_inserter(dst));
        src.clear();
    }
}

// Encapsulate FILE* in a smart pointer that closes the file on destruction.
using SmrtFilePtr = std::unique_ptr<std::FILE, int (*)(std::FILE*)>;

// Encapsulate manually allocated byte stream memory in a smart pointer
// that frees on destruction.
template<typename T> using SmrtMemPtr = std::unique_ptr<T, void (*)(void*)>;
using SmrtBytePtr = SmrtMemPtr<uint8_t>;

class SmartMemory
{
public:
/// \brief Opens file stream
/// \details File stream is wrapped into an unique_ptr that takes care of
/// closing the file stream at destruction.
/// \param flags Flags for file stream
/// \return SmrtFilePtr
static SmrtFilePtr OpenFile(const char* fileName, const char* flags)
{ 
    auto* f = std::fopen(fileName, flags);
    if (f == nullptr) 
        throw PBException("Could not open file \""  + std::string(fileName) 
                          + "\" for mode:" + std::string(flags));
    return SmrtFilePtr(f, std::fclose); 
}

/// \brief CAllocates memory of type T for given number of occurrences.
/// \details Wraps allocated pointer in a unique_ptr 
///          and frees memory on destruction.
template<typename T>
static SmrtMemPtr<T> AllocMemPtr(const size_t occurrences)
{ 
    T* ptr = (T*)calloc(occurrences, sizeof(T));
    if (ptr == NULL) exit(ENOMEM);
    return SmrtMemPtr<T>(ptr, free); 
}

/// \brief CAllocates byte stream memory for given number of bytes.
/// \details Wraps allocated pointer in a unique_ptr 
///          and frees memory on destruction.
static SmrtBytePtr AllocBytePtr(const size_t count)
{ return AllocMemPtr<uint8_t>(count); }
};

}} // namespace
