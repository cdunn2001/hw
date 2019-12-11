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

#include <array>

namespace PacBio {
namespace Primary {

/// Provides methods to write, read, and verify SANITY blocks.
struct Sanity
{
public:
    typedef std::unique_ptr<std::FILE, int (*)(std::FILE*)> SmrtFilePtr;
    
public:
    static const uint8_t C0 = 0xD2;
    static const uint8_t C1 = 0x58;
    static const uint8_t C2 = 0x4C;
    static const uint8_t C3 = 0x52;

public:
    static uint64_t FindAndVerify(SmrtFilePtr& file)
    { return FindAndVerify(file.get()); }

    static uint64_t FindAndVerify(FILE* file)
    {
        while (!ReadAndVerify(file)) {}
        return std::ftell(file) - 4;
    }

    static bool ReadAndVerify(SmrtFilePtr& file)
    { return ReadAndVerify(file.get()); }

    static bool ReadAndVerify(FILE* file)
    {
        if (std::fgetc(file) != Sanity::C0) return false;
        if (std::fgetc(file) != Sanity::C1) return false;
        if (std::fgetc(file) != Sanity::C2) return false;
        if (std::fgetc(file) != Sanity::C3) return false;
        return true;
    }

    static bool ReadAndVerify(uint8_t* ptr)
    {
        if (*ptr != Sanity::C0) return false;
        if (*(ptr+1) != Sanity::C1) return false;
        if (*(ptr+2) != Sanity::C2) return false;
        if (*(ptr+3) != Sanity::C3) return false;
        return true;
    }

    static bool Write(SmrtFilePtr& file)
    {
        if (std::fputc(Sanity::C0, file.get()) != Sanity::C0) return false;
        if (std::fputc(Sanity::C1, file.get()) != Sanity::C1) return false;
        if (std::fputc(Sanity::C2, file.get()) != Sanity::C2) return false;
        if (std::fputc(Sanity::C3, file.get()) != Sanity::C3) return false;
        return true;
    }

    static void Write(std::array<uint8_t, 4>& data)
    {
        data[0] = C0;
        data[1] = C1;
        data[2] = C2;
        data[3] = C3;
    }

    static constexpr size_t SizeOf()
    { return 4; }

public: // structors
    Sanity() = delete;
    // Move constructor
    Sanity(Sanity&&) = delete;
    // Copy constructor
    Sanity(const Sanity&) = delete;
    // Move assignment operator
    Sanity& operator=(Sanity&&) = delete;
    // Copy assignment operator
    Sanity& operator=(const Sanity&) = delete;
    // Destructor
    ~Sanity() = delete;
};

}}
