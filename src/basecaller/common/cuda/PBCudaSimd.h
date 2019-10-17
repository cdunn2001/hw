// Copyright (c) 2019, Pacific Biosciences of California, Inc.
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

#ifndef PACBIO_CUDA_SIMD_H
#define PACBIO_CUDA_SIMD_H

#include <cstdint>

#include <cuda_fp16.h>
#include <common/cuda/CudaFunctionDecorators.h>

#include <vector_types.h>
#include <vector_functions.h>

namespace PacBio {
namespace Cuda {

// Aliasing for now, mostly so there is an easy hook if we want to either
// have a proper wrapper type like PBHalf2, or need to stub in third
// part half precision float for the host.
using PBHalf = half;

class PBShort2
{
    // 16 least significant bits hold X, the
    // 16 most significant hold Y
    static constexpr uint32_t yshift = 0x10;
    static constexpr uint32_t ymask = 0xFFFF0000;
    static constexpr uint32_t xmask = 0x0000FFFF;
public:
    PBShort2() = default;

    CUDA_ENABLED PBShort2(short s) : PBShort2(s,s) {}
    CUDA_ENABLED PBShort2(short s1, short s2)
        // We have to be careful about sign extension, as bit operations automatically promote
        // operands to 32 bit.  So we have to make sure to mask out the unwanted bits in s1
        // after the promotion to full width.
        : data_{ (static_cast<uint32_t>(s1) & xmask ) | (static_cast<uint32_t>(s2) << yshift) }
    {}

private:
    // We need to be able to construct from a raw uint32_t, to capture the return from various
    // cuda intrinsics.  However adding a constructor that just takes that introduces confusion,
    // whenever handing something like an integer literal in.  PBShort2(12) could potentially
    // mean cast the 12 to a short, and construct a PBShort2 with both slots set to 12, or cast
    // 12 to a uint32_t, which is effectively y=0,x=12.
    // Making this ctor private and adding a Dummy type to disambiguate the signature.  Construction
    // from a raw uint32_t must be done through the `FromRaw` static named constructor, though
    // probably no one outside PBCudaSimd.cuh needs do this.
    struct Dummy {};
    CUDA_ENABLED PBShort2(uint32_t data, Dummy)
        : data_{data}
    {}

public:
    CUDA_ENABLED static PBShort2 FromRaw(uint32_t raw) { return PBShort2(raw, Dummy{}); }

    // Set/get individual elements
    CUDA_ENABLED void X(short s) { data_ = (data_ & ymask) | (static_cast<uint32_t>(s) & xmask); }
    CUDA_ENABLED void Y(short s) { data_ = (data_ & xmask) | (static_cast<uint32_t>(s) << yshift); }
    CUDA_ENABLED short X() const {return static_cast<short>(data_ & xmask); }
    CUDA_ENABLED short Y() const {return static_cast<short>((data_ & ymask) >> yshift); }

    template <int id>
    CUDA_ENABLED short Get() const
    {
        static_assert(id < 2, "Out of bounds access in PBShort2");
        if (id == 0) return X();
        else return Y();
    }

    uint32_t CUDA_ENABLED data() const { return data_; }

private:
    uint32_t data_;
};

class PBHalf2
{
public:
    PBHalf2() = default;

    CUDA_ENABLED PBHalf2(float f) : data_{__float2half2_rn(f)} {}
    CUDA_ENABLED PBHalf2(float f1, float f2) : data_{__floats2half2_rn(f1, f2)} {}
    CUDA_ENABLED PBHalf2(PBShort2 f) : PBHalf2(static_cast<float>(f.X()), static_cast<float>(f.Y())) {}
    CUDA_ENABLED PBHalf2(uint2 f) : PBHalf2(static_cast<float>(f.x), static_cast<float>(f.y)) {}
    CUDA_ENABLED PBHalf2(half f)  : data_{f,f} {}
    CUDA_ENABLED PBHalf2(half2 f) : data_{f} {}
#if defined(__CUDA_ARCH__)
    __device__ PBHalf2(float2 f) : data_{__float22half2_rn(f)} {}
#else
    PBHalf2(float2 f) : data_{__floats2half2_rn(f.x, f.y)} {}
#endif

    // Set/get individual elements
    CUDA_ENABLED void X(half f) {data_.x = f; }
    CUDA_ENABLED void Y(half f) {data_.y = f; }
    CUDA_ENABLED half X() const {return data_.x; }
    CUDA_ENABLED half Y() const {return data_.y; }

    // Helper functions to ease float/half incompatability
    CUDA_ENABLED void X(float f) {data_.x = __float2half(f); }
    CUDA_ENABLED void Y(float f) {data_.y = __float2half(f); }
    CUDA_ENABLED float FloatX() const { return __half2float(data_.x); }
    CUDA_ENABLED float FloatY() const { return __half2float(data_.y); }

    template <int id>
    CUDA_ENABLED float Get() const
    {
        static_assert(id < 2, "Out of bounds access in PBHalf2");
        if (id == 0) return FloatX();
        else return FloatY();
    }

    half2 CUDA_ENABLED data() const { return data_; }
private:
    half2 data_;
};

class PBBool2
{
public:
    PBBool2() = default;

    CUDA_ENABLED PBBool2(bool b) : data_{__float2half2_rn(b ? 1.0f : 0.0f)} {}
    CUDA_ENABLED PBBool2(bool b1, bool b2) : data_{ __floats2half2_rn(b1 ? 1.0f : 0.0f,
                                                                      b2 ? 1.0f : 0.0f)} {}
#ifdef __CUDA_ARCH__
    __device__ bool X() const { return data_.x != __short_as_half(0); }
    __device__ bool Y() const { return data_.y != __short_as_half(0); }
#else
    bool X() const { return __half2float(data_.x) != 0.0f; }
    bool Y() const { return __half2float(data_.y) != 0.0f; }
#endif

    CUDA_ENABLED PBBool2(half2 cond) : data_{cond} {}
    half2 CUDA_ENABLED data() const { return data_; }
private:
    half2 data_;
};

}}

#endif // PACBIO_CUDA_SIMD_H
