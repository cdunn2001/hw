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

#ifndef PACBIO_CUDA_CUDA_ARRAY_H_
#define PACBIO_CUDA_CUDA_ARRAY_H_

#include <common/cuda/CudaFunctionDecorators.h>

#include "assert.h"
#include <cstring>
#include <algorithm>
#include <array>
#include <iterator>

namespace PacBio {
namespace Cuda {
namespace Utility {

/// A CUDA-friendly replacement for std::array.
template <typename T, size_t len>
struct CudaArray
{
    using value_type = T;

    CudaArray() = default;

    // implicit conversion from std::array intentional
    CudaArray(const std::array<T, len>& data)
    {
        memcpy(data_, data.data(), sizeof(T)*len);
    }

    CudaArray& operator=(const T& val)
    {
        std::fill(begin(), end(), val);
        return *this;
    }

    CUDA_ENABLED constexpr size_t size() const noexcept
    { return len; }

    CUDA_ENABLED T& operator[](unsigned idx) { return data_[idx]; }
    CUDA_ENABLED const T& operator[](unsigned idx) const { return data_[idx]; }
    CUDA_ENABLED T* data() { return data_; }
    CUDA_ENABLED const T* data() const { return data_; }

    CUDA_ENABLED T* begin()  { return data_; }
    CUDA_ENABLED T* end()  { return data_ + len; }

    CUDA_ENABLED const T* begin() const  { return data_; }
    CUDA_ENABLED const T* end() const  { return data_ + len; }

    CUDA_ENABLED const T* cbegin() const  { return data_; }
    CUDA_ENABLED const T* cend() const  { return data_ + len; }

private:
    T data_[len];
};

}}}

#endif // PACBIO_CUDA_CUDA_ARRAY_H_
