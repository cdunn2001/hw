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

#ifndef PACBIO_COMMON_UTILITY_STRONG_TYPEDEF_H
#define PACBIO_COMMON_UTILITY_STRONG_TYPEDEF_H

#include <common/cuda/CudaFunctionDecorators.h>

// Based loosely off BOOST_STRONG_TYPEDEF, but re-implemented as that
// doesn't support constexpr or cuda
#define PB_STRONG_TYPEDEF(Type, Name)                                                              \
struct Name                                                                                        \
{                                                                                                  \
    using UnderlyingType = Type;                                                                   \
    CUDA_ENABLED explicit constexpr Name(const UnderlyingType t_) : t(t_) {};                      \
    CUDA_ENABLED constexpr Name & operator=(const UnderlyingType & rhs) { t = rhs; return *this;}  \
    CUDA_ENABLED constexpr operator const UnderlyingType & () const {return t; }                   \
    CUDA_ENABLED constexpr operator UnderlyingType & () { return t; }                              \
private:                                                                                           \
    UnderlyingType t;                                                                              \
};

#endif //PACBIO_COMMON_UTILITY_STRONG_TYPEDEF_H
