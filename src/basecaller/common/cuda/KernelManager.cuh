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

#ifndef PACBIO_CUDA_KERNEL_MANAGER_CUH_
#define PACBIO_CUDA_KERNEL_MANAGER_CUH_

#include "KernelManager.h"

namespace PacBio {
namespace Cuda {

// TODO change namespace?
// Conversion function.  This is a default pass-through implementation
// For any types that need to convert from a host-based storage class to
// a gpu usable view class, provide an overload that is a more specific
// match
template <typename T>
T&& KernelArgConvert(T&& t, const LaunchInfo& info) { return std::forward<T>(t); }

template <typename FT, typename... LaunchParams>
template <typename... Args>
void Launcher<FT, LaunchParams...>::operator()(Args&&... args) const
{
    LaunchInfo info;
    Invoke(std::make_index_sequence<sizeof...(LaunchParams)>{}, KernelArgConvert(std::forward<Args>(args), info)...);
}

template <typename FT, typename... LaunchParams>
template <size_t... Is, typename... Args>
void Launcher<FT, LaunchParams...>::Invoke(std::index_sequence<Is...>, Args&&... args) const
{
   std::get<0>(params_)<<<std::get<Is+1>(params_)...>>>(std::forward<Args>(args)...);
}

}}

#endif // ::PACBIO_CUDA_KERNEL_MANAGER_CUH_
