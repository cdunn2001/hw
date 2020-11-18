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

#ifndef PACBIO_CUDA_LAUNCH_MANAGER_CUH_
#define PACBIO_CUDA_LAUNCH_MANAGER_CUH_

#include <tuple>

#include <common/cuda/streams/KernelLaunchInfo.h>

namespace PacBio {
namespace Cuda {

// Conversion function.  This is a default pass-through implementation
// For any types that need to convert from a host-based storage class to
// a gpu usable view class, provide an overload that is a more specific
// match
template <typename T>
T&& KernelArgConvert(T&& t, const KernelLaunchInfo&) { return std::forward<T>(t); }

// Need to disable handing raw pointers to the device, as that's almost certainly
// not what you wanted to do.  Any pointers to gpu memory really need to be
// wrapped in one of the smart container types under Cuda::Memory
template <typename T>
T* KernelArgConvert(T* t, const KernelLaunchInfo&)
{
    static_assert(std::is_pointer<T*>::value, "Cannot launch kernels with pointer arguments");
}
template <typename T>
const T* KernelArgConvert(const T* t, const KernelLaunchInfo&)
{
    static_assert(std::is_pointer<const T*>::value, "Cannot launch kernels with pointer arguments");
}


// Class for managing kernel invokations.  This is implemeted as a class so that we can keep the
// arguments for the kernel invocation itself (number of threads, grid layout, etc) separate
// from the actual kernel args.  Both need to be variadic templates, so some level
// of separation is necessary.
template <typename FT, typename... LaunchParams>
class LaunchManager
{
    // Need to ensure some constraints on LaunchParams.  The most common invocation will be
    // with two arguments, specifying the layout of threads in a block, and blocks in a grid.
    // They need to be templated because for each we can hand in either raw integers, or a dim3 or
    // an initializer list.  The third (optional) argument is how much shared memory space the
    // kernel needs to reserve if that is being done dynamically.  The 4th (optional) argument
    // is for specifying which stream to execute on, but in our case specifying this is
    // not allowed, as we maintain a 1-1 correlation between cpu threads and gpu streams
    static_assert(sizeof...(LaunchParams) > 1, "Must specify kernel threading parameters");
    static_assert(sizeof...(LaunchParams) < 4,
                  "Cannot specify stream, usage of per-thread default stream is enforced");
public:
    LaunchManager(FT f, LaunchParams... params) : params_{f, params...} {}

    template <typename... Args>
    void operator()(Args&&... args) const
    {
        auto event = std::make_shared<CudaEvent>();
        KernelLaunchInfo info(event);
        Invoke(std::make_index_sequence<sizeof...(LaunchParams)>{}, KernelArgConvert(std::forward<Args>(args), info)...);
        ThrowIfCudaError();
        event->RecordEvent();
    }

private:
    // Helper function, just exists to get the Is... template pack
    // so we can extract from the tuple
    template <size_t... Is, typename... Args>
    void Invoke(std::index_sequence<Is...>, Args&&... args) const
    {
        std::get<0>(params_)<<<std::get<Is+1>(params_)...>>>(std::forward<Args>(args)...);
    }

    std::tuple<FT, LaunchParams...> params_;
};

// This is the main intended entry point.  Wouldn't be necessary
// in c++17 where you can have class argument deduction guides, but
// for now we'll have to make do with a free function
template <typename FT, typename... Params>
auto PBLauncher(FT f, Params&&... params)
{
    return LaunchManager<FT, Params...>(f, std::forward<Params>(params)...);
}

}}

#endif // ::PACBIO_CUDA_LAUNCH_MANAGER_CUH_
