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

#ifndef PACBIO_CUDA_KERNEL_MANAGER_H_
#define PACBIO_CUDA_KERNEL_MANAGER_H_

#include <tuple>
#include <utility>

namespace PacBio {
namespace Cuda {

struct LaunchInfo;

template <typename FT, typename... LaunchParams>
class Launcher
{
public:
    Launcher(FT f, LaunchParams... params) : params_{f, params...} {}

    template <typename... Args>
    void operator()(Args&&... args) const;

private:
    template <size_t... Is, typename... Args>
    void Invoke(std::index_sequence<Is...>, Args&&... args) const;

    std::tuple<FT, LaunchParams...> params_;
};

struct LaunchInfo {
private:
    template <typename FT, typename... LaunchParams>
    friend class Launcher;

    LaunchInfo() = default;
};

template <typename FT, typename... Params>
auto PBLaunch(FT f, Params&&... params)
{
    return Launcher<FT, Params...>(f, std::forward<Params>(params)...);
}

}}

#endif // ::PACBIO_CUDA_KERNEL_MANAGER_H_
