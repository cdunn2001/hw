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

#ifndef PACBIO_CUDA_CUDA_TUPLE_H_
#define PACBIO_CUDA_CUDA_TUPLE_H_

// std::tuple does work in device code, but for whatever reason it caused catastrophic
// slowdown in compilation speeds.  I'm sure this version is missing features, but
// for whatever reason it compiles vastly faster

#include <cstddef>
#include <utility>

namespace PacBio {
namespace Cuda {
namespace Utility {

namespace detail {

// Tuple will be implemented via multiple inheritance of TupleLeaf, one
// leaf for each tuple member.  The idx template parameter is to keep the
// parents unique even if the tuple contains multiple of the same type.
template <size_t idx, typename T>
struct TupleLeaf
{
    TupleLeaf() = default;

    template <typename... Args>
    __device__ TupleLeaf(Args&&... args) : data(std::forward<Args>(args)...) {}
    T data;
};

// Recursive template to extract the Nth type in a template parameter pack
template <size_t curr, size_t idx, typename T1, typename... Ts>
struct nth
{
    using next = nth<curr+1, idx, Ts...>;
    using type = typename next::type;
};
template <size_t i, typename T1, typename... Ts>
struct nth<i, i, T1, Ts...>
{
    using type = T1;
};

// Tuple implementation class.  The indexes are an implementation detail and go
// from 0 to N-1, where there are N tuple elements.  Consumer code should use
// the CudaTuple alias below, which handles the integer sequence for you.
//
template <typename Ids, typename... Ts>
struct _TupleImpl;
template <size_t... Ids, typename... Ts>
struct _TupleImpl<std::index_sequence<Ids...>, Ts...> : public TupleLeaf<Ids, Ts>...
{
    // Default construct all elements
    _TupleImpl() = default;

    // Copy construct all elements
    __device__ _TupleImpl(const Ts&... t) : TupleLeaf<Ids, Ts>(t)... {}

    // Construct all elements with the same arguments.  Intentionally not using
    // std::forward since moves are generally destructive and moving the same
    // argument to multiple constructors would likely cause problems.
    template <typename... Args>
    __device__ _TupleImpl(Args&&... args) : TupleLeaf<Ids, Ts>(args...)... {}

    // Get a reference to a given element.
    template <size_t idx>
    __device__ auto& Get()
    {
        return static_cast<TupleLeaf<idx, typename nth<0, idx, Ts...>::type>&>(*this).data;
    }

    // Directly call element, assuming it is callable.
    template <size_t idx, typename... Args>
    __device__ auto Invoke(Args&&... args)
    {
        auto& member = static_cast<TupleLeaf<idx, typename nth<0, idx, Ts...>::type>&>(*this).data;
        return member(std::forward<Args>(args)...);
    }
};

}

template <typename... Ts>
using CudaTuple = detail::_TupleImpl<std::make_index_sequence<sizeof...(Ts)>, Ts...>;

}}}

#endif //PACBIO_CUDA_CUDA_TUPLE_H_
