// Copyright (c) 2019-2021, Pacific Biosciences of California, Inc.
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

#include <cstddef>
#include <limits>
#include <utility>

#include <common/cuda/CudaFunctionDecorators.h>

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

    template <typename Arg>
    CUDA_ENABLED TupleLeaf(Arg&& arg) : data(std::forward<Arg>(arg)) {}
    T data;
};

// Tuple implementation class.  The indexes are an implementation detail and go
// from 0 to N-1, where there are N tuple elements.  Consumer code should use
// the CudaTuple alias below, which handles the integer sequence for you.
//
template <typename Ids, typename... Ts>
class TupleImpl;
template <size_t... Ids, typename... Ts>
class TupleImpl<std::index_sequence<Ids...>, Ts...> : TupleLeaf<Ids, Ts>...
{
    // Helper functions to extract a tuple member.  For all of these functions
    // if you manually specify the first template argument and hand in a
    // CudaTuple, then the compiler can deduce the second type and cast
    // the CudaTuple to the appropriate TupleLeaf parent.
    template <size_t I, typename T>
    CUDA_ENABLED static auto& ExtractById(TupleLeaf<I, T>& l)
    {
        return l.data;
    }
    template <size_t I, typename T>
    CUDA_ENABLED static const auto& ExtractById(const TupleLeaf<I, T>& l)
    {
        return l.data;
    }

    template <typename T, size_t I>
    CUDA_ENABLED static auto& ExtractByType(TupleLeaf<I, T>& l)
    {
        return l.data;
    }
    template <typename T, size_t I>
    CUDA_ENABLED static const auto& ExtractByType(const TupleLeaf<I, T>& l)
    {
        return l.data;
    }


 public:

    /// Default construct all elements
    TupleImpl() = default;

    /// Directly construt each tuple member from the associated argument.
    /// The forwarding templates are mostly there so we can just move
    /// arguments if appropriate, but this also enables converting constructors
    template <typename... Us, std::enable_if_t<sizeof...(Us) == sizeof...(Ts), int> = 0>
    CUDA_ENABLED TupleImpl(Us&&... us)
        : TupleLeaf<Ids, Ts>(std::forward<Us>(us))...
    {}

    // Construct all elements with the same argument.  Intentionally not using
    // std::forward since moves are generally destructive and moving the same
    // argument to multiple constructors would likely cause problems.
    // Note: This function only exists in support of old prototype code.  If it
    // gets in the way we can evaluate killing that off
    template <typename Arg>
    CUDA_ENABLED TupleImpl(const Arg& arg) : TupleLeaf<Ids, Ts>(arg)... {}

    /// Get a reference to a given element specified by index.
    template <size_t idx>
    CUDA_ENABLED auto& Get()
    {
        static_assert(idx <= sizeof...(Ids),
                      "Index larger than tuple size");
        return ExtractById<idx>(*this);
    }

    /// Get a const reference to a given element specified by index.
    template <size_t idx>
    CUDA_ENABLED const auto& Get() const
    {
        static_assert(idx <= sizeof...(Ids),
                      "Index larger than tuple size");
        return ExtractById<idx>(*this);
    }

    /// Get a reference to a given element specified by type.
    /// There must be exactly one occurance of T in the tuple members
    template <typename T>
    CUDA_ENABLED auto& Get()
    {
        return ExtractByType<T>(*this);
    }

    /// Get a const reference to a given element specified by type.
    /// There must be exactly one occurance of T in the tuple members
    template <typename T>
    CUDA_ENABLED const auto& Get() const
    {
        return ExtractByType<T>(*this);
    }

    /// Directly call a given element, assuming it is callable.
    template <size_t idx, typename... Args>
    CUDA_ENABLED auto Invoke(Args&&... args)
    {
        auto& member = Get<idx>();
        return member(std::forward<Args>(args)...);
    }
};

}

template <typename... Ts>
using CudaTuple = detail::TupleImpl<std::make_index_sequence<sizeof...(Ts)>, Ts...>;

}}}

#endif //PACBIO_CUDA_CUDA_TUPLE_H_
