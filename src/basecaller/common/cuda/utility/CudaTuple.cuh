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
