// Copyright (c) 2020, Pacific Biosciences of California, Inc.
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
//
// This file defines the API for GraphNodeBodies.  In general, these classes
// will serve as functors that get invoked as data is passed to a particular
// node in a graph.  There are currently three types of nodes:
// 1. LeafNodes -- Accepts a single input and has no output
// 2. TransformNodes -- Accepts a single input, and produces a single output.
//                      That output can be sent to multiple downstream nodes
// 3. MultiTransformNodes -- Accepts a single input, and produces any number of
//                           outputs.  Each output generated can go to multiple
//                           downstream nodes.
//
//  Input/Output types must be value types (no pointers, references).  If
//  it is a const value output, then all downstream nodes will receive the
//  data as a const reference input.  If it is a non-const value, then the
//  data will be moved (or coppied if no move available) to downstream nodes.
//  If a node has multiple children, then the output type *must* be const.
//  (TODO: This could potentially be reworked, so that non-const output
//   with multiple children explicitly uses copy and no move, but that's extra
//   metaprogramming work)
//
//  Note: For now, not all graphs are representable.  In reality it really only
//        represents a tree, you cannot have multiple nodes as input to the same
//        node.  This could potentially be extended in the future if there is need.
//
//  Note: For now, the API is heavily reminiscent of tbb::flow_graph, which is used
//        for the implementation.  That's not an intentional part of the design,
//        and this API should be able to diverge significantly away from that of tbb,
//        and even stop using tbb as an implementation.

#ifndef PACBIO_GRAPHS_NODE_BODY_H
#define PACBIO_GRAPHS_NODE_BODY_H

#include <cstddef>
#include <deque>
#include <mutex>
#include <type_traits>

namespace PacBio {
namespace Graphs {

namespace detail {

// Constexper function to make sure graph node bodies are instantiated with valid input/output
// types.  In short, only value types are valid.  Things like pointers and references won't
// add any meaningfully beneficial semantics to the graph API, and only complicate the
// template metaprogramming
template <typename T>
static constexpr bool valid_type()
{
    static_assert(!std::is_reference<T>::value, "Reference detected, use value types for graph node input/output");
    static_assert(!std::is_pointer<T>::value, "Pointer detected, use value types for graph node input/output");
    static_assert(!std::is_volatile<T>::value, "Why are you using volatile?  Don't use volatile!");
    static_assert(!std::is_void<T>::value, "void is invalid for graph input/output.  Did you mean to use a LeafBody?");

    return true;
}

template <typename T>
struct ref_if_const { using type = T; };
template <typename T>
struct ref_if_const<const T> { using type = const T&; };

template <typename T>
using ref_if_const_t = typename ref_if_const<T>::type;

}

// Base class for all node bodies.  This class is not really meant to be used directly,
// instead inherit from one of the children below
struct IGraphNodeBody
{
    virtual ~IGraphNodeBody() {};

    virtual size_t ConcurrencyLimit() const = 0;
    virtual float MaxDutyCycle() const = 0;
};

// Single input to single output node
template <typename In, typename Out>
struct TransformBody : public IGraphNodeBody
{
    static_assert(detail::valid_type<In>(), "Invalid input");
    static_assert(detail::valid_type<Out>(), "Invalid output");

    // If the graph link is a const type, that is how it will be stored
    // when passing through the graph infrastructure.  However here at the
    // low level point of creation/consumption, it makes the most sense for
    // the output to be a non-const value type, and the input to be a const
    // reference
    virtual std::remove_const_t<Out> Process(detail::ref_if_const_t<In> in) = 0;
};

// Single input and no output node
template <typename In>
struct LeafBody : public IGraphNodeBody
{
    static_assert(detail::valid_type<In>(), "Invalid input");

    // If the graph link is a const type, that is how it will be stored
    // when passing through the graph infrastructure.  However here at the
    // low level point of creation/consumption, it makes the most sense for
    // the output to be a non-const value type, and the input to be a const
    // reference
    virtual void Process(detail::ref_if_const_t<In> in) = 0;
};

// Transformation node capable of generating any number (including 0)
// outputs each invocation
template <typename In, typename Out>
struct MultiTransformBody : public LeafBody<In>
{
    static_assert(detail::valid_type<In>(), "Invalid input");
    static_assert(detail::valid_type<Out>(), "Invalid output");

    // TODO think access controls
public:
    // TODO parent should define this?
    using InternalOut = std::remove_const_t<Out>;

    // Thread-safe routine to push a completed piece of work to
    // the output queue (which is potentially shared among threads)
    void PushOut(InternalOut out)
    {
        std::lock_guard<std::mutex> lm(outputMutex_);
        output.emplace_back(std::move(out));
    }

    // Thread-safe routine to consume all completed work
    // waiting in the output queue.  Must supply a functor
    // `f` which will accept output values as an argument
    template <typename Func>
    void ConsumeAllOutput(Func&& f)
    {
        std::lock_guard<std::mutex> lm(outputMutex_);
        while(!output.empty())
        {
            f(std::move(output.back()));
            output.pop_back();
        }
    }
private:
    std::mutex outputMutex_;
    std::deque<InternalOut> output;
};

}}

#endif // PACBIO_GRAPHS_NODE_BODY_H
