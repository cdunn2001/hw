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
//  Graph Nodes cannot be directly constructed.  You must either have a handle to
//  a graph and ask it to create a node for you (in which case this is now a purely)
//  input node, it will have no parents), or you have a handle to an existing node
//  and create a new node that will be it's child.  In general you just have to call
//  the AddNode() function of a graph/node and supply it with a suitable IBody child
//  class.  When that is done the graph will be updated with the new node and necessary
//  edges, and a non-owning graph node pointer will be returned so you can potentially
//  add even more children
//
//  Note: For now, not all graphs are representable.  In reality it really only
//        represents a tree, you cannot have multiple nodes as input to the same
//        node.  This could potentially be extended in the future if there is need.
//
//  Note: For now, the API is heavily reminiscent of tbb::flow_graph, which is used
//        for the implementation.  That's not an intentional part of the design,
//        and this API should be able to diverge significantly away from that of tbb,
//        and even stop using tbb as an implementation.
//
//  Note: Graphs and their associated nodes are all threaded by a PerfEnum template
//        parameter.  This should be a SmartEnum, with each node being associated with
//        an enum value.  These enums will be used in automatic performance monitoring
//        and reporting

#ifndef PACBIO_GRAPHS_NODE_H
#define PACBIO_GRAPHS_NODE_H

#include <memory>
#include <type_traits>

#include <tbb/flow_graph.h>

#include <common/graphs/NodeMonitor.h>
#include <common/graphs/GraphsFwd.h>
#include <common/graphs/GraphNodeBody.h>

namespace PacBio {
namespace Graphs {

namespace detail {

// The underlying tbb implementation does not handle move-only types.
// Thus WrappedVal is introduced to wrap everything in a shared_ptr
// under the hood.  The implementation of the graph API itself will
// work so that WrappedVal<T> has the same semantics as T (e.g. respect
// move-only types)
template <typename T>
struct WrappedVal
{
    std::shared_ptr<T> val;
};

}


// Base class of all nodes.  Note useful for much other than a generic handle
// and doing performance reporting
template <typename PerfEnum>
class INode
{
public:
    INode(int maxThreads, PerfEnum stage)
        : monitor_(maxThreads)
        , stage_(stage)
    {}
    virtual ~INode() {}

    auto Timings()
    {
        return monitor_.Timings();
    }
    PerfEnum Stage() const { return stage_; }
    virtual float MaxDutyCycle() const = 0;
    virtual size_t ConcurrencyLimit() const = 0;

protected:

    detail::NodeMonitor monitor_;
    PerfEnum stage_;
};

// Intermediate base class that just exposes the input template parameter.
// Also provides the functionality for adding new inputs to be processed
// by the graph
template <typename In, typename PerfEnum>
class InputNode : public INode<PerfEnum>
{
public:
    InputNode(int maxThreads, PerfEnum stage)
        :INode<PerfEnum>(maxThreads, stage)
    {}
    // Inserts new data into the graph
    virtual void ProcessInput(std::decay_t<In> in) = 0;
};

template <typename In, typename Out, typename PerfEnum>
class TransformNode : public InputNode<In, PerfEnum>
{
public:
    // Adds a new child node to this node using body as an implementation.
    // Returns another graph node pointer (with input/output types matching body)
    // that can be used to add yet more children nodes (unless body is a leaf)
    template <typename T>
    auto * AddNode(std::unique_ptr<T> body, PerfEnum stage)
    {
        if (hasChild_ && !std::is_const<Out>::value)
            throw PBException("Cannot have multiple outputs from a graph node if the output type is not const!");

        auto * ret = graph_->AddNode(std::move(body), stage);
        hasChild_ = true;
        graph_->MakeEdge(*this, *ret);
        return ret;
    }

    // Even if this node is set to accept const T, for an original input we want a nonconst version.
    // We need to copy/move it and claim ownership, since we don't know when or what thread will consume it
    void ProcessInput(std::decay_t<In> in) override
    {
        WrappedIn tmp;
        tmp.val = std::make_shared<In>(std::move(in));
        if (!node.try_put(tmp)) throw PBException("Failure to launch graph task");
    }

private:
    friend class GraphManager<PerfEnum>;
    using WrappedIn = detail::WrappedVal<In>;
    using WrappedOut = detail::WrappedVal<Out>;

    TransformNode(GraphManager<PerfEnum>* graph,
                  std::unique_ptr<TransformBody<In, Out>> body,
                  PerfEnum stage)
        : InputNode<In, PerfEnum>(body->ConcurrencyLimit(), stage)
        , graph_(graph)
        , node(graph->g, body->ConcurrencyLimit(), [this](WrappedIn in) ->WrappedOut { return Run(in); })
    {
        body_ = std::move(body);
    }

    static auto MakeUnique(GraphManager<PerfEnum>* graph,
                           std::unique_ptr<TransformBody<In, Out>> body,
                           PerfEnum stage)
    {
        return std::unique_ptr<TransformNode>(new TransformNode(graph, std::move(body), stage));
    }

    float MaxDutyCycle() const override { return body_->MaxDutyCycle(); }
    size_t ConcurrencyLimit() const override { return body_->ConcurrencyLimit(); }

    // Two version of run, depending on if the input type is const or not.
    // If it's a const input then we'll pass it in as a const reference.  If it's
    // not, we'll do a move/copy to give the body ownership
    template <typename T = In, std::enable_if_t<std::is_const<T>::value, int> = 0>
    WrappedOut Run(WrappedIn in)
    {
        auto tmp = this->monitor_.StartScope();
        if (!hasChild_)
            throw PBException("Output of TransformNode is ignored, graph is incomplete");
        WrappedOut out;
        out.val = std::make_shared<Out>(body_->Process(*in.val));
        return out;
    }

    template <typename T = In, std::enable_if_t<!std::is_const<T>::value, int> = 0>
    WrappedOut Run(WrappedIn in)
    {
        auto tmp = this->monitor_.StartScope();
        if (!hasChild_)
            throw PBException("Output of TransformNode is ignored, graph is incomplete");
        WrappedOut out;
        out.val = std::make_shared<Out>(body_->Process(std::move(*in.val)));
        return out;
    }

    GraphManager<PerfEnum>* graph_;
    std::unique_ptr<TransformBody<In, Out>> body_;
    tbb::flow::function_node<WrappedIn, WrappedOut> node;
    bool hasChild_ = false;
};

template <typename In, typename Out, typename PerfEnum>
class MultiTransformNode : public InputNode<In, PerfEnum>
{
public:
    template <typename T>
    auto * AddNode(std::unique_ptr<T> body, PerfEnum stage)
    {
        if (hasChild_ && !std::is_const<Out>::value)
            throw PBException("Cannot have multiple outputs from a graph node if the output type is not const!");

        auto * ret = graph_->AddNode(std::move(body), stage);
        hasChild_ = true;
        graph_->MakeEdge(*this, *ret);
        return ret;
    }

    // Even if this node is set to accept const T, for an original input we want a nonconst version.
    // We need to copy/move it and claim ownership, since we don't know when or what thread will consume it
    void ProcessInput(std::decay_t<In> in) override
    {
        WrappedIn tmp;
        tmp.val = std::make_shared<In>(std::move(in));
        if (!node.try_put(tmp)) throw PBException("Failure to launch graph task");
    }

private:
    friend class GraphManager<PerfEnum>;
    using WrappedIn = detail::WrappedVal<In>;
    using WrappedOut = detail::WrappedVal<Out>;

    MultiTransformNode(GraphManager<PerfEnum>* graph,
                       std::unique_ptr<MultiTransformBody<In, Out>> body,
                       PerfEnum stage)
        : InputNode<In, PerfEnum>(body->ConcurrencyLimit(), stage)
        , graph_(graph)
        , node(graph->g, body->ConcurrencyLimit(), [this](WrappedIn in, auto& out) { this->Run(in, out); })
    {
        body_ = std::move(body);
    }

    static auto MakeUnique(GraphManager<PerfEnum>* graph,
                           std::unique_ptr<MultiTransformBody<In, Out>> body,
                           PerfEnum stage)
    {
        return std::unique_ptr<MultiTransformNode>(new MultiTransformNode(graph, std::move(body), stage));
    }

    float MaxDutyCycle() const override { return body_->MaxDutyCycle(); }
    size_t ConcurrencyLimit() const override { return body_->ConcurrencyLimit(); }

    // Two version of run, depending on if the input type is const or not.
    // If it's a const input then we'll pass it in as a const reference.  If it's
    // not, we'll do a move/copy to give the body ownership
    template <typename Output, typename T = In, std::enable_if_t<std::is_const<T>::value, int> = 0>
    void Run(WrappedIn in, Output& output)
    {
        auto tmp = this->monitor_.StartScope();
        if (!hasChild_)
            throw PBException("Output of TransformNode is ignored, graph is incomplete");

        body_->Process(*in.val);
        auto& deque = body_->GetDeque();
        while (!deque.empty())
        {
            WrappedOut out;
            out.val = std::make_shared<Out>(std::move(deque.back()));
            deque.pop_back();
            std::get<0>(output).try_put(out);
        }
    }

    template <typename Output, typename T = In, std::enable_if_t<!std::is_const<T>::value, int> = 0>
    void Run(WrappedIn in, Output& output)
    {
        auto tmp = this->monitor_.StartScope();
        if (!hasChild_)
            throw PBException("Output of TransformNode is ignored, graph is incomplete");

        body_->Process(std::move(*in.val));
        auto& deque = body_->GetDeque();
        while (!deque.empty())
        {
            WrappedOut out;
            out.val = std::make_shared<Out>(std::move(deque.back()));
            deque.pop_back();
            std::get<0>(output).try_put(out);
        }
    }

    GraphManager<PerfEnum>* graph_;
    std::unique_ptr<MultiTransformBody<In, Out>> body_;
    tbb::flow::multifunction_node<WrappedIn, std::tuple<WrappedOut>> node;
    bool hasChild_ = false;
};

template <typename In, typename PerfEnum>
class LeafNode : public InputNode<In, PerfEnum>
{
public:
    void ProcessInput(std::decay_t<In> in) override
    {
        WrappedIn tmp;
        tmp.val = std::make_shared<In>(std::move(in));
        if (!node.try_put(tmp)) throw PBException("Failure to launch graph task");
    }

private:
    friend class GraphManager<PerfEnum>;
    using WrappedIn = detail::WrappedVal<In>;

    LeafNode(GraphManager<PerfEnum>* graph,
             std::unique_ptr<LeafBody<In>> body,
             PerfEnum stage)
        : InputNode<In, PerfEnum>(body->ConcurrencyLimit(), stage)
        , graph_(graph)
        , node(graph->g, body->ConcurrencyLimit(), [this](WrappedIn in) { Run(std::move(in)); })
    {
        body_ = std::move(body);
    }

    static auto MakeUnique(GraphManager<PerfEnum>* graph,
                           std::unique_ptr<LeafBody<In>> body,
                           PerfEnum stage)
    {
        return std::unique_ptr<LeafNode>(new LeafNode(graph, std::move(body), stage));
    }

    float MaxDutyCycle() const override { return body_->MaxDutyCycle(); }
    size_t ConcurrencyLimit() const override { return body_->ConcurrencyLimit(); }

    template <typename T = In, std::enable_if_t<std::is_const<T>::value, int> = 0>
    void Run(WrappedIn in)
    {
        auto tmp = this->monitor_.StartScope();
        body_->Process(*in.val);
    }

    template <typename T = In, std::enable_if_t<!std::is_const<T>::value, int> = 0>
    void Run(WrappedIn in)
    {
        auto tmp = this->monitor_.StartScope();
        body_->Process(std::move(*in.val));
    }

    GraphManager<PerfEnum>* graph_;
    std::unique_ptr<LeafBody<In>> body_;
    tbb::flow::function_node<WrappedIn, tbb::flow::continue_msg> node;
};


}}

#endif //PACBIO_GRAPHS_NODE_H
