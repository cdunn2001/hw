// Copyright (c) 2020,2021 Pacific Biosciences of California, Inc.
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

#ifndef PACBIO_GRAPH_GRAPH_MANAGER_H
#define PACBIO_GRAPH_GRAPH_MANAGER_H

#include <memory>
#include <map>
#include <vector>

#include <tbb/flow_graph.h>
#include <tbb/task_scheduler_init.h>


#include <pacbio/PBException.h>
#include <pacbio/logging/Logger.h>

#include <common/graphs/GraphsFwd.h>
#include <common/graphs/GraphNodeBody.h>

namespace PacBio {
namespace Graphs {

namespace detail {

// Extracts the first template parameter out of a sequence
template <typename T>
struct FirstArg;
template <template <typename T1, typename...Ts> class T, typename T1, typename...Ts>
struct FirstArg<T<T1, Ts...>>
{
    using type = T1;
};

// Extracts the second template parameter out of a sequence
template <typename T>
struct SecondArg;
template <template <typename T1, typename T2, typename...Ts> class T, typename T1, typename T2, typename...Ts>
struct SecondArg<T<T1, T2, Ts...>>
{
    using type = T2;
};

// This function is necessary for certain tbb versions when creating an edge with
// a node that has multiple ports.  In newer versions you can make an edge with
// such a node and it will automatically use port 0, which is what we want.  Remove
// these if we upgrade
template <typename T>
T& GetPort0(T& t)
{
    return t;
}
template <typename T1, typename T2>
decltype(auto) GetPort0(tbb::flow::multifunction_node<T1, T2>& node)
{
    return tbb::flow::output_port<0>(node);
}

} // detail

/// This class manages a collection of nodes that form a graph.  Calling AddNode()
/// with an IGraphNodeBody child will create a new graph node with no parents (i.e. a graph input)
/// The function will return a pointer to a graph node, upon which subsequent calls to
/// AddNode() can be made, forming a dependance chain.
///
/// All node pointers returned by an AddNode() call are non owning.  They should be
/// kept around either only temporarily, if just used to form an execution dependancy
/// chain, or longer term if they are meant to serve as an actual input into the graph.
///
/// Any number of data can be inputed into the graph, by any number of threads, and
/// not all input need to use the same entry point.
///
/// The Synchronize() function is provided to block the calling thread until all active computations
/// have finished.  Beware that when using in a multithreading context, nothing will prevent
/// other threads from adding more data, and causing the Synchronize() thread to stall longer.
/// Note: This is separate from the GraphNode::FlushNode function, which is used to recursively
///       traverse a subtree of the graph and handles any final processing that needs to happen
///       once the main compute loop has terminated
///
/// SynchronizeAndReport() does the same, but also produces a performance report, with statistics
/// gathered since the last time SynchronizeAndReport() was called.
///
/// Destruction of this object will block until any active computations have completed
/// TODO: could potentially add an abort mechanism.
template <typename PerfEnum>
class GraphManager
{
public:
    // 0 defaults to basically hardware threads
    GraphManager(size_t threads = 0)
        : init_(threads == 0 ? tbb::task_scheduler_init::automatic : static_cast<int>(threads))
    {}

    // Do not enable the move constructors.  Graph nodes will retain
    // references to this object, and those references would be invalidated
    // by a move
    GraphManager(const GraphManager&) = delete;
    GraphManager(GraphManager&&) = delete;
    GraphManager& operator=(const GraphManager&) = delete;
    GraphManager& operator=(GraphManager&&) = delete;

    /// Adds a new node to this graph using body as an implementation.
    ///
    /// The return type will be a pointer to an actual GraphNode.
    /// That GraphNode will be the node that now owns the T handed in here and now
    /// The type of the GraphNode will correspond to the type of the "body" handed in
    /// (e.g. a TransformBody<In, Out> will result in a pointer to a TransformNode<In, Out>
    ///
    /// This is a non-owning pointer.  There are two reasons to keep it around:
    /// * The graph node types also have an AddNode function you can call,
    ///    when building up dependency chains in the graph
    /// * The graph nodes have a ProcessInput function you can call to drop
    ///   data into the graph for processing
    /// * If you don't want to do either of these things (e.g a leaf node),
    ///   or you are finished adding dependancies, you can just discard this pointer.
    ///
    /// \param body a type of "graph node body" (guaranteed by a static assert in
    ///             GraphManager::AddNode) In particular T will be a child of:
    ///             * TransformBody<In, Out>
    ///             * MultiTransformBody<In, Out>
    ///             * LeafBody<In>
    /// \param stage A PerfEnum label to be used when recording performance metrics
    ///
    /// \return a non-owning pointer to a graph node, as described above
    template <typename T>
    auto * AddNode(std::unique_ptr<T> body, PerfEnum stage)
    {
        static_assert(std::is_base_of<IGraphNodeBody, T>::value,
                      "Cannot generate a graph node from this type.  Must be an IGraphNodeBody child");

        T* dummy = nullptr;  //used for type deduction
        // Unique pointer to a type of graph node
        auto node = GenerateNodeImpl(std::move(body), dummy, stage);
        auto * ret = node.get();
        graphNodes_.emplace_back(std::move(node));
        return ret;
    }

    /// Wait for all outstanding tasks to finish
    void Synchronize()
    {
        g.wait_for_all();
    }

    struct Report
    {
        bool realtime;
        float dutyCycle;
        float totalTime;
        float avgOccupancy;
        float avgDuration;
        float idlePercent;
        PerfEnum stage;
    };
    /// Wait for all tasks to finish, and generate a performance report
    std::vector<Report> SynchronizeAndReport(double expectedDurationMS)
    {
        Synchronize();

        std::vector<Report> reports;
        reports.resize(graphNodes_.size());

        for (size_t i = 0; i < graphNodes_.size(); ++i)
        {
            auto& report = reports[i];
            auto& node = graphNodes_[i];

            const auto& timings = node->Timings();
            report.stage = node->Stage();
            report.dutyCycle = static_cast<float>((timings.partTime + timings.fullTime) / expectedDurationMS);
            report.totalTime = static_cast<float>(timings.idleTime + timings.partTime + timings.fullTime);
            report.avgOccupancy = static_cast<float>(timings.accumulatedOccupancy / report.totalTime);
            report.idlePercent = static_cast<float>(timings.idleTime / report.totalTime * 100.0f);
            report.avgDuration = static_cast<float>(timings.accumulatedDuration / timings.count);

            report.realtime = report.dutyCycle <= node->MaxDutyCycle();
            if (!report.realtime)
                errorCounts_[report.stage]++;
        }

        return reports;
    }

    ~GraphManager()
    {
        try
        {
            Synchronize();
        }
        catch(std::exception& e)
        {
            PBLOG_ERROR << "Swallowing (additional?) exceptions while deconstructing execution graph";
            PBLOG_ERROR << e.what();
        }

        for (auto& kv : errorCounts_)
        {
            // TODO tune this?
            static constexpr size_t errorThreshold = 5;
            // TODO something better than a log message may be appropriate, even if we don't otherwise crash from lack
            // of realtime
            if (kv.second > errorThreshold)
                PBLOG_ERROR << kv.first.toString() << " has " << kv.second << " duty cycle violations";
            else if (kv.second > 0)
                PBLOG_WARN << kv.first.toString() << " has " << kv.second << " duty cycle violations";
        }
    }

private:

    template <typename In, typename Out, typename>
    friend class TransformNode;
    template <typename In, typename Out, typename>
    friend class MultiTransformNode;
    template <typename In, typename>
    friend class LeafNode;

    template <typename T1, typename T2>
    void MakeEdge(T1& sender, T2& receiver)
    {
        static_assert(std::is_same<typename detail::SecondArg<T1>::type, typename detail::FirstArg<T2>::type>::value,
                      "Incompatible types for graph edge");
        if (sender.graph_ != receiver.graph_) throw PBException("Linking nodes not in the same graph");
        tbb::flow::make_edge(GetPort0(sender.node), receiver.node);
    }


    template <typename T, typename In, typename Out>
    auto GenerateNodeImpl(std::unique_ptr<T> ptr, TransformBody<In, Out>*, PerfEnum stage)
    {
        return TransformNode<In, Out, PerfEnum>::MakeUnique(this, std::move(ptr), stage);
    }
    template <typename T, typename In, typename Out>
    auto GenerateNodeImpl(std::unique_ptr<T> ptr, MultiTransformBody<In, Out>*, PerfEnum stage)
    {
        return MultiTransformNode<In, Out, PerfEnum>::MakeUnique(this, std::move(ptr), stage);
    }
    template <typename T, typename In>
    auto GenerateNodeImpl(std::unique_ptr<T> ptr, LeafBody<In>*, PerfEnum stage)
    {
        return LeafNode<In, PerfEnum>::MakeUnique(this, std::move(ptr), stage);
    }

    // Needs to be the first member.  Instantiating any other tbb object will
    // implicitly create a scheduler that locks us in on the wrong number of
    // threads.
    // Note: This of course does not guard against someone externally creating
    //       a separate scheduler first.  So far I've found no robust way to
    //       guard against that.
    tbb::task_scheduler_init init_;
    tbb::flow::graph g;
    std::vector<std::unique_ptr<INode<PerfEnum>>> graphNodes_;
    std::map<PerfEnum, uint32_t> errorCounts_;
};


}}


#endif //PACBIO_GRAPH_GRAPH_MANAGER_H
