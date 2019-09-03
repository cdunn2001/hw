#ifndef PACBIO_CUDA_DATA_ZMW_DATA_POPULATOR_H
#define PACBIO_CUDA_DATA_ZMW_DATA_POPULATOR_H

#include <atomic>
#include <cstdlib>
#include <thread>
#include <memory>
#include <mutex>
#include <vector>

#include <common/DataGenerators/GeneratorBase.h>

#include <dataTypes/TraceBatch.h>
#include <pacbio/dev/AutoTimer.h>

namespace PacBio {
namespace Cuda {
namespace Data {

// All the arguments needed to construct a ZmwDataManager object.
// Helps to be more safe and  expressive than just calling a ctor
// with a ton of arguments
struct DataManagerParams
{
    size_t numZmwLanes = 125000;
    size_t numBlocks = 10;
    size_t laneWidth = 64;
    size_t blockLength = 64;
    size_t kernelLanes = 5000;
    bool immediateCopy = false;
    float frameRate = 100;

    // Builder pattern functions
    DataManagerParams& NumZmwLanes(size_t val) { numZmwLanes   = val; return *this; }
    DataManagerParams& NumBlocks(size_t val)   { numBlocks     = val; return *this; }
    DataManagerParams& LaneWidth(size_t val)   { laneWidth     = val; return *this; }
    DataManagerParams& BlockLength(size_t val) { blockLength   = val; return *this; }
    DataManagerParams& KernelLanes(size_t val) { kernelLanes   = val; return *this; }
    DataManagerParams& ImmediateCopy(bool val) { immediateCopy = val; return *this; }
    DataManagerParams& FrameRate(float val)    { frameRate     = val; return *this; }
};

// Class used to simulate an IO stream with a given input rate.
// Filler template parameter is used to substitute in different
// data generation routines.
template <typename TIn, typename TOut = TIn>
class ZmwDataManager
{
public:

    ZmwDataManager(const DataManagerParams& params, std::unique_ptr<GeneratorBase<TIn>> source, bool silent = false);

    ZmwDataManager(const ZmwDataManager&) = delete;
    ZmwDataManager(ZmwDataManager&&) = delete;
    ZmwDataManager& operator=(const ZmwDataManager&) = delete;
    ZmwDataManager& operator=(ZmwDataManager&&) = delete;

    // Check if there is (or will be) more data to send to the GPU
    bool MoreData() const
    {
        return running_ || batchesReady_ > 0;
    }

    // Helper class to "loan" data to external code for processing.
    // The API is still a little loose, but essentially the
    // compute buffer is not "complete" until each batch has been
    // loaned and then returned.  Do *not* steal and keep the data
    // reference contained herein.
    class LoanedData
    {
        friend class ZmwDataManager;
        LoanedData(Mongo::Data::TraceBatch<TIn>&& input,
                   Mongo::Data::TraceBatch<TOut>&& output)
            : input_(std::move(input))
            , output_(std::move(output))
        {}
    public:
        LoanedData(const LoanedData&) = delete;
        LoanedData(LoanedData&&) = default;
        LoanedData& operator=(const LoanedData&) = delete;
        LoanedData& operator=(LoanedData&&) = default;

        ~LoanedData() = default;

        size_t Batch() const { return input_.Metadata().PoolId(); }
        size_t FirstFrame() const {return input_.Metadata().FirstFrame(); }
        Mongo::Data::TraceBatch<TIn>& KernelInput() { return input_; }
        Mongo::Data::TraceBatch<TOut>& KernelOutput() { return output_; }

    private:
        Mongo::Data::TraceBatch<TIn> input_;
        Mongo::Data::TraceBatch<TOut> output_;
    };

    // TODO re-think exceptions
    // TODO Explanation: In an effort to enforce "loan" semantics, I wanted to make
    //                   this function the only way to obtain a LoanedData, for that
    //                   object to always be valid, and to force an std::move upon
    //                   return to make it clear the consuming code relinquishes
    //                   ownership.  However a flaw in the current approach is that
    //                   even when calling `MoreData` before this, another thread
    //                   could swoop in and steal the last piece of data before this
    //                   thread can.  The only way out of that situation is to:
    //                     - throw an exception
    //                     - Change `LoanedData` to have an empty/invalid state
    //                     - Have this function accept a LoanedData reference/ptr
    //                       to populate.
    //                   The exception is here now because it was the easy way forward
    //                   once I noticed the problem.  I don't really like using an
    //                   exception here, but there are parts of all alternates I don't
    //                   particularly like either, so punting on this for now until it's
    //                   been considered more thoroughly.
    LoanedData NextBatch();

    // Returns data to this class, so that it can be re-filled by the writer thread.
    // Only accepting an rvalue reference is intentional, to make it clear that calling
    // code is relinquishing ownership.
    void ReturnBatch(LoanedData&& data);

    ~ZmwDataManager();

private:
    // Fills a batch of data.  Loops through all the "lanes" and delegates to the Filler object
    // to actually populate the data.
    void FillNext(Mongo::Data::TraceBatch<TIn>& ptr);

    // Function for separate IO thread to use, to continuously populate the data banks.
    void FillerThread();

    DataManagerParams params_;

    std::mutex nextBatchMutex_;
    std::atomic<bool> exitRequested_;
    std::atomic<bool> running_;
    std::atomic<bool> fillingBank_;
    std::atomic<size_t> batchesReady_;
    std::atomic<size_t> batchesLoaned_;
    std::atomic<size_t> computeBlock_;
    std::thread thread_;

    std::vector<Mongo::Data::TraceBatch<TIn>> bank0;
    std::vector<Mongo::Data::TraceBatch<TIn>> bank1;
    std::vector<Mongo::Data::TraceBatch<TOut>> output;
    std::vector<int> checkedOut_;

    size_t numBatches_;
    bool quiet_;

    Dev::QuietAutoTimer computeTimer_;
    size_t laneIdx_ = 0;
    size_t blockIdx_ = 0;

    std::unique_ptr<GeneratorBase<TIn>> source_;
};

}}} // ::PacBio::Cuda::Data

#endif //PACBIO_CUDA_DATA_ZMW_DATA_POPULATOR_H
