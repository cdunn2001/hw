#include "FrameLabeler.h"

#include <cuda_fp16.h>
#include <cuda_fp16.hpp>

#include <common/cuda/memory/DeviceOnlyArray.cuh>
#include <common/KernelThreadPool.h>
#include <dataTypes/TraceBatch.cuh>

using namespace PacBio::Cuda::Memory;
using namespace PacBio::Cuda::Data;
using namespace PacBio::Mongo::Data;

namespace PacBio {
namespace Cuda {

//this is annoying... are there really no blend intrinsics??
__device__ half2 Blend(half2 cond, half2 l, half2 r)
{
    half zero = __float2half(0.0f);
    half low =  (__low2half(cond)  == zero) ? __low2half(r)  : __low2half(l);
    half high = (__high2half(cond) == zero) ? __high2half(r) : __high2half(l);
    return __halves2half2(low, high);
}

__device__ half2 HalfOr(half2 first, half2 second)
{
    half zero = __float2half(0.0f);
    half low  = (__low2half(first)  != zero) || (__low2half(second)  != zero);
    half high = (__high2half(first) != zero) || (__high2half(second) != zero);
    return __halves2half2(low, high);
}

__device__ half2 h2pow2(half2 val) { return val*val; }

__device__ half2 h2min(half2 l, half2 r)
{
    auto cond = __hltu2(l, r);
    return Blend(cond, l, r);
}

template <typename T>
class DevicePtr
{
public:
    DevicePtr(T* data, detail::DataManagerKey)
        : data_(data)
    {}

    __device__ T* operator->() { return data_; }
    __device__ const T* operator->() const { return data_; }
private:
    T* data_;
};

template <typename T>
class DeviceOnlyObj : private detail::DataManager
{
public:
    template <typename... Args>
    DeviceOnlyObj(Args&&... args)
        : data_(1, std::forward<Args>(args)...)
    {}

    DevicePtr<T> GetDevicePtr()
    {
        return DevicePtr<T>(data_.GetDeviceView().Data(DataKey()), DataKey());
    }

private:
    Memory::DeviceOnlyArray<T> data_;
};

struct TransitionMatrix
{

};

template <typename T, size_t len>
struct __align__(128) CudaArray
{
    __device__ T& operator[](unsigned idx) { return data_[idx]; }
    __device__ const T& operator[](unsigned idx) const { return data_[idx]; }
private:
    T data_[len];
};

template <size_t laneWidth>
class __align__(128) NormalLog
{
    using Row = CudaArray<half2, laneWidth>;
    static constexpr float log2pi_f = 1.8378770664f;
public:     // Structors
    NormalLog() = default;
    __device__ NormalLog(const half2& mean,
              const half2& variance)
    {
        Mean(mean);
        Variance(variance);
    }

public:     // Properties
    __device__ const half2& Mean() const
    { return mean_[threadIdx.x]; }

    /// Sets Mean to new value.
    /// \returns *this.
    __device__ NormalLog& Mean(const half2& value)
    {
        mean_[threadIdx.x] = value;
        return *this;
    }

    __device__ half2 Variance() const
    { return 0.5f / scaleFactor_[threadIdx.x]; }

    /// Sets Variance to new value, which must be positive.
    /// \returns *this.
    __device__ NormalLog& Variance(const half2& value)
    {
        assert(all(value > 0.0f));
        normTerm_[threadIdx.x] = __float2half2_rn(-0.5f) * (__float2half2_rn(log2pi_f) + h2log(value));
        scaleFactor_[threadIdx.x] = __float2half2_rn(0.5f) / value;
        return *this;
    }

public:
    /// Evaluate the distribution at \a x.
    __device__ half2 operator()(const half2& x) const
    { return normTerm_[threadIdx.x] - h2pow2(x - mean_[threadIdx.x]) * scaleFactor_[threadIdx.x]; }

private:    // Data
    Row mean_;
    Row normTerm_;
    Row scaleFactor_;
};

template <size_t laneWidth>
struct __align__(128) BlockAnalogMode
{
    using Row = CudaArray<half2, laneWidth>;
    Row means;
    Row vars;
};

template <size_t laneWidth>
struct __align__(128) BlockModelParameters
{
    static constexpr unsigned int numAnalogs = 4;

    __device__ const BlockAnalogMode<laneWidth>& BaselineMode() const
    {
        return baseline_;
    }

    __device__ const BlockAnalogMode<laneWidth>& AnalogMode(unsigned i) const
    {
        return analogs_[i];
    }

 private:
    CudaArray<BlockAnalogMode<laneWidth>, numAnalogs> analogs_;
    BlockAnalogMode<laneWidth> baseline_;
};

template <size_t laneWidth>
struct __align__(128) SubframeScorer
{
    static constexpr unsigned int numAnalogs = 4;
    using Row = CudaArray<half2, laneWidth>;

    SubframeScorer() = default;

    __device__ void Setup(const BlockModelParameters<laneWidth>& model)
    {
        static constexpr float pi_f = 3.1415926536f;
        const half2 sqrtHalfPi = __float2half2_rn(std::sqrt(0.5f * pi_f));
        const half2 halfVal = __float2half2_rn(0.5f);
        const half2 nhalfVal = __float2half2_rn(-0.5f);
        const half2 oneSixth = __float2half2_rn(1.0f / 6.0f);

        const auto& bMode = model.BaselineMode();
        const auto bMean = bMode.means[threadIdx.x];
        const auto bVar = bMode.vars[threadIdx.x];
        const auto bSigma = h2sqrt(bVar);

        // Low joint in the score function.
        x0_[threadIdx.x] = bMean + bSigma * sqrtHalfPi;
        bFixedTerm_[threadIdx.x] = __float2half2_rn(-0.5f) / bVar;

        for (unsigned int a = 0; a < numAnalogs; ++a)
        {
            const auto& dma = model.AnalogMode(a);
            const auto aMean = dma.means[threadIdx.x];
            logPMean_[a][threadIdx.x] = h2log(aMean - bMean);
            const auto aVar = dma.vars[threadIdx.x];
            const auto aSigma = h2sqrt(aVar);
            // High joint in the score function.
            x1_[a][threadIdx.x] = aMean - aSigma * sqrtHalfPi;
            pFixedTerm_[a][threadIdx.x] = nhalfVal / aVar;
        }

        // Assume that the dimmest analog is the only one at risk for
        // violating x0_ <= x1_[a].
        {
            const auto& dm = model.AnalogMode(3);
            const auto aMean = dm.means[threadIdx.x];
            const auto aVar = dm.vars[threadIdx.x];
            // Edge-frame mean and variance.
            auto eMean = halfVal * (bMean + aMean);
            auto eVar = oneSixth * (aMean - bMean);
            eVar += bVar + aVar;
            eVar *= halfVal;
            // Normal approximation fallback when x1 < x0.
            dimFallback_.Mean(eMean).Variance(eVar);
        }
    }

    __device__ half2 operator()(unsigned int a, half2 val) const
    {
        assert(nCam == 1);
        assert(a < numAnalogs);
        assert(a < detModel_.DetectionModes().size());

        // Score between x0 and x1.
        const auto lowExtrema = __hltu2(val, x0_[threadIdx.x]);
        const auto cap = Blend(lowExtrema, x0_[threadIdx.x], x1_[a][threadIdx.x]);
        const auto fixedTerm = Blend(lowExtrema, bFixedTerm_[threadIdx.x], pFixedTerm_[a][threadIdx.x]);

        const auto extrema = HalfOr(lowExtrema, __hgtu2(val, x1_[a][threadIdx.x]));
        half2 s = Blend(extrema, fixedTerm * h2pow2(cap - cap), __float2half2_rn(0.0f)) - logPMean_[a][threadIdx.x];

        if (a == numAnalogs-1)
        {
            // Fall back to normal approximation when x1 < x0.
            s = Blend(__hleu2(x0_[threadIdx.x], x1_[a][threadIdx.x]), s, dimFallback_(val));
        }
        return s;
    }
 private:
    // Log of displacement of pulse means from baseline mean.
    CudaArray<Row, numAnalogs> logPMean_;

    // Joints in the score function.
    Row x0_;
    CudaArray<Row, numAnalogs> x1_;

    Row bFixedTerm_;
    CudaArray<Row, numAnalogs> pFixedTerm_;

    // Normal approximation fallback for dimmest analog when x1 < x0.
    NormalLog<laneWidth> dimFallback_;
};

template <size_t laneWidth>
struct BlockStateScorer
{
    static constexpr unsigned int numStates = 13;
    static constexpr unsigned int numAnalogs = 4;
    using Row = CudaArray<half2, laneWidth>;

    __device__ static constexpr int FullState(int i) { return i+1; }
    __device__ static constexpr int UpState(int i) { return i+i + numAnalogs; }
    __device__ static constexpr int DownState(int i) { return i+i + 2*numAnalogs; }

    BlockStateScorer() = default;
    __device__ void Setup(const BlockModelParameters<laneWidth>& model)
    {
        static constexpr float log2pi_f = 1.8378770664f;
        const half2 nhalfVal = __float2half2_rn(-0.5f);
        const half2 one = __float2half2_rn(1.0f);

        sfScore_.Setup(model);

        const half2 normConst = __float2half2_rn(log2pi_f / 2.0f);

        // Background
        const auto& bgMode = model.BaselineMode();

        // Put -0.5 * log(det(V)) into bgFixedTerm_; compute bgInvCov_.
        // sym2x2MatrixInverse returns the determinant and writes the inverse
        // into the second argument.
        bgInvCov_[threadIdx.x] = one / bgMode.vars[threadIdx.x];
        bgFixedTerm_[threadIdx.x] = nhalfVal * h2log(bgInvCov_[threadIdx.x]) - normConst;
        bgMean_[threadIdx.x] = bgMode.means[threadIdx.x];

        for (unsigned int i = 0; i < numAnalogs; ++i)
        {
            // Full-frame states
            const auto& aMode = model.AnalogMode(i);
            ffFixedTerm_[i][threadIdx.x] = nhalfVal * h2log(one / aMode.vars[threadIdx.x]) - normConst;
            ffmean_[i][threadIdx.x] = aMode.means[threadIdx.x];
        }
    }

    __device__ CudaArray<half2, numStates> StateScores(half2 data) const
    {
        CudaArray<half2, numStates> score;
        const half2 halfVal = __float2half2_rn(0.5f);
        const half2 nhalfVal = __float2half2_rn(-0.5f);
        const half2 one = __float2half2_rn(1.0f);
        const half2 zero = __float2half2_rn(0.0f);

        {   // Compute the score for the background state.
            const auto mu = bgMean_[threadIdx.x];
            const auto y = data - mu;
            score[0] = nhalfVal * y * bgInvCov_[threadIdx.x] * y + bgFixedTerm_[threadIdx.x];
            // By definition, the ROI trace value for background is 0.
        }

        // Compute scores for full-frame pulse states.
        for (int i = 0; i < numAnalogs; ++i)
        {
            const auto j = FullState(i);
            const auto mu = ffmean_[i][threadIdx.x];
            const auto y = data - mu;
            score[j] = nhalfVal * y * ffInvCov_[i][threadIdx.x]*y + ffFixedTerm_[i][threadIdx.x];
            //score[j] += trc_.Roi(j, frame); // ROI exclusion
        }

        // Compute scores for subframe states.
        for (int i = 0; i < numAnalogs; ++i)
        {
            using std::min;
            const auto j = UpState(i);
            const auto k = DownState(i);
            const auto ff = FullState(i);

            score[j] = sfScore_(i, data);

            //score[j] += trc_.Roi(ff, frame);

            // When signal is stronger than full-frame mean, ensure that
            // subframe density does not exceed full-frame density.
            const auto mu = ffmean_[i][threadIdx.x];
            const auto xmu = data * mu;
            const auto mumu = mu*mu;
            // xmu > mumu means that the Euclidean projection of x onto mu is
            // greater than the norm of mu.
            auto cond = __hgtu2(xmu, mumu);
            score[j] = Blend(cond,
                             h2min(h2min(score[j], score[ff] - one), zero),
                             score[j]);

            // Could apply a similar constraint on the other side to prevent
            // large negative spikes from getting labelled as subframe instead
            // of baseline.

            // "Down" states assigned same score as "up" states.
            score[k] = score[j];
        }

        return score;
    }

    // Pre-computed inverse covariances.
    Row bgInvCov_;    // background (a.k.a. baseline)
    Row bgMean_;
    CudaArray<Row, numAnalogs> ffInvCov_; // full-frame states
    CudaArray<Row, numAnalogs> ffmean_; // full-frame states

    // The terms in the state scores that do not depend explicitly on the trace.
    // -0.5 * log det(V) - log 2\pi.
    Row bgFixedTerm_;    // background (a.k.a. baseline)
    CudaArray<Row, numAnalogs> ffFixedTerm_;    // full-frame states

    SubframeScorer<laneWidth> sfScore_;
};

// First arg should be const?
__global__ void FrameLabelerKernel(DevicePtr<TransitionMatrix> trans,
                                   DeviceView<BlockModelParameters<32>> models,
                                   GpuBatchData<short2> input,
                                   GpuBatchData<short2> output)
{
    static constexpr unsigned laneWidth = 32;
    assert(blockDim.x == laneWidth);
    __shared__ BlockStateScorer<laneWidth> scorer;
    scorer.Setup(models[blockIdx.x]);

    auto inZmw = input.ZmwData(blockIdx.x, threadIdx.x);
    auto score = scorer.StateScores({__short2half_rn(inZmw[0].x), __short2half_rn(inZmw[0].y)});

}

std::unique_ptr<GeneratorBase<int16_t>> MakeDataGenerator(const Data::DataManagerParams& dataParams,
                                                         const Data::PicketFenceParams& picketParams,
                                                         const Data::TraceFileParams& traceParams)
{
    return traceParams.traceFileName.empty()
        ? std::unique_ptr<GeneratorBase<int16_t>>(new PicketFenceGenerator(dataParams, picketParams))
        : std::unique_ptr<GeneratorBase<int16_t>>(new SignalGenerator(dataParams, traceParams));

}


void run(const Data::DataManagerParams& dataParams,
         const Data::PicketFenceParams& picketParams,
         const Data::TraceFileParams& traceParams,
         size_t simulKernels)
{

    DeviceOnlyObj<TransitionMatrix> trans;
    std::vector<UnifiedCudaArray<BlockModelParameters<32>>> models;
    const auto numBatches = dataParams.numZmwLanes / dataParams.kernelLanes;
    for (size_t i = 0; i < numBatches; ++i)
    {
        models.emplace_back(dataParams.kernelLanes, SyncDirection::HostWriteDeviceRead);
        // TODO populate models
    }

    auto tmp = [&trans, &models, &dataParams](
        TraceBatch<int16_t>& batch,
        size_t batchIdx,
        TraceBatch<int16_t>& ret)
    {
        if (dataParams.laneWidth != 64) throw PBException("Lane width not currently configurable.  Must be 64 zmw");
        FrameLabelerKernel<<<dataParams.kernelLanes, dataParams.laneWidth/2>>>(
            trans.GetDevicePtr(),
            models[batchIdx].GetDeviceHandle(),
            batch,
            ret);
    };

    ZmwDataManager<int16_t, int16_t> manager(dataParams, MakeDataGenerator(dataParams, picketParams, traceParams));
    RunThreads(simulKernels, manager, tmp);
}

}}
