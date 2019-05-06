#include "FrameLabeler.h"

#include <common/cuda/PBCudaSimd.cuh>
#include <common/cuda/memory/DeviceOnlyArray.cuh>
#include <common/KernelThreadPool.h>

#include <dataTypes/TraceBatch.cuh>

using namespace PacBio::Cuda::Memory;
using namespace PacBio::Cuda::Data;
using namespace PacBio::Mongo::Data;

namespace PacBio {
namespace Cuda {

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

template <typename T, size_t len>
struct CudaArray
{
    //temporary hack, please kill
    CudaArray() = default;
    CudaArray(T val)
    {
        for (size_t i = 0; i < len; ++i)
        {
            data_[i] = val;
        }
    }
    __device__ __host__ T& operator[](unsigned idx) { return data_[idx]; }
    __device__ __host__ const T& operator[](unsigned idx) const { return data_[idx]; }
private:
    T data_[len];
};

struct __align__(128) TransitionMatrix
{
    static constexpr int numAnalogs = 4;
    static constexpr int numStates = 13;
    using T = half;
    using Row = CudaArray<T, numStates>;

    // These function is now duplicated...
    __device__ static constexpr int FullState(int i) { return i+1; }
    __device__ static constexpr int UpState(int i) { return i+1 + numAnalogs; }
    __device__ static constexpr int DownState(int i) { return i+1 + 2*numAnalogs; }

    __device__ T operator()(int row, int col) const { return data_[row][col]; }
    __device__ T Entry(int row, int col) const { return data_[row][col]; }

    __device__ TransitionMatrix(//BasecallerPulseDetectionConfig& config,
                                CudaArray<float, numAnalogs> pw,
                                CudaArray<float, numAnalogs> ipd,
                                CudaArray<float, numAnalogs> pwSSRatios,
                                CudaArray<float, numAnalogs> ipdSSRatios)
    {
        half zero = __float2half(0.0f);
        for (int i = 0; i < numStates; ++i)
        {
            for (int j = 0; j < numStates; ++j)
            {
                data_[i][j] = zero;
            }
        }
        // Algorithm courtesy Rob Grothe.

        // Avoid singularities in the computations to follow.  Could probably
        // find a new analytic form without singularities, but just avoid
        // ratios of 1.0 for now.
        auto pwsRatios = pwSSRatios;
        auto ipdRatios = ipdSSRatios;
        for (size_t i = 0; i < numAnalogs; ++i)
        {
            static constexpr float delta = 0.01f;
            if (std::abs(pwsRatios[i] - 1.0f) < delta)
            {
                pwsRatios[i] = (pwsRatios[i] < 1.0f) ? 1.0f - delta : 1.0f + delta;
            }
            if (std::abs(ipdRatios[i] - 1.0f) < delta)
            {
                ipdRatios[i] = (ipdRatios[i] < 1.0f) ? 1.0f - delta : 1.0f + delta;
            }
        }

        // Mean IPD for the input analog set.
        const float meanIpd = [&ipd](){
            float sum = 0.0f;
            for (int i = 0; i < numAnalogs; ++i) sum+=ipd[i];
            return sum/numAnalogs;
        }();

        // Given that an analog, with an average width and two-step ratio,
        // what are the probabilities that we detect a 1, 2 or 3 frame pulse.
        // See analytic work done in SPI-1674 and particularly subtask
        // SPI-1754
        auto computeLenProbs = [&](float width, float ratio) -> CudaArray<float, 3>
        {
            // Make sure r is small instead of 0, to avoid indeterminate forms
            // in the math below.
            ratio = std::max(ratio, 1e-6f);

            const auto k1 = (1.0f + ratio) / width;
            const auto k2 = (1.0f + 1.0f / ratio) / width;
            const auto dk = k1 - k2;

            const auto c1 = 1.0f - std::exp(-k1);
            const auto c2 = 1.0f - std::exp(-k2);

            CudaArray<float, 3> ret;
            ret[0] = 1 - k1 * c2 / k2 / dk + k2 * c1 / k1 / dk;
            ret[1] = k1 * (c2 * c2) / k2 / dk - k2 * (c1 * c1) / k1 / dk;
            ret[2] = k1 * (c2 * c2) * (1 - c2) / k2 / dk - k2 * (c1 * c1) * (1 - c1) / k1 / dk;
            return ret;
        };

        // Slurp in fudge factors from the config file
        const auto alphaFactor = 1.0f;//config.Alpha;
        const auto betaFactor = 1.0f;//config.Beta;
        const auto gammaFactor = 1.0f;//config.Gamma;

        // Precompute the probabilities of 1, 2 and 3 frame ipds after each analog
        CudaArray<CudaArray<float, 3>, numAnalogs> ipdLenProbs;
        ipdLenProbs[0] = computeLenProbs(ipd[0], ipdRatios[0]);
        ipdLenProbs[1] = computeLenProbs(ipd[1], ipdRatios[1]);
        ipdLenProbs[2] = computeLenProbs(ipd[2], ipdRatios[2]);
        ipdLenProbs[3] = computeLenProbs(ipd[3], ipdRatios[3]);

        // alpha[i] is probability of going from the i down to another up state.
        // The direct down-to-full is currently disallowed, so we roll both
        // the 1 and 2 frame ipd detection probabilities into here.  Factor
        // of 0.25 is to account that each of the 4 analogs are equally likely
        // when coming out of a down state.
        CudaArray<float, numAnalogs> alpha;
        alpha[0] = alphaFactor * 0.25f * (ipdLenProbs[0][0] + ipdLenProbs[0][1]);
        alpha[1] = alphaFactor * 0.25f * (ipdLenProbs[1][0] + ipdLenProbs[1][1]);
        alpha[2] = alphaFactor * 0.25f * (ipdLenProbs[2][0] + ipdLenProbs[2][1]);
        alpha[3] = alphaFactor * 0.25f * (ipdLenProbs[3][0] + ipdLenProbs[3][1]);
        const auto& alphaSum = [&]() {
            float sum = 0.0f;
            for (int i = 0; i < numAnalogs; ++i)
            {
                sum += alpha[i];
            }
            return sum;
        }();

        // If we're passing through a full baseline state, the transition to
        // a new pulse will not know what previous pulse it came from.  In this
        // case use an averaged IPD, with a ratio of 0 (since it could potentially
        // be a long IPD)
        CudaArray<float, 3> meanIpdLenProbs = computeLenProbs(meanIpd, 0.0f);

        // Probability of 3 *or more* frame IPD. Since the d->u transition is
        // up to 2 frames, this is also the probability of entering the full
        // baseline state (given that a pulse has ended)
        const auto meanIpd3m = 1 - meanIpdLenProbs[0] - meanIpdLenProbs[1];

        // an IPD of 3 means we went d->0->u'.  Thus the 0->u probability is
        // that, given that we've passed through the full 0 state (e.g. meanIpd3m)
        const auto pulseStartProb = meanIpdLenProbs[2] / meanIpd3m;

        data_[0][0] = 1.0f - pulseStartProb;

        for (int i = 0; i < numAnalogs; ++i)
        {
            const auto full = FullState(i);
            const auto up = UpState(i);
            const auto down = DownState(i);

            // probabilities of 1,2 and 3 frame pulses for this analog
            const auto& analogLenProbs = computeLenProbs(pw[i], pwsRatios[i]);

            // Probability of a 1 frame pulse is the sum of P(u->0) and all of P(u->u')
            // The former is gamma and the later is alpha. So summing over alpha
            // and solving for gamma gives us:
            const auto gamma = gammaFactor * analogLenProbs[0] / (1 + alphaSum);

            // Fudged probability of 2 frame pulse
            const auto beta = betaFactor * analogLenProbs[1];

            // N.B. For the below, remember that the matrix is indexed as
            // transScore_(current, previous).  Transitions read reversed
            // from what one might expect!

            // Give each analog equal probability to start
            data_[up][0] = 0.25f * pulseStartProb;

            // Record exactly 1 and 2 frame pulses.
            data_[0][up] = gamma;
            data_[down][up] = beta;

            // Normalize by subtracting out all alternative paths:
            // gamma            -- u -> 0
            // gamma * alphaSum -- u->u' (summed over all u')
            // beta             -- u -> d
            data_[full][up] = 1 - gamma - gamma * alphaSum - beta;

            // Probability that pulse is 3 or more frames long.  Subsequently
            // also the probability that we at least pass through the full frame
            // state.
            const auto prob3ms = 1.0f - analogLenProbs[0] - analogLenProbs[1];

            // analogLenProbs[2] = u->T->d path.  The T->d transition is that
            // probability *given* we've passed through the T state (e.g. prob3sum)
            data_[down][full] = analogLenProbs[2] / prob3ms;

            // Normalize by subtracting out the (only) alternative path
            data_[full][full] = __float2half(1.0f) - data_[down][full];

            // Again normalize by subtracting out the alternatives, which are
            // all the alpha terms that let us go from d->u'
            data_[0][down] = 1 - alphaSum;

            // Handle the dense section of u->u' and d->u' transitions
            for (int j = 0; j < numAnalogs; j++)
            {
                auto upPrime = UpState(j);

                // Alpha already defined as 1 or 2 frame ipd event
                // (no full baseline frame)
                data_[upPrime][down] = alpha[j];

                // To go handle u->u' we also need to include the probability
                // of a 1 frame pulse before the 1 or 2 frame ipd.
                data_[upPrime][up] = gamma * alpha[j];
            }
        }

        // Convert to log-prob scores
        for (int i = 0; i < numStates; ++i)
        {
            for (int j = 0; j < numStates; ++j)
            {
                if (data_[i][j] != zero) data_[i][j] = hlog(data_[i][j]);
            }
        }
    }

private:
    CudaArray<Row, numStates> data_;
};

template <size_t laneWidth>
class __align__(128) NormalLog
{
    using Row = CudaArray<PBHalf2, laneWidth>;
    static constexpr float log2pi_f = 1.8378770664f;
public:     // Structors
    NormalLog() = default;
    __device__ NormalLog(const PBHalf2& mean,
                         const PBHalf2& variance)
    {
        Mean(mean);
        Variance(variance);
    }

public:     // Properties
    __device__ const PBHalf2& Mean() const
    { return mean_[threadIdx.x]; }

    /// Sets Mean to new value.
    /// \returns *this.
    __device__ NormalLog& Mean(const PBHalf2& value)
    {
        mean_[threadIdx.x] = value;
        return *this;
    }

    __device__ PBHalf2 Variance() const
    { return 0.5f / scaleFactor_[threadIdx.x]; }

    /// Sets Variance to new value, which must be positive.
    /// \returns *this.
    __device__ NormalLog& Variance(const PBHalf2& value)
    {
        //assert(all(value > 0.0f));
        normTerm_[threadIdx.x] = PBHalf2(-0.5f) * (PBHalf2(log2pi_f) + log(value));
        scaleFactor_[threadIdx.x] = PBHalf2(0.5f) / value;
        return *this;
    }

public:
    /// Evaluate the distribution at \a x.
    __device__ PBHalf2 operator()(const PBHalf2& x) const
    { return normTerm_[threadIdx.x] - pow2(x - mean_[threadIdx.x]) * scaleFactor_[threadIdx.x]; }

private:    // Data
    Row mean_;
    Row normTerm_;
    Row scaleFactor_;
};

template <size_t laneWidth>
struct __align__(128) BlockAnalogMode
{
    using Row = CudaArray<PBHalf2, laneWidth>;
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
    using Row = CudaArray<PBHalf2, laneWidth>;

    SubframeScorer() = default;

    __device__ void Setup(const BlockModelParameters<laneWidth>& model)
    {
        static constexpr float pi_f = 3.1415926536f;
        const PBHalf2 sqrtHalfPi = PBHalf2(std::sqrt(0.5f * pi_f));
        const PBHalf2 halfVal = PBHalf2(0.5f);
        const PBHalf2 nhalfVal = PBHalf2(-0.5f);
        const PBHalf2 oneSixth = PBHalf2(1.0f / 6.0f);

        const auto& bMode = model.BaselineMode();
        const auto bMean = bMode.means[threadIdx.x];
        const auto bVar = bMode.vars[threadIdx.x];
        const auto bSigma = sqrt(bVar);

        // Low joint in the score function.
        x0_[threadIdx.x] = bMean + bSigma * sqrtHalfPi;
        bFixedTerm_[threadIdx.x] = nhalfVal / bVar;

        for (unsigned int a = 0; a < numAnalogs; ++a)
        {
            const auto& dma = model.AnalogMode(a);
            const auto aMean = dma.means[threadIdx.x];
            logPMean_[a][threadIdx.x] = log(aMean - bMean);
            const auto aVar = dma.vars[threadIdx.x];
            const auto aSigma = sqrt(aVar);
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

    __device__ PBHalf2 operator()(unsigned int a, PBHalf2 val) const
    {
        // Score between x0 and x1.
        const auto lowExtrema = val < x0_[threadIdx.x];
        const auto cap = Blend(lowExtrema, x0_[threadIdx.x], x1_[a][threadIdx.x]);
        const auto fixedTerm = Blend(lowExtrema, bFixedTerm_[threadIdx.x], pFixedTerm_[a][threadIdx.x]);

        const auto extrema = lowExtrema || (val > x1_[a][threadIdx.x]);
        PBHalf2 s = Blend(extrema, fixedTerm * pow2(cap - cap), PBHalf2(0.0f)) - logPMean_[a][threadIdx.x];

        if (a == numAnalogs-1)
        {
            // Fall back to normal approximation when x1 < x0.
            s = Blend(x0_[threadIdx.x] <= x1_[a][threadIdx.x], s, dimFallback_(val));
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
struct __align__(128) BlockStateScorer
{
    static constexpr unsigned int numStates = 13;
    static constexpr unsigned int numAnalogs = 4;
    using Row = CudaArray<PBHalf2, laneWidth>;

    __device__ static constexpr int FullState(int i) { return i+1; }
    __device__ static constexpr int UpState(int i) { return i+1 + numAnalogs; }
    __device__ static constexpr int DownState(int i) { return i+1 + 2*numAnalogs; }

    BlockStateScorer() = default;
    __device__ void Setup(const BlockModelParameters<laneWidth>& model)
    {
        static constexpr float log2pi_f = 1.8378770664f;
        const PBHalf2 nhalfVal = PBHalf2(-0.5f);
        const PBHalf2 one = PBHalf2(1.0f);

        sfScore_.Setup(model);

        const PBHalf2 normConst = PBHalf2(log2pi_f / 2.0f);

        // Background
        const auto& bgMode = model.BaselineMode();

        // Put -0.5 * log(det(V)) into bgFixedTerm_; compute bgInvCov_.
        // sym2x2MatrixInverse returns the determinant and writes the inverse
        // into the second argument.
        bgInvCov_[threadIdx.x] = one / bgMode.vars[threadIdx.x];
        bgFixedTerm_[threadIdx.x] = nhalfVal * log(bgInvCov_[threadIdx.x]) - normConst;
        bgMean_[threadIdx.x] = bgMode.means[threadIdx.x];

        for (unsigned int i = 0; i < numAnalogs; ++i)
        {
            // Full-frame states
            const auto& aMode = model.AnalogMode(i);
            ffFixedTerm_[i][threadIdx.x] = nhalfVal * log(one / aMode.vars[threadIdx.x]) - normConst;
            ffmean_[i][threadIdx.x] = aMode.means[threadIdx.x];
        }
    }

    __device__ CudaArray<PBHalf2, numStates> StateScores(PBHalf2 data) const
    {
        CudaArray<PBHalf2, numStates> score;
        const PBHalf2 halfVal = PBHalf2(0.5f);
        const PBHalf2 nhalfVal = PBHalf2(-0.5f);
        const PBHalf2 one = PBHalf2(1.0f);
        const PBHalf2 zero = PBHalf2(0.0f);

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
            score[j] = Blend(xmu > mumu,
                             min(min(score[j], score[ff] - one), zero),
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

template <typename T, size_t laneWidth>
struct ViterbiDataHost
{
    static constexpr int numStates = 13;
    ViterbiDataHost(size_t numFrames, T val = T{})
        : data_(numFrames*laneWidth*numStates, val)
    {}

    DeviceView<T> Data(detail::DataManagerKey)
    {
        return data_.GetDeviceView();
    }
 private:
    DeviceOnlyArray<T> data_;
};

template <typename T, size_t laneWidth>
struct ViterbiData : private detail::DataManager
{
    static constexpr int numStates = 13;
    ViterbiData(ViterbiDataHost<T, laneWidth>& hostData)
        : data_(hostData.Data(DataKey()))
    {}

    __device__ T& operator()(int frame, int state)
    {
        return data_[frame*laneWidth*numStates +  state*laneWidth +  threadIdx.x];
    }
 private:
    DeviceView<T> data_;
};

// First arg should be const?
__global__ void FrameLabelerKernel(DevicePtr<TransitionMatrix> trans,
                                   DeviceView<BlockModelParameters<32>> models,
                                   ViterbiData<PBHalf2, 32> recur,
                                   ViterbiData<short2, 32> labels,
                                   GpuBatchData<short2> input,
                                   GpuBatchData<short2> output)
{
    static constexpr unsigned laneWidth = 32;
    static constexpr int numStates = 12;
    assert(blockDim.x == laneWidth);
    __shared__ BlockStateScorer<laneWidth> scorer;
    scorer.Setup(models[blockIdx.x]);

    // Forward recursion
    const int numFrames = input.Dims().framesPerBatch;
    auto inZmw = input.ZmwData(blockIdx.x, threadIdx.x);
    for (int frame = 0; frame < numFrames; ++frame)
    {
        auto scores = scorer.StateScores(PBHalf2(inZmw[frame]));
        for (int nextState = 0; nextState < numStates; ++nextState)
        {
            auto score = scores[nextState];
            // Is this transposed?
            auto maxVal = score + PBHalf2(trans->Entry(0, nextState)) + recur(frame, 0);
            auto maxIdx = make_short2(0,0);
            for (int prevState = 1; prevState < numStates; ++prevState)
            {
                auto val = score + PBHalf2(trans->Entry(prevState, nextState)) + recur(frame, prevState);

                auto cond = maxVal > val;
                maxVal = Blend(cond, maxVal, val);
                maxIdx = Blend(cond, maxIdx, make_short2(prevState, prevState));
            }
            recur(frame+1, nextState) = maxVal;
            labels(frame, nextState) = maxIdx;
        }
    }

    // Cheat for now, find it for real later
    const int anchorIdx = numFrames-1;
    auto maxVal = recur(anchorIdx, 0);
    short2 traceState = make_short2(0,0);
    for (int state = 1; state < numStates; ++state)
    {
        auto val = recur(anchorIdx, state);
        auto cond = maxVal > val;

        maxVal = Blend(cond, maxVal, val);
        traceState = Blend(cond, traceState, make_short2(state, state));
    }

    // Traceback
    auto outZmw = output.ZmwData(blockIdx.x, threadIdx.x);
    for (int frame = numFrames-1; frame >= 0; --frame)
    {
        outZmw[frame] = traceState;
        traceState.x = labels(frame, traceState.x).x;
        traceState.y = labels(frame, traceState.x).y;
    }

    // Set the new boundary conditions for next block
    for (int state = 0; state < numStates; ++state)
    {
        recur(0, state) = PBHalf2(0);
    }

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
    static constexpr size_t gpuBlockThreads = 32;

    CudaArray<float, 4> pw(0.2f);
    CudaArray<float, 4> ipd(0.5f);
    CudaArray<float, 4> pwss(3.2f);
    CudaArray<float, 4> ipdss(0.0f);

    DeviceOnlyObj<TransitionMatrix> trans(pw, ipd, pwss, ipdss);
    std::vector<UnifiedCudaArray<BlockModelParameters<gpuBlockThreads>>> models;
    std::vector<ViterbiDataHost<PBHalf2, gpuBlockThreads>> scores;
    std::vector<ViterbiDataHost<short2, gpuBlockThreads>> labels;

    const auto numBatches = dataParams.numZmwLanes / dataParams.kernelLanes;
    models.reserve(numBatches);
    scores.reserve(numBatches);
    labels.reserve(numBatches);
    for (size_t i = 0; i < numBatches; ++i)
    {
        // ------------------------------------------------------
        // TODO we need to set our initial conditions somehow!!!
        // Currently hard coding to start in zero state.  Should be fine,
        // but we need to make sure it's built properly into the API
        // ------------------------------------------------------
        scores.emplace_back(dataParams.blockLength+1, PBHalf2(0.0f));
        labels.emplace_back(dataParams.blockLength);
        models.emplace_back(dataParams.kernelLanes, SyncDirection::HostWriteDeviceRead);
        // TODO populate models
    }

    auto tmp = [&trans, &models, &dataParams, &scores, &labels](
        TraceBatch<int16_t>& batch,
        size_t batchIdx,
        TraceBatch<int16_t>& ret)
    {
        if (dataParams.laneWidth != 2*gpuBlockThreads) throw PBException("Lane width not currently configurable.  Must be 64 zmw");
        FrameLabelerKernel<<<dataParams.kernelLanes, gpuBlockThreads>>>(
            trans.GetDevicePtr(),
            models[batchIdx].GetDeviceHandle(),
            scores[batchIdx],
            labels[batchIdx],
            batch,
            ret);
    };

    ZmwDataManager<int16_t, int16_t> manager(dataParams, MakeDataGenerator(dataParams, picketParams, traceParams));
    RunThreads(simulKernels, manager, tmp);
}

}}
