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

#ifndef PACBIO_APPLICATION_SIMULATED_DATA_SOURCE_H
#define PACBIO_APPLICATION_SIMULATED_DATA_SOURCE_H

#include <pacbio/datasource/DataSourceBase.h>

namespace PacBio {
namespace Application {

// Virtual API for generating data.  Serves as a customization point for
// SimulatedDataSource, allowing multiple implementations of data simulation
class SignalGenerator
{
public:
    virtual std::vector<int16_t> GenerateSignal(size_t numFrames, size_t idx) = 0;
    virtual ~SignalGenerator() = default;
};

// DataSource implementation for simulated data.  Actual data generation
// is handled by the SignalGenerator handed in, with this class primarily
// concerned with data replication and satisfying the DataSourceBase API
class SimulatedDataSource : public DataSource::DataSourceBase
{
public:
    // Forward delaration for the class that will hold the generated data
    // and replicate it out on demand.
    class DataCache;

    // Describes the extents of data to simulate.  Both numSignals and
    // numFrames are strictly minimums, and may be rounded up to more
    // naturally fit chunk/lane boundaries.
    struct SimConfig {
        SimConfig(size_t numSignals, size_t numFrames)
            : numSignals_(numSignals)
            , numFrames_(numFrames)
        {}

        size_t NumSignals() const { return numSignals_; }
        size_t NumFrames() const { return numFrames_; }
    private:
        size_t numSignals_;
        size_t numFrames_;
    };

    SimulatedDataSource(size_t minZmw,
                        const SimConfig& sim,
                        DataSource::DataSourceBase::Configuration cfg,
                        std::unique_ptr<SignalGenerator> generator);

    ~SimulatedDataSource();

    std::vector<uint32_t> UnitCellFeatures() const override
    {
        return std::vector<uint32_t>(NumZmw(), 0);
    }

    std::vector<uint32_t> UnitCellIds() const override
    {
        std::vector<uint32_t> ret(NumZmw());
        iota(ret.begin(), ret.end(), 0);
        return ret;
    }
    std::vector<uint32_t> PoolIds() const override
    {
        std::vector<uint32_t> ret(NumBatches());
        iota(ret.begin(), ret.end(), 0);
        return ret;
    }

    size_t NumBatches() const override
    {
        return numZmw_ / GetConfig().layout.NumZmw();
    }

    size_t NumFrames() const override
    {
        return GetConfig().numFrames;
    }

    size_t NumZmw() const override
    {
        return numZmw_;
    }

    double FrameRate() const override
    {
        return 100.0;
    }

    DataSource::HardwareInformation GetHardwareInformation() override
    {
        DataSource::HardwareInformation ret;
        ret.SetShortName("Simulated DataSource");
        return ret;
    }

    void ContinueProcessing() override;
private:
    std::unique_ptr<DataCache> cache_;
    size_t numZmw_;

    size_t batchIdx_ = 0;
    size_t chunkIdx_ = 0;

    DataSource::SensorPacketsChunk currChunk_;
};

// Generates a constant signal where the entire signal is the same as the signal index
class ConstantGenerator : public SignalGenerator
{
public:
    ConstantGenerator() = default;

    std::vector<int16_t> GenerateSignal(size_t numFrames, size_t idx) override
    {
        std::vector<int16_t> ret(numFrames, idx);
        return ret;
    }
};

// Generates a crude sequence of baseline/pulses.
// * Pulses are generated, either with a fixed width/ipd
//   or with a poisson distrbution
// * Pulse amplitudes are configurable (can be distinct or all the same)
// * A noisy baseline is generated and added to *all* frames (so pulses have
//   some amplitude variability as well)
// Baseline (with noise) is generated
class PicketFenceGenerator : public SignalGenerator
{
public:
    struct Config
    {
        // Determines if fixed width/ipd or poisson distribution is used
        bool generatePoisson = true;
        // If fixed, directly controls the pulse characteritics
        uint16_t pulseWidth = 5;
        uint16_t pulseIpd = 10;
        // If generatePoisson==true controls the distrubtion
        float pulseWidthRate = 0.2f;
        float pulseIpdRate = 0.1f;
        // Mean/sigma of the baseline noise
        short baselineSignalLevel = 200;
        short baselineSigma = 20;
        // Pulse amplitudes, should be 1 or 4 elements long
        std::vector<short> pulseSignalLevels = { 200 };
        // Optional function controlling rng seed.  Input will be the
        // signal number.
        std::function<size_t(size_t)> seedFunc = [](size_t i) { return i; };
    };
    PicketFenceGenerator(const Config& config)
        : config_(config)
    {}

    std::vector<int16_t> GenerateSignal(size_t numFrames, size_t idx) override;

private:
    Config config_;
};

// Generates a sawtooth signal with configurable
// peaks and period.
class SawtoothGenerator : public SignalGenerator
{
public:
    struct Config
    {
        size_t periodFrames = 128;
        int16_t minAmp = 0;
        int16_t maxAmp = 200;
        // A nonzero stagger means subsequent ZMW will start
        // at a different point in the configured sequence.
        int16_t startFrameStagger = 0;
    };
    SawtoothGenerator(const Config& config)
        : config_(config)
    {
        if (config_.maxAmp <= config_.minAmp)
            throw PBException("Invalid SawtoothConfig: Max not greater than min");
    }

    std::vector<int16_t> GenerateSignal(size_t numFrames, size_t idx) override;

private:
    Config config_;
};

// Meta generator, that takes another generator and sorts it's signal.
// Mostly useful for niche cases, e.g. binning perf tests, where data
// order doesn't actually matter, but a temporal correlations (or lack of)
// in the data can affect the speed of some binning techniques.
class SortedGenerator : public SignalGenerator
{
public:
    template <typename Gen, typename... Args>
    static std::unique_ptr<SortedGenerator> Create(Args&&... args)
    {
        return std::make_unique<SortedGenerator>(std::make_unique<Gen>(std::forward<Args>(args)...));
    }
    SortedGenerator(std::unique_ptr<SignalGenerator> gen)
        : gen_(std::move(gen))
    {}

    std::vector<int16_t> GenerateSignal(size_t numFrames, size_t idx) override;
private:
    std::unique_ptr<SignalGenerator> gen_;
};

// Meta generator, that takes another generator and randomizes it's signal.
// Mostly useful for niche cases, e.g. binning perf tests, where data
// order doesn't actually matter, but a temporal correlations (or lack of)
// in the data can affect the speed of some binning techniques.
class RandomizedGenerator : public SignalGenerator
{
public:
    struct Config
    {
        std::function<size_t(size_t)> seedFunc = [](size_t i) { return i; };
    };
    template <typename Gen, typename... Args>
    static std::unique_ptr<RandomizedGenerator> Create(const Config& config, Args&&... args)
    {
        return std::make_unique<RandomizedGenerator>(config, std::make_unique<Gen>(std::forward<Args>(args)...));
    }
    RandomizedGenerator(const Config& config, std::unique_ptr<SignalGenerator> gen)
        : gen_(std::move(gen))
        , config_(config)
    {}

    std::vector<int16_t> GenerateSignal(size_t numFrames, size_t idx) override;
private:
    std::unique_ptr<SignalGenerator> gen_;
    Config config_;
};

}} // ::PacBio::Application

#endif //PACBIO_APPLICATION_SIMULATED_DATA_SOURCE_H
