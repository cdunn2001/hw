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

#include "SimulatedDataSource.h"

#include <random>

#include <boost/multi_array.hpp>

namespace PacBio {
namespace Application {

using namespace PacBio::DataSource;

class SimulatedDataSource::DataCache {
public:

    DataCache(const SimulatedDataSource::SimConfig& config,
              size_t zmwPerBlock, size_t framesPerBlock,
              std::unique_ptr<SignalGenerator> generator)
    {
        auto RoundUp = [](size_t count, size_t quantum)
        {
            auto numQuanta = (count + quantum - 1) / quantum;
            return numQuanta * quantum;
        };

        // Need to handle the case where the requested extents don't tile easily into
        // the block size.  Just roudning them up as a simple solution that lets us only
        // worry about data replication on the block level.
        const auto numSignals = RoundUp(config.NumSignals(), zmwPerBlock);
        const auto numFrames = RoundUp(config.NumFrames(), framesPerBlock);

        numChunks_ = numFrames / framesPerBlock;
        framesPerBlock_ = framesPerBlock;
        numLanes_ = numSignals / zmwPerBlock;
        zmwPerBlock_ = zmwPerBlock;

        // Set up a data cache that is populated with integral blocks.  This makes it easy to
        // replicate it out as requests for new blocks come in (you just modulo the lane/chunk idx)
        data_.resize(boost::extents[numChunks_][numLanes_][framesPerBlock_][zmwPerBlock_]);

        size_t lane = 0;
        for (size_t signalIdx = 0; signalIdx < numSignals; lane++)
        {
            for(size_t zmwIdx = 0; zmwIdx < zmwPerBlock; ++zmwIdx, ++signalIdx)
            {
                const auto& signal = generator->GenerateSignal(numFrames, signalIdx);
                size_t chunk = 0;
                for (size_t frame = 0; frame < numFrames; ++chunk)
                {
                    for (size_t frameIdx = 0; frameIdx < framesPerBlock; ++frameIdx, ++frame)
                    {
                        data_[chunk][lane][frameIdx][zmwIdx] = signal[frame];
                    }
                }
            }
        }

    }

    // Takes a block from the cache and uses it to populate a block in a SensorPacket
    void FillBlock(size_t laneIdx, size_t chunkIdx, SensorPacket::DataView data)
    {
        auto cacheBlock = data_[chunkIdx % numChunks_][laneIdx % numLanes_].origin();
        memcpy(data.Data(),
               cacheBlock,
               framesPerBlock_*zmwPerBlock_*sizeof(int16_t));
    }

private:
    size_t numLanes_ = 0;
    size_t numChunks_ = 0;
    size_t zmwPerBlock_ = 0;
    size_t framesPerBlock_ = 0;
    boost::multi_array<int16_t, 4> data_;
};

SimulatedDataSource::~SimulatedDataSource() = default;

SimulatedDataSource::SimulatedDataSource(size_t minZmw,
                    const SimConfig& sim,
                    DataSource::DataSourceBase::Configuration cfg,
                    std::unique_ptr<SignalGenerator> generator)
    : DataSource::DataSourceBase(std::move(cfg))
    , cache_(std::make_unique<DataCache>(sim,
                                         GetConfig().layout.BlockWidth(),
                                         GetConfig().layout.NumFrames(),
                                         std::move(generator)))
    , numZmw_([&](){
        const auto zmwPerBatch = GetConfig().layout.NumZmw();
        const auto batchesRoundUp = (minZmw + zmwPerBatch - 1) / zmwPerBatch;
        return batchesRoundUp * zmwPerBatch;
    }())
{
    currChunk_ = SensorPacketsChunk(0, GetConfig().layout.NumFrames());
    currChunk_.SetZmwRange(0, numZmw_);
}

void SimulatedDataSource::ContinueProcessing()
{
    const auto& layout = GetConfig().layout;

    const auto startZmw = batchIdx_ * layout.NumZmw();
    const auto startFrame = chunkIdx_ * layout.NumFrames();
    SensorPacket batchData(layout, startZmw,
                           startFrame,
                           *GetConfig().allocator);
    for (size_t i = 0; i < layout.NumBlocks(); ++i)
    {
        cache_->FillBlock(batchIdx_ * layout.NumBlocks() + i,
                              chunkIdx_,
                              batchData.BlockData(i));
    }
    currChunk_.AddPacket(std::move(batchData));

    batchIdx_++;
    if (batchIdx_ == NumBatches())
    {
        batchIdx_ = 0;
        chunkIdx_++;
        this->PushChunk(std::move(currChunk_));
        currChunk_ = SensorPacketsChunk(chunkIdx_ * layout.NumFrames(),
                                        (chunkIdx_ + 1) * layout.NumFrames());
        currChunk_.SetZmwRange(0, numZmw_);
    }

    if (currChunk_.StartFrame() >= GetConfig().numFrames)
    {
        SetDone();
    }
}

std::vector<int16_t> SawtoothGenerator::GenerateSignal(size_t numFrames, size_t idx)
{
    std::vector<int16_t> ret(numFrames);

    const auto range = config_.maxAmp - config_.minAmp;
    const auto slope = static_cast<double>(range) / config_.periodFrames;
    for (size_t i = 0; i < numFrames; ++i)
    {
        ret[i] = config_.minAmp + static_cast<int16_t>(slope*(i+idx*config_.startFrameStagger)) % range;
    }

    return ret;
}

std::vector<int16_t> PicketFenceGenerator::GenerateSignal(size_t numFrames, size_t idx)
{
    std::mt19937 gen(config_.seedFunc(idx));

    auto GetBaselineSignal = [&](uint16_t frames)
    {
        std::normal_distribution<> rng(config_.baselineSignalLevel, config_.baselineSigma);
        std::vector<short> baselineSignal;
        for (size_t f = 0; f < frames; ++f)
        {
            baselineSignal.push_back(std::floor(rng(gen)));
        }
        return baselineSignal;
    };

    auto GetPulseSignal = [&](uint16_t frames)
    {
        std::uniform_int_distribution<> rng(0, config_.pulseSignalLevels.size() - 1);
        short pulseLevel = config_.pulseSignalLevels[rng(gen)] - config_.baselineSignalLevel;
        std::vector<short> baselineSignal = GetBaselineSignal(frames);

        std::vector<short> pulseSignal;
        for (size_t f = 0; f < frames; ++f)
        {
            pulseSignal.push_back(baselineSignal[f] + pulseLevel);
        }
        return pulseSignal;
    };

    auto GetPulseWidth = [&]()
    {
        if (config_.generatePoisson)
        {
            std::exponential_distribution<> rng(config_.pulseWidthRate);
            return static_cast<uint16_t>(std::ceil(rng(gen)));
        }
        else
        {
            return config_.pulseWidth;
        }
    };

    auto GetPulseIpd = [&]()
    {
        if (config_.generatePoisson)
        {
            std::exponential_distribution<> rng(config_.pulseIpdRate);
            return static_cast<uint16_t>(std::ceil(rng(gen)));
        }
        else
        {
            return config_.pulseIpd;
        }
    };

    std::vector<int16_t> signal;

    size_t frame = 0;
    while (frame < numFrames)
    {
        // Generate IPD followed by PW.
        uint16_t pulseIpd = GetPulseIpd();
        uint16_t nIpdFrames = std::min(numFrames - frame, static_cast<size_t>(pulseIpd));
        std::vector<short> baselineSignal = GetBaselineSignal(pulseIpd);
        signal.insert(std::end(signal), std::begin(baselineSignal),
                      std::begin(baselineSignal) + nIpdFrames);
        frame += nIpdFrames;

        uint16_t pulseWidth = GetPulseWidth();
        uint16_t nPwFrames = std::min(numFrames - frame, static_cast<size_t>(pulseWidth));
        std::vector<short> pulseSignal = GetPulseSignal(pulseWidth);
        signal.insert(std::end(signal), std::begin(pulseSignal),
                           std::begin(pulseSignal) + nPwFrames);
        frame += nPwFrames;
    }
    assert(signal.size() == numFrames);

    return signal;
}

std::vector<int16_t> SortedGenerator::GenerateSignal(size_t numFrames, size_t idx)
{
    auto signal = gen_->GenerateSignal(numFrames, idx);
    std::sort(signal.begin(), signal.end());
    return signal;
}

std::vector<int16_t> RandomizedGenerator::GenerateSignal(size_t numFrames, size_t idx)
{
    std::mt19937 gen(config_.seedFunc(idx));
    auto signal = gen_->GenerateSignal(numFrames, idx);
    std::shuffle(signal.begin(), signal.end(), gen);
    return signal;
}

}} //::PacBio::Application
