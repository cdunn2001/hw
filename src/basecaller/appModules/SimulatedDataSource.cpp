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

#include <common/MongoConstants.h>

namespace PacBio {
namespace Application {

using namespace PacBio::DataSource;
using namespace PacBio::Mongo;

size_t RoundUp(size_t count, size_t blksz)
{
    return (count + blksz - 1) / blksz * blksz;
}

class SimulatedDataSource::DataCache
{
public:
    DataCache(size_t numSignals, size_t numFrames,
              size_t zmwPerBlock, size_t framesPerBlock,
              std::unique_ptr<SignalGenerator> generator)
    {
        // Extend the tiling with rounding up
        const auto numSignalsUpper = RoundUp(numSignals, zmwPerBlock);
        const auto numFramesUpper  = RoundUp(numFrames, framesPerBlock);

        numChunks_ = numFramesUpper / framesPerBlock;
        framesPerBlock_ = framesPerBlock;
        numLanes_ = numSignalsUpper / zmwPerBlock;
        zmwPerBlock_ = zmwPerBlock;

        // Set up a data cache that is populated with integral blocks. This makes it easy to
        // replicate it out as requests for new blocks come in (you just modulo the lane/chunk idx)
        data_.resize(boost::extents[numChunks_][numLanes_][framesPerBlock_][zmwPerBlock_]);

        for (size_t lane = 0, signalIdx = 0; signalIdx < numSignalsUpper; lane++)
        {
            for(size_t zmwIdx = 0; zmwIdx < zmwPerBlock; ++zmwIdx, ++signalIdx)
            {
                const auto& signal = generator->GenerateSignal(numFramesUpper, signalIdx);
                for (size_t chunk = 0, frame = 0; frame < numFramesUpper; ++chunk)
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
                                         const SimConfig &sim,
                                         DataSource::DataSourceBase::Configuration cfg,
                                         std::unique_ptr<SignalGenerator> generator)
    : DataSource::DataSourceBase(std::move(cfg)),
      cache_(std::make_unique<DataCache>(
          sim.NumSignals(), sim.NumFrames(), 
          GetConfig().requestedLayout.BlockWidth(), GetConfig().requestedLayout.NumFrames(), 
          std::move(generator))),
      numZmw_((minZmw + laneSize - 1) / laneSize * laneSize)
{
    const auto& reqLayout = GetConfig().requestedLayout;
    if (reqLayout.BlockWidth() != laneSize)
        PBLOG_WARN << "Unexpected block width requested for SimulatedDataSource. "
                   << "Requested value will be ignored and we will use: " << laneSize;

    std::array<size_t, 3> layoutDims { reqLayout.NumBlocks(), reqLayout.NumFrames(), laneSize};
    PacketLayout nominalLayout(PacketLayout::BLOCK_LAYOUT_DENSE,
                               PacketLayout::INT16,
                               layoutDims);
    const auto numPools = (numZmw_ + nominalLayout.NumZmw() - 1) / nominalLayout.NumZmw();
    for (size_t i = 0; i < numPools-1; ++i)
    {
        layouts_[i] = nominalLayout;
    }

    // Last layout is different
    layoutDims[0] = (numZmw_ - nominalLayout.NumZmw() * (numPools - 1)) / laneSize;
    layouts_[numPools-1] = PacketLayout(PacketLayout::BLOCK_LAYOUT_DENSE,
                            PacketLayout::INT16,
                            layoutDims);

    currChunk_ = SensorPacketsChunk(0, GetConfig().requestedLayout.NumFrames());
    currChunk_.SetZmwRange(0, numZmw_);
}

void SimulatedDataSource::ContinueProcessing()
{
    const auto& layout = layouts_[batchIdx_];
    SensorPacket batchData(layout, batchIdx_, currZmw_,
                           chunkIdx_ * layout.NumFrames(),
                           *GetConfig().allocator);
    for (size_t i = 0; i < layout.NumBlocks(); ++i)
    {
        cache_->FillBlock(currZmw_ / laneSize + i,
                          chunkIdx_,
                          batchData.BlockData(i));
    }
    currChunk_.AddPacket(std::move(batchData));

    batchIdx_++;
    currZmw_ += layout.NumZmw();
    if (currZmw_ == numZmw_)
    {
        batchIdx_ = 0;
        currZmw_ = 0;
        chunkIdx_++;
        this->PushChunk(std::move(currChunk_));
        currChunk_ = SensorPacketsChunk(chunkIdx_ * layout.NumFrames(),
                                        (chunkIdx_ + 1) * layout.NumFrames());
        currChunk_.SetZmwRange(0, numZmw_);
    }
    if (currZmw_ > numZmw_)
        throw PBException("Bookkeeping error in SimulatedDataSource!");

    if (currChunk_.StartFrame() >= GetConfig().numFrames)
    {
        SetDone();
    }
}

std::vector<int16_t> SawtoothGenerator::GenerateSignal(size_t numFrames, size_t idx)
{
    std::vector<int16_t> ret(numFrames);

    const auto range = config_.maxAmp - config_.minAmp + 1;
    const auto slope = static_cast<double>(range) / config_.periodFrames;
    for (size_t i = 0; i < numFrames; ++i)
    {
        ret[i] = static_cast<int16_t>(config_.minAmp +
                                      static_cast<int16_t>(slope*(i+idx*config_.startFrameStagger)) % range);
    }

    return ret;
}

std::vector<int16_t> PicketFenceGenerator::GenerateSignal(size_t numFrames, size_t idx)
{
    std::mt19937 gen(config_.seedFunc(idx));

    auto GetBaselineSignal = [&](size_t frames)
    {
        std::vector<short> baselineSignal(frames);
        std::normal_distribution<> dist(config_.baselineSignalLevel, config_.baselineSigma);
        for (auto &b : baselineSignal) b = std::floor(dist(gen));
        return baselineSignal;
    };

    auto GetPulseSignal = [&](size_t frames)
    {
        std::uniform_int_distribution<> rng(0, config_.pulseSignalLevels.size() - 1);
        auto pulseLevel = config_.pulseSignalLevels[rng(gen)] - config_.baselineSignalLevel;
        auto pulseSignal = GetBaselineSignal(frames);
        for (auto &b : pulseSignal) b += pulseLevel;
        return pulseSignal;
    };

    std::exponential_distribution<> expWidth(config_.pulseWidthRate), expIpd(config_.pulseIpdRate);
    auto GetPulseWidth = [&]()
    {
        return config_.generatePoisson ? 
            static_cast<uint16_t>(std::ceil(expWidth(gen))) : config_.pulseWidth;
    };

    auto GetPulseIpd = [&]()
    {
        return (config_.generatePoisson) ? 
            static_cast<uint16_t>(std::ceil(expIpd(gen))) : config_.pulseIpd;
    };

    size_t frame = 0;
    std::vector<int16_t> signal; signal.reserve(numFrames);
    while (frame < numFrames)
    {
        // Generate IPD followed by PW
        uint16_t nIpdFrames = std::min<size_t>(numFrames - frame, GetPulseIpd());
        std::vector<short> baselineSignal = GetBaselineSignal(nIpdFrames);
        signal.insert(signal.end(), baselineSignal.begin(), baselineSignal.end());
        frame += nIpdFrames;

        uint16_t nPwFrames = std::min<size_t>(numFrames - frame, GetPulseWidth());
        std::vector<short> pulseSignal = GetPulseSignal(nPwFrames);
        signal.insert(signal.end(), pulseSignal.begin(), pulseSignal.end());
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
    std::mt19937 rng(config_.seedFunc(idx));
    auto signal = gen_->GenerateSignal(numFrames, idx);
    std::shuffle(signal.begin(), signal.end(), rng);
    return signal;
}

}} //::PacBio::Application
