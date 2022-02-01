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

#include <pacbio/datasource/MallocAllocator.h>
#include <pacbio/tracefile/ScanData.h>

#include <common/MongoConstants.h>

namespace PacBio {
namespace Application {

using namespace PacBio::DataSource;
using namespace PacBio::Mongo;

class SimulatedDataSource::DataCache
{
public:

    template <typename T>
    DataCache(const SimulatedDataSource::SimConfig& config,
              size_t zmwPerBlock, size_t framesPerBlock,
              std::unique_ptr<SignalGenerator> generator,
              T* /*dummy to deduce T*/)
        : bytesPerVal_(sizeof(T))
        , pedestal_(generator->Pedestal())
        , simulatorName_("SimulatedDataSource:" + generator->Name())
    {
        static_assert(std::is_same<T, int16_t>::value
                      || std::is_same<T, uint8_t>::value);
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
        data_.resize(boost::extents[numChunks_][numLanes_][framesPerBlock_][zmwPerBlock_*bytesPerVal_]);

        size_t lane = 0;
        size_t outOfRangeCount = 0;
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
                        auto * ptr = reinterpret_cast<T*>(data_[chunk][lane][frameIdx].origin());
                        T val = std::clamp<int16_t>(signal[frame],
                                                    std::numeric_limits<T>::lowest(),
                                                    std::numeric_limits<T>::max());
                        if (val != signal[frame]) outOfRangeCount++;
                        ptr[zmwIdx] = val;
                    }
                }
            }
        }
        if (outOfRangeCount > 0)
            PBLOG_WARN << "SimulatedDataSource saturated " << outOfRangeCount
                       << " values out of " << data_.num_elements() / bytesPerVal_;
    }

    // Takes a block from the cache and uses it to populate a block in a SensorPacket
    void FillBlock(size_t laneIdx, size_t chunkIdx, SensorPacket::DataView data)
    {
        assert(data.Count() == zmwPerBlock_ * framesPerBlock_ * bytesPerVal_);
        auto cacheBlock = data_[chunkIdx % numChunks_][laneIdx % numLanes_].origin();
        memcpy(data.Data(),
               cacheBlock,
               framesPerBlock_*zmwPerBlock_*bytesPerVal_);
    }

    int16_t Pedestal() const { return pedestal_; }

    std::string Name() const { return simulatorName_; }

private:
    size_t bytesPerVal_ = 0;
    size_t numLanes_ = 0;
    size_t numChunks_ = 0;
    size_t zmwPerBlock_ = 0;
    size_t framesPerBlock_ = 0;
    boost::multi_array<uint8_t, 4> data_;

    int16_t pedestal_ = 0;
    std::string simulatorName_ = "SimulatedDataSource";
};

SimulatedDataSource::~SimulatedDataSource() = default;

template <typename Generator>
std::unique_ptr<SimulatedDataSource::DataCache> MakeCache(PacketLayout::EncodingFormat encoding,
                                                          const SimulatedDataSource::SimConfig& sim,
                                                          size_t zmwPerBlock,
                                                          size_t framesPerBlock,
                                                          std::unique_ptr<Generator> generator)
{
    if (encoding == PacketLayout::INT16)
    {
        return std::make_unique<SimulatedDataSource::DataCache>(sim,
                                                                zmwPerBlock,
                                                                framesPerBlock,
                                                                std::move(generator),
                                                                static_cast<int16_t*>(nullptr));
    }
    if (encoding == PacketLayout::UINT8)
    {
        return std::make_unique<SimulatedDataSource::DataCache>(sim,
                                                                zmwPerBlock,
                                                                framesPerBlock,
                                                                std::move(generator),
                                                                static_cast<uint8_t*>(nullptr));
    }
    throw PBException("Unsupported packet encoding");
}

SimulatedDataSource::SimulatedDataSource(size_t minZmw,
                                         const SimConfig &sim,
                                         DataSource::DataSourceBase::Configuration cfg,
                                         std::unique_ptr<SignalGenerator> generator)
    : BatchDataSource(std::move(cfg)),
      cache_(MakeCache(GetConfig().requestedLayout.Encoding(),
                       sim, GetConfig().requestedLayout.BlockWidth(),
                       GetConfig().requestedLayout.NumFrames(), std::move(generator))),
      numZmw_((minZmw + laneSize - 1) / laneSize * laneSize)
{
    const auto& reqLayout = GetConfig().requestedLayout;
    if (reqLayout.BlockWidth() != laneSize)
        PBLOG_WARN << "Unexpected block width requested for SimulatedDataSource. "
                   << "Requested value will be ignored and we will use: " << laneSize;

    std::array<size_t, 3> layoutDims {
        reqLayout.NumBlocks(),
        reqLayout.NumFrames(),
        laneSize};
    PacketLayout nominalLayout(PacketLayout::BLOCK_LAYOUT_DENSE,
                               reqLayout.Encoding(),
                               layoutDims);
    const auto numPools = (numZmw_ + nominalLayout.NumZmw() - 1) / nominalLayout.NumZmw();
    layoutDims[0] = (numZmw_ - nominalLayout.NumZmw() * (numPools - 1)) / laneSize;
    PacketLayout lastLayout(PacketLayout::BLOCK_LAYOUT_DENSE,
                            reqLayout.Encoding(),
                            layoutDims);

    for (size_t i = 0; i < numPools-1; ++i)
    {
        layouts_[i] = nominalLayout;
    }
    layouts_[numPools-1] = lastLayout;

    currChunk_ = SensorPacketsChunk(0, GetConfig().requestedLayout.NumFrames());
    currChunk_.SetZmwRange(0, numZmw_);
}

int16_t SimulatedDataSource::Pedestal() const { return cache_->Pedestal(); }

std::string SimulatedDataSource::InstrumentName() const { return cache_->Name(); }

void SimulatedDataSource::ContinueProcessing()
{
    const auto& layout = layouts_[batchIdx_];
    const auto startFrame = chunkIdx_ * layout.NumFrames();
    SensorPacket batchData(layout, batchIdx_, currZmw_,
                           startFrame,
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

    auto GetBaselineSignal = [&](uint16_t frames)
    {
        std::normal_distribution<> rng(config_.baselineSignalLevel, config_.baselineSigma);
        std::vector<short> baselineSignal;
        for (size_t f = 0; f < frames; ++f)
        {
            baselineSignal.push_back(std::floor(rng(gen)) + config_.pedestal);
        }
        return baselineSignal;
    };

    auto GetPulseSignal = [&](uint16_t frames)
    {
        std::uniform_int_distribution<> rng(0, config_.pulseSignalLevels.size() - 1);
        auto pulseLevel = config_.pulseSignalLevels[rng(gen)] - config_.baselineSignalLevel;
        auto pulseSignal = GetBaselineSignal(frames);
        for (size_t f = 0; f < frames; ++f)
        {
            pulseSignal[f] += pulseLevel;
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

SimulatedDataSource::SimulatedDataSource(size_t minZmw,
                                         const SimConfig& sim,
                                         size_t lanesPerPool,
                                         size_t framesPerChunk,
                                         std::unique_ptr<SignalGenerator> generator)
    : SimulatedDataSource(minZmw,
                          sim,
                          DataSourceBase::Configuration(PacketLayout(PacketLayout::LayoutType::BLOCK_LAYOUT_DENSE,
                                                                     PacketLayout::EncodingFormat::INT16,
                                                                     {lanesPerPool, framesPerChunk, Mongo::laneSize}),
                                                        std::make_unique<MallocAllocator>()),
                          std::move(generator))
{}

}}  // namespace PacBio::Application
