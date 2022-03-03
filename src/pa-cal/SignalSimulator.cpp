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


#include "SignalSimulator.h"

#include <random>
#include <vector>
#include <algorithm>
#include <functional>

#include <boost/multi_array.hpp>

#include <pacbio/datasource/MallocAllocator.h>

#include "PaCalConstants.h"

using namespace PacBio::Calibration;
using namespace PacBio::DataSource;

DataSourceSimulator::DataSourceSimulator(DataSourceBase::Configuration baseConfig, SimInputConfig simConfig)
    : DataSourceBase(std::move(baseConfig))
    , simCfg_(std::move(simConfig))
    , waitTill_(std::chrono::system_clock::now())
{
    auto& cfg = GetConfig();
    if (simCfg_.nRows * simCfg_.nCols % 64 != 0)
        throw PBException("Incorrect chip dimension. Should be 64 based.");

    const auto& reqLayout = cfg.requestedLayout;

    size_t zmwPerBlock    = reqLayout.BlockWidth();
    size_t framesPerBlock = reqLayout.NumFrames();
    size_t numZmw         = RoundUp(simCfg_.nRows * simCfg_.nCols, laneSize);

    if (zmwPerBlock != laneSize) 
        PBLOG_WARN 
        << "Unexpected block width requested for SimulatedDataSource. "
        << "Requested value will be ignored and we will use: " << laneSize;

    std::array<size_t, 3> layoutDims = { reqLayout.NumBlocks(), framesPerBlock, laneSize };
    PacketLayout nominalLayout(PacketLayout::BLOCK_LAYOUT_DENSE, reqLayout.Encoding(), layoutDims);
    const size_t numPools = (numZmw + nominalLayout.NumZmw() - 1) / nominalLayout.NumZmw();
    for (size_t i = 0; i < numPools - 1; ++i)
    {
        layouts_.emplace(std::make_pair(i, nominalLayout));
    }

    layoutDims[0] = (numZmw - nominalLayout.NumZmw() * (numPools - 1)) / laneSize;
    PacketLayout lastLayout(PacketLayout::BLOCK_LAYOUT_DENSE, reqLayout.Encoding(), layoutDims);
    layouts_.emplace(std::make_pair(numPools - 1, lastLayout));

    currChunk_ = SensorPacketsChunk(0, framesPerBlock);
    currChunk_.SetZmwRange(0, numZmw);
}

std::map<uint32_t, PacketLayout> DataSourceSimulator::PacketLayouts() const
{
    return layouts_;
}

std::vector<DataSourceBase::UnitCellProperties> DataSourceSimulator::GetUnitCellProperties() const
{
    DataSourceBase::UnitCellProperties prop = { DataSource::ZmwFeatures::Sequencing, 0, 0, 0 };
    std::vector<DataSourceBase::UnitCellProperties> retProps(NumZmw(), prop);

    for (size_t i = 0; i < retProps.size(); ++i)
    {
        auto coord = Id2Coords(i);
        retProps[i].x = coord.second;
        retProps[i].y = coord.first;
    }

    return retProps;
}

std::vector<uint32_t> DataSourceSimulator::UnitCellIds() const
{
    std::vector<uint32_t> retIds(NumZmw());
    std::iota(retIds.begin(), retIds.end(), 0);
    return retIds;
}

size_t DataSourceSimulator::NumZmw() const
{ 
    return RoundUp(simCfg_.nRows * simCfg_.nCols, laneSize);
};

HardwareInformation DataSourceSimulator::GetHardwareInformation()
{
    HardwareInformation retInfo;
    retInfo.SetShortName("Simulated DataSource");
    return retInfo;
}

boost::multi_array<float, 2> DataSourceSimulator::CrosstalkFilterMatrix() const
{
    return MakeUnityMatrix();
}

boost::multi_array<float, 2> DataSourceSimulator::ImagePsfMatrix() const
{
    return MakeUnityMatrix();
}

MovieInfo DataSourceSimulator::MovieInformation() const
{
    return DataSource::MockMovieInfo();
}

template<typename T>
SensorPacket DataSourceSimulator::GenerateBatch() const
{
    assert(!layouts_.empty());

    const auto& layout  = layouts_.at(batchIdx_); // preserves const modifier
    auto numBlocks      = layout.NumBlocks();
    auto zmwPerBlock    = layout.BlockWidth();
    auto framesPerBlock = layout.NumFrames();

    auto startZmw       = batchIdx_ * layouts_.at(0).NumZmw();
    auto startFrame     = chunkIdx_ * layouts_.at(0).NumFrames();
    SensorPacket batchData(layout, batchIdx_, startZmw, startFrame, *GetConfig().allocator);
    for (size_t i = 0; i < numBlocks; ++i)
    {
        boost::multi_array_ref<T, 2> blkData(
            reinterpret_cast<T*>(batchData.BlockData(i).Data()),
            boost::extents[zmwPerBlock][framesPerBlock],
            boost::fortran_storage_order());

        for (size_t z = 0; z < zmwPerBlock; ++z)
        {
            size_t zmwIdx = startZmw + i * zmwPerBlock + z;
            auto [ mean, std ] = Id2Norm(zmwIdx);
            std::normal_distribution<> dist(mean, std);
            std::generate(blkData[z].begin(), blkData[z].end(), std::bind(dist, gnr_));
        }
    }

    return batchData;
}

void DataSourceSimulator::ContinueProcessing()
{
    assert(!layouts_.empty());

    PacketLayout nominalLayout = layouts_[0];
    auto batch = nominalLayout.Encoding() == PacketLayout::UINT8 ? 
        GenerateBatch<uint8_t>() : GenerateBatch<int16_t>();
    currChunk_.AddPacket(std::move(batch));

    batchIdx_++;
    auto numZmw = NumZmw();
    auto startZmw = (batchIdx_-1)*nominalLayout.NumZmw() + layouts_[batchIdx_-1].NumZmw();
    if (startZmw == numZmw)
    {
        chunkIdx_++;
        batchIdx_ = 0;
        while (std::chrono::system_clock::now() < waitTill_)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(15));
        } 

        this->PushChunk(std::move(currChunk_));

        int delayMs = simCfg_.minInputDelaySeconds * 1000;
        waitTill_ = std::chrono::system_clock::now() + std::chrono::milliseconds(delayMs);

        currChunk_ = SensorPacketsChunk(chunkIdx_ * nominalLayout.NumFrames(),
                                        (chunkIdx_ + 1) * nominalLayout.NumFrames());
        currChunk_.SetZmwRange(0, numZmw);

    }
    if (startZmw > numZmw)
        throw PBException("Bookkeeping error in SimulatedDataSource!");

    auto chunkFrame = currChunk_.StartFrame();
    auto maxFrames  = GetConfig().numFrames;
    if (chunkFrame >= maxFrames)
    {
        SetDone();
    }
}

void DataSourceSimulator::vStart()
{
    int delayMs = simCfg_.minInputDelaySeconds * 1000;
    waitTill_ = std::chrono::system_clock::now() + std::chrono::milliseconds(delayMs);
}
