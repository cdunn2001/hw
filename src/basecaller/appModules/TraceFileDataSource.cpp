// Copyright (c) 2020-2021, Pacific Biosciences of California, Inc.
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

#include <appModules/TraceFileDataSource.h>

#include <numeric>

#include <pacbio/datasource/ZmwFeatures.h>
#include <pacbio/logging/Logger.h>


#include <common/MongoConstants.h>
#include <dataTypes/configs/BasecallerAlgorithmConfig.h>

using namespace PacBio::DataSource;
using namespace PacBio::Mongo;

namespace PacBio {
namespace Application {

namespace {

/// Helper function, as several times we'll need to do something
/// like convert frames to chunks, while rouding up to the nearest
/// integral chunk boundary
///
/// \param numerator
/// \param denominator
/// \return numerator/denominotor rounded up to the nearest integer
size_t DivideWithCeil(size_t numerator, size_t denominator)
{
    return (numerator + denominator - 1) / denominator;
}

/// Computes the number of chunks we intend to read from the tracefile.
/// Nominally this is just the minimum of the frames requested compared to the
/// frames in the file, though this is rounded up to the nearest chunk boundary
///
/// \param numTraceFrames   The number of frames actually stored in the trace file
/// \param blockLen         The number of frames per chunk
/// \param framesRequested  The number of frames desired by external code.  A value
///                         of 0 means all frames present in the tracefile
size_t ComputeTraceChunks(size_t numTraceFrames, size_t blockLen, size_t framesRequested)
{
    auto defaultVal = DivideWithCeil(numTraceFrames, blockLen);
    auto chunksRequested = DivideWithCeil(framesRequested, blockLen);
    if (framesRequested != 0 && chunksRequested < defaultVal)
        return chunksRequested;
    else
        return defaultVal;
}

/// \param requestedType  The data type requested by the application
///                       (int16_t or uint8_t or don't care)
/// \param storageType    The actual storage type used in the tracefile
/// \return The actual layout encoding format to use.  This generally
///         is just the requested type, with the caveat that the "Natural"
///         request translates to "whatever is in the trace file", and
///         if the settings may result in an int16_t to uint8_t data
///         truncation, a warning is issued
PacketLayout::EncodingFormat ComputeEncoding(PacBio::Mongo::Data::TraceInputType requestedType,
                                             File::TraceDataType storageType)
{
    PacketLayout::EncodingFormat ret;
    switch (requestedType)
    {
    case PacBio::Mongo::Data::TraceInputType::Natural:
    {
        if (storageType == File::TraceDataType::INT16)
            ret = PacketLayout::INT16;
        else if (storageType == File::TraceDataType::UINT8)
            ret = PacketLayout::UINT8;
        else
            throw PBException("Unexpected request for trace data type");
        break;
    }
    case PacBio::Mongo::Data::TraceInputType::INT16:
    {
        ret = PacketLayout::INT16;
        break;
    }
    case PacBio::Mongo::Data::TraceInputType::UINT8:
    {
        ret = PacketLayout::UINT8;
        break;
    }
    default:
        throw PBException("Unexpected request for trace data type");
    }

    if (storageType == File::TraceDataType::INT16
        && ret == PacketLayout::UINT8)
    {
        PBLOG_WARN << "Trace data is 16 bit but we are configured to produce 8 bit data.  "
                   << "Values will be saturated to [0,255]";
    }

    return ret;
}

/// Generates a list of trace lanes that we will read from the trace file.
/// \param traceFile  The TraceFile we intend to load data from
/// \param whitelist  A list of lanes we explicitly want to load.  If this is empty
///                   then we will load all lanes
/// \param maxLanes   The max number of lanes we intend the data source to produce.
///                   This is to prevent loading and caching more data than we'll
///                   actually use.  A value of 0 indicates that we will load all
///                   lanes present in the traceFile, modulo any whitelist specification
std::vector<uint32_t> SelectedTraceLanes(const File::TraceFile& traceFile,
                                         const std::vector<uint32_t>& whitelist,
                                         size_t maxLanes = 0)
{
    if (whitelist.empty())
    {
        const auto numZmw = traceFile.Traces().HoleNumber().size();
        if (numZmw % laneSize != 0)
            throw PBException("Invalid tracefile, does not contain an integral number of lanes");

        const auto numLanes = numZmw / laneSize;
        if (maxLanes == 0) maxLanes = numLanes;

        std::vector<uint32_t> ret(std::min(numLanes, maxLanes));
        std::iota(ret.begin(), ret.end(), 0);
        return ret;
    }

    std::set<uint32_t> requestedLanes;
    const auto& holeNumbers = traceFile.Traces().HoleNumber();
    for (const auto& w : whitelist)
    {
        auto itr = std::find(holeNumbers.begin(), holeNumbers.end(), w);
        if (itr == holeNumbers.end())
            PBLOG_WARN << "Requested hole number " + std::to_string(w) + " not present in trace file";
        else
            requestedLanes.insert((itr - holeNumbers.begin()) / laneSize);
    }

    std::vector<uint32_t> ret(requestedLanes.begin(), requestedLanes.end());
    if (maxLanes > 0 && maxLanes > ret.size())
        ret.resize(maxLanes);
    return ret;
}

/// Generates a packet layouts that preserve the original groupings stored
/// in the tracefile.  This means that things will be "sparse", where most batches
/// are small compared to the nominally requested lanesPerPook, or even not present
/// entirely
///
/// \param selectedLanes The lane indexes we wish to read from the tracefile
/// \param traceFile     A handle for the actual tracefile, which will be
///                      examined for things like the batchIds
std::map<uint32_t, size_t> ComputeSparsePools(const std::vector<uint32_t>& selectedLanes,
                                              const File::TraceFile& traceFile)
{
    std::map<uint32_t, size_t> ret;
    if (!traceFile.Traces().HasAnalysisBatch())
    {
        PBLOG_WARN << "Running in ReAnalysis mode with a tracefile that does not "
                   << "have an AnalysisBatch dataset, is this a Kestrel tracefile? "
                   << "All ZMW will be placed in batch 0";

        ret[0] = selectedLanes.size();
        return ret;
    }

    const auto& batchIds = traceFile.Traces().AnalysisBatch();
    // The re-analysis code currently relies implicitly on the assumption that
    // data is stored in the tracefile in order of ascending batchID.  This is
    // in fact how it's done right now, and it's even a natural enough things to
    // do that I doubt it will change, but the code in this file shouldn't really
    // rely on that.
    for (const auto lane : selectedLanes)
    {
        // There is a genuine programming bug if this is violated
        assert((lane+1)*laneSize <= batchIds.size());
        const auto zmw = lane * laneSize;
        const auto batchId = batchIds.at(zmw);
        if (!std::all_of(batchIds.begin() + zmw,
                         batchIds.begin() + zmw + laneSize,
                         [&](auto val) { return val == batchId; }))
        {
            throw PBException("Trace file has multiple batchIDs per lane");
        }
        ret[batchId]++;
    }
    return ret;
}

/// Generates a "dense" packet layout map, where all pools have
/// the requested lanes per pool, save perhaps a runt at the end
std::map<uint32_t, size_t> ComputeDensePools(size_t numLanes,
                                             size_t requestedLanesPerPool)
{
    assert(requestedLanesPerPool > 0);

    std::map<uint32_t, size_t> widths;
    uint32_t poolIdx = 0;
    while (numLanes > 0)
    {
        auto poolWidth = std::min(requestedLanesPerPool, numLanes);
        widths[poolIdx] = poolWidth;;
        poolIdx++;
        numLanes -= poolWidth;
    }

    return widths;
}

} // anonymous

TraceFileDataSource::TraceFileDataSource(
        DataSourceBase::Configuration cfg,
        std::string file,
        uint32_t frames,
        uint32_t numZmwLanes,
        bool cache,
        size_t preloadChunks,
        size_t maxQueueSize,
        Mode mode,
        std::vector<uint32_t> zmwWhitelist,
        Mongo::Data::TraceInputType type)
    : BatchDataSource(std::move(cfg))
    , filename_(file)
    , traceFile_(filename_)
    , numTraceZmws_(traceFile_.Traces().NumZmws())
    , numTraceFrames_(traceFile_.Traces().NumFrames())
    , selectedTraceLanes_(SelectedTraceLanes(traceFile_, zmwWhitelist, numZmwLanes))
    , numTraceChunks_(ComputeTraceChunks(numTraceFrames_, BlockLen(), frames))
    , frameRate_(traceFile_.Scan().AcqParams().frameRate)
    , numZmwLanes_(numZmwLanes == 0 ? selectedTraceLanes_.size() : numZmwLanes)
    , numChunks_(frames == 0 ? numTraceChunks_ : DivideWithCeil(frames, BlockLen()))
    , maxQueueSize_(maxQueueSize == 0 ? preloadChunks + 1 : maxQueueSize)
    , cache_(cache)
    , currChunk_(0, BlockLen())
    , mode_(mode)
{
    const auto& config = GetConfig();
    if (config.darkFrame != nullptr && config.darkFrame->darkCalFileName != "")
        throw PBException("Dark frame subtraction not currently supported for trace files");
    if (config.crosstalkFilter != nullptr && config.crosstalkFilter->kernel.shape()[0] != 0)
        throw PBException("Cross talk correction not currently supported for trace files");
    if (config.decimationMask != nullptr)
        throw PBException("Decimation mask not currently supported for trace files");

    const auto storageType = traceFile_.Traces().StorageType();
    const auto encoding = ComputeEncoding(type, storageType);
    bytesPerValue_ = [&]()
    {
        switch(encoding)
        {
        case PacBio::DataSource::PacketLayout::INT16:
            return 2;
        case PacBio::DataSource::PacketLayout::UINT8:
            return 1;
        default:
            throw PBException("Unsupported encoding");
        }
    }();

    if (config.requestedLayout.BlockWidth() != laneSize)
        throw PBException("Unexpected lane width requested");

    bool sparse = mode_ == Mode::Reanalysis;
    const auto layoutType = sparse
        ? PacketLayout::BLOCK_LAYOUT_SPARSE
        : PacketLayout::BLOCK_LAYOUT_DENSE;
    const auto& widths = sparse
        ? ComputeSparsePools(selectedTraceLanes_, traceFile_)
        : ComputeDensePools(numZmwLanes_, config.requestedLayout.NumBlocks());
    for (const auto& kv : widths)
    {
        layouts_[kv.first] = PacketLayout(layoutType,
                                          encoding,
                                          {kv.second,
                                           config.requestedLayout.NumFrames(),
                                           config.requestedLayout.BlockWidth()});
    }

    if (cache_)
    {
        // Cache requested portion of trace file into memory.
        traceDataCache_.resize(boost::extents[numTraceChunks_][selectedTraceLanes_.size()][BlockWidth() * BlockLen() * bytesPerValue_]);
        for (size_t cacheIdx = 0; cacheIdx < selectedTraceLanes_.size(); ++cacheIdx)
        {
            for (size_t traceChunk = 0; traceChunk < numTraceChunks_; traceChunk++)
            {
                auto* ptr = traceDataCache_[traceChunk][cacheIdx].origin();
                switch(encoding)
                {
                case PacBio::DataSource::PacketLayout::INT16:
                    ReadBlockFromTraceFile(selectedTraceLanes_[cacheIdx], traceChunk, reinterpret_cast<int16_t*>(ptr));
                    break;
                case PacBio::DataSource::PacketLayout::UINT8:
                    ReadBlockFromTraceFile(selectedTraceLanes_[cacheIdx], traceChunk, ptr);
                    break;
                default:
                    throw PBException("Unsupported Encoding");
                }
            }
        }
    }
    else
    {
        // Maintain cache of blocks for current active chunk to support replicating in ZMW space.
        traceDataCache_.resize(boost::extents[1][selectedTraceLanes_.size()][BlockWidth()*BlockLen()*bytesPerValue_]);
        laneCurrentChunk_.resize(selectedTraceLanes_.size(), std::numeric_limits<size_t>::max());
    }

    if (preloadChunks != 0) PreloadInputQueue(preloadChunks);
}

void TraceFileDataSource::ContinueProcessing()
{
    if (ChunksReady() >= maxQueueSize_) return;


    size_t currZmw = 0;
    uint32_t wrappedChunkIndex = chunkIndex_ % numTraceChunks_;
    for (const auto& kv : layouts_)
    {
        const auto batchId = kv.first;
        const auto& currLayout = kv.second;

        uint32_t traceStartZmwLane = currZmw / BlockWidth();
        const auto startFrame = chunkIndex_ * BlockLen();
        SensorPacket batchData(currLayout, batchId, currZmw, startFrame, *GetConfig().allocator);

        for (size_t lane = 0; lane < currLayout.NumBlocks(); lane++)
        {
            auto block = batchData.BlockData(lane);
            assert(block.Count() == BlockWidth()*BlockLen()*bytesPerValue_);

            uint32_t wrappedLane = (traceStartZmwLane + lane) % selectedTraceLanes_.size();
            PopulateBlock(wrappedLane, wrappedChunkIndex, block.Data());
        }

        currChunk_.AddPacket(std::move(batchData));
        currZmw += currLayout.NumZmw();
    }

    auto chunk = SensorPacketsChunk(currChunk_.StopFrame(), currChunk_.StopFrame() + BlockLen(), layouts_.size());
    std::swap(chunk, currChunk_);
    this->PushChunk(std::move(chunk));
    chunkIndex_++;

    if (chunkIndex_ == NumChunks())
        this->SetDone();
}

std::vector<uint32_t> TraceFileDataSource::UnitCellIds() const
{

    if (mode_ == Mode::Replication)
    {
        std::vector<uint32_t> unitCellNumbers(numZmwLanes_ * BlockWidth());
        std::iota(unitCellNumbers.begin(), unitCellNumbers.end(), 0);
        return unitCellNumbers;
    }

    assert(numZmwLanes_ == selectedTraceLanes_.size());
    assert(mode_ == Mode::Reanalysis);

    const auto& fullHoleNumbers = traceFile_.Traces().HoleNumber();
    std::vector<uint32_t> unitCellNumbers;
    unitCellNumbers.reserve(numZmwLanes_ * BlockWidth());

    // For each selected lane from the tracefile, grab the corresponding
    // hole numbers stored there
    for (auto lane : selectedTraceLanes_)
    {
        assert((lane+1) * laneSize <= fullHoleNumbers.size());
        auto startItr = fullHoleNumbers.begin() + lane*laneSize;
        auto endItr = startItr + laneSize;
        unitCellNumbers.insert(unitCellNumbers.end(), startItr, endItr);
    }

    assert(unitCellNumbers.size() == NumZmw());
    return unitCellNumbers;
}

std::vector<DataSourceBase::UnitCellProperties> TraceFileDataSource::GetUnitCellProperties() const
{
    const auto numZmw = numZmwLanes_ * BlockWidth();
    std::vector<DataSourceBase::UnitCellProperties> features(numZmw);
    const auto& holexy = traceFile_.Traces().HoleXY();
    const auto& holeType = traceFile_.Traces().HoleType();
    const auto& holeFeaturesMask = [&](){
        if (traceFile_.Traces().HasHoleFeaturesMask())
        {
            return traceFile_.Traces().HoleFeaturesMask();
        }
        else
        {
            PBLOG_WARN << "Trace file does not contain hole features mask dataset"
                       << " setting all ZMWs as Sequencing";
            return std::vector<uint32_t>(traceFile_.Traces().NumZmws(), DataSource::ZmwFeatures::Sequencing);
        }
    }();

    for(uint32_t i = 0; i < numZmw; i++)
    {
        // i is ZMW position in chunk.
        // The chunk is filled with lanes from the tracefile, modulo the size of the trace file. It is
        // possible that the trace file ZMW count is not modulo the BlockWidth. In other words, the final
        // ZMWs could be ragged in the last trace file lane. So to calculate the ZMW position in the trace file,
        // we have to down convert to chunk lanes, then modulo that with the number of tracelanes, then scale back up.
        // Thankfully the blocks are the same size in the chunk as in the trace file.
        const auto chunkLane = i / BlockWidth();
        const auto offset = i % BlockWidth();
        const auto traceLane = chunkLane % selectedTraceLanes_.size();
        const auto traceZmw = selectedTraceLanes_[traceLane] * BlockWidth() + offset; // position in trace file

        if (traceZmw >= holexy.shape()[0])
        {
            throw PBException("internally calculated traceZmw position is larger than trace file dimension");
        }
        features[i].flags = holeFeaturesMask[traceZmw];
        features[i].type = holeType[traceZmw];
        features[i].x = holexy[traceZmw][0];
        features[i].y = holexy[traceZmw][1];
    }

    // If we're replicating the tracefile, we need to re-work the x/y coordinates
    // for two reasons:
    // * If we're re-using the same input ZMW, then the coordinates won't be unique
    // * The input zmw are typically "sparse", and we wan't to re-organize things
    //   a bit to look like a properly filled out 2D chip, so that this mode can
    //   be used better as a WXIPCDataSource replacement in tests.
    //
    // Note: Any lane count should be fine, but if it's a prime (or nearly prime)
    //       number of lanes, then the "chip" is going to end up looking very
    //       tall and skinny!
    if (mode_ == Mode::Replication)
    {
        // Loop to try and find the "most square" layout that we can.  At the least
        // we know we can do `laneSize X numLanes` so we'll start iterating from there
        uint32_t nCols = BlockWidth();
        for (uint32_t tryCols = nCols; tryCols < numZmwLanes_; tryCols += BlockWidth())
        {
            // Not a valid square layout, so skip this one
            if (numZmw % tryCols != 0) continue;
            auto tryRows = numZmw / tryCols;
            // We've transitioned past the midpoint, so there's no point in looking
            // as everything will now be increasingly rectangular
            if (tryRows < tryCols) break;
            // Keep track of our most recent valid guess.  The last valid guess
            // after this loop terminates will be as close to a square layout as
            // we can be.
            nCols = tryCols;
        }
        assert(numZmw % BlockWidth() == 0);
        assert(numZmw % nCols == 0);

        for (uint32_t i = 0; i < features.size(); ++i)
        {
            features[i].x = i % nCols;
            features[i].y = i / nCols;
        }
    }
    return features;
}

DataSource::MovieInfo TraceFileDataSource::MovieInformation() const
{
    DataSource::MovieInfo movieInfo;

    const auto& acqParams = traceFile_.Scan().AcqParams();
    movieInfo.frameRate = acqParams.frameRate;
    movieInfo.photoelectronSensitivity = acqParams.aduGain;

    const auto& chipInfo = traceFile_.Scan().ChipInfo();
    movieInfo.refSnr = chipInfo.analogRefSnr;

    const auto& dyeSet = traceFile_.Scan().DyeSet();
    assert(dyeSet.numAnalog == movieInfo.analogs.size());
    for (size_t i = 0; i < movieInfo.analogs.size(); i++)
    {
        movieInfo.analogs[i].baseLabel = dyeSet.baseMap[i];
        movieInfo.analogs[i].ipd2SlowStepRatio = dyeSet.ipd2SlowStepRatio[i];
        movieInfo.analogs[i].pw2SlowStepRatio = dyeSet.pw2SlowStepRatio[i];
        movieInfo.analogs[i].excessNoiseCV = dyeSet.excessNoiseCV[i];
        movieInfo.analogs[i].interPulseDistance = dyeSet.ipdMean[i];
        movieInfo.analogs[i].pulseWidth = dyeSet.pulseWidthMean[i];
        movieInfo.analogs[i].relAmplitude = dyeSet.relativeAmp[i];
    }

    return movieInfo;
}

void TraceFileDataSource::LoadGroundTruth(Mongo::Data::BasecallerAlgorithmConfig& config) const
{
    auto setBlMeanAndCovar = [&](float& blMean,
                                 float& blCovar,
                                 const std::string& exceptMsg)
    {
        if (traceFile_.IsSimulated())
        {
            const auto groundTruth = traceFile_.GroundTruth();
            blMean = groundTruth.stateMean[0][0];
            blCovar = groundTruth.stateCovariance[0][0];
        } else
        {
            throw PBException(exceptMsg);
        }
    };

    if (config.modelEstimationMode == Mongo::Data::BasecallerAlgorithmConfig::ModelEstimationMode::FixedEstimations)
    {
        setBlMeanAndCovar(config.staticDetModelConfig.baselineMean,
                          config.staticDetModelConfig.baselineVariance,
                          "Requested static pipeline analysis but input trace file is not simulated!");
    }
    else if (config.dmeConfig.Method == Mongo::Data::BasecallerDmeConfig::MethodName::Fixed &&
             config.dmeConfig.SimModel.useSimulatedBaselineParams == true)
    {
        setBlMeanAndCovar(config.dmeConfig.SimModel.baselineMean,
                          config.dmeConfig.SimModel.baselineVar,
                          "Requested fixed DME with baseline params but input trace file is not simulated!");
    }
}

void TraceFileDataSource::PreloadInputQueue(size_t chunks)
{
    size_t numPreload = std::min(chunks, NumChunks());

    if (numPreload > 0)
    {
        PBLOG_INFO << "Preloading input data queue with " + std::to_string(numPreload) + " chunks";
        while (chunkIndex_ < numPreload)
        {
            PBLOG_INFO << "Preloading chunk " << chunkIndex_;
            ContinueProcessing();
        }
        PBLOG_INFO << "Done preloading input queue.";
    }
}

void TraceFileDataSource::PopulateBlock(size_t cacheIdx, size_t traceChunk, uint8_t* data)
{
    if (cache_)
    {
        std::memcpy(data,
                    traceDataCache_[traceChunk][cacheIdx].origin(),
                    BlockLen()*BlockWidth()*bytesPerValue_);
    }
    else
    {
        if (laneCurrentChunk_[cacheIdx] != traceChunk)
        {
            auto* ptr = traceDataCache_[0][cacheIdx].origin();
            switch(layouts_.begin()->second.Encoding())
            {
            case PacBio::DataSource::PacketLayout::INT16:
                ReadBlockFromTraceFile(selectedTraceLanes_[cacheIdx], traceChunk, reinterpret_cast<int16_t*>(ptr));
                break;
            case PacBio::DataSource::PacketLayout::UINT8:
                ReadBlockFromTraceFile(selectedTraceLanes_[cacheIdx], traceChunk, ptr);
                break;
            default:
                throw PBException("Unsupported Encoding");
            }
            laneCurrentChunk_[cacheIdx] = traceChunk;
        }
        std::memcpy(data, traceDataCache_[0][cacheIdx].origin(),
                    BlockWidth()*BlockLen()*bytesPerValue_);
    }
}

template <typename T>
void TraceFileDataSource::ReadBlockFromTraceFile(size_t traceLane, size_t traceChunk, T* data)
{
    size_t nZmwsToRead = std::min(BlockWidth(), numTraceZmws_ - (traceLane*BlockWidth()));
    size_t nFramesToRead = std::min(BlockLen(), numTraceFrames_ - (traceChunk*BlockLen()));
    using range = boost::multi_array_types::extent_range;
    const range zmwRange(traceLane*BlockWidth(), (traceLane*BlockWidth()) + nZmwsToRead);
    const range frameRange(traceChunk*BlockLen(), (traceChunk*BlockLen()) + nFramesToRead);
    boost::multi_array<T,2> d{boost::extents[zmwRange][frameRange]};
    boost::multi_array_ref<T,2> out{data, boost::extents[nFramesToRead][nZmwsToRead]};
    traceFile_.Traces().ReadTraceBlock(d);
    d.reindex(0);
    for (size_t zmw = 0; zmw < nZmwsToRead; zmw++)
    {
        for (size_t frame = 0; frame < nFramesToRead; frame++)
        {
            out[frame][zmw] = d[zmw][frame];
        }
    }
    if (nZmwsToRead*nFramesToRead < BlockWidth()*BlockLen())
    {
        for (size_t frame = nFramesToRead; frame < BlockLen(); frame++)
        {
            for (size_t zmw = nZmwsToRead; zmw < BlockWidth(); zmw++)
            {
                out[frame][zmw] = 0;
            }
        }
    }
}

// I wouldn't be surprised if this gets overhauled in the future.  For a real sensor acquisition, the ROI
// is generaly a list of rectangles, specified in the chips x/y coordinates.  That's very difficult
// to imitate here, since for trace replication the original x/y coordinates don't mean anything, and even
// for re-analysis it would be hard to specify rectangles that are a subset of the original trace collection
// roi.
//
// So for now:
// * TraceReplication accepts a list of vectors with either one or two elements.  The first element is a ZMW *index*
//   (that is 0-N), and the optional second index is a count to select.
// * TraceReanalysis accepts a list of vectors only with a single element.  That single element is to be a ZMW
//   hole number.  Asking for hole numbers not present in the tracefile will result in a warning.
TraceFileDataSource::LaneSelector TraceFileDataSource::SelectedLanesWithinROI(const std::vector<std::vector<int>>& vec) const
{
    if (vec.empty())
    {
        std::vector<LaneIndex> dummy(0);
        return LaneSelector(dummy);
    }

    if (mode_ == Mode::Replication)
    {
        std::set<LaneIndex> selected;
        for (const auto& range : vec)
        {
            if (range.size() == 0 || range.size() > 2)
                throw PBException("Unexpected format for TraceReplication ROI.  "
                                  "The inner most vector should be a single element "
                                  "representing a ZMW, or two values representing a start ZMW and count");

            // always going to enter at least one lane, corresponding to the first element.
            // This first ZMW may be in the middle of a lane, but we'll add the whole lane
            // anyway
            selected.insert(range[0]/laneSize);
            if (range.size() == 2)
            {
                // We've already addd the first lane, now we use some intentional
                // integer arithmetic to get the rest of the lanes, even if the ROI
                // specified doesn't line up with lane boundaries.
                int laneStart = (range[0] + laneSize) / laneSize;
                int laneEnd = (range[0] + range[1] - 1) / laneSize;
                for (int i = laneStart; i <= laneEnd; ++i)
                {
                    selected.insert(i);
                }
            }
        }
        std::vector<LaneIndex> retValues(selected.begin(), selected.end());
        return LaneSelector(retValues);
    }
    else if (mode_ == Mode::Reanalysis)
    {
        std::vector<uint32_t> whitelist;
        whitelist.reserve(vec.size());
        for (const auto& inner : vec)
        {
            if (inner.size() != 1)
                throw PBException("Unexpected format for TraceReanalysis ROI.  "
                                  "The inner most vector should be a single element "
                                  "representing a ZMW hole number");
            whitelist.push_back(inner[0]);
        }
        const auto& lanes = SelectedTraceLanes(traceFile_, whitelist);
        return LaneSelector(lanes);
    }
    throw PBException("Unexpected TraceFileDataSource mode");
}


}}
