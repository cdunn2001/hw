// Copyright (c) 2022, Pacific Biosciences of California, Inc.
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

#include "FrameAnalyzer.h"

#include <cmath>

#include <Eigen/Core>

#include <pacbio/datasource/DataSourceRunner.h>
#include <pacbio/file/FrameStatsFile.h>

using namespace PacBio::DataSource;
using namespace PacBio::File;

namespace PacBio::Calibration {


bool AnalyzeSourceInput(std::unique_ptr<DataSource::DataSourceBase> source,
                        std::shared_ptr<Threading::IThreadController> controller,
                        uint32_t movieNum,
                        std::string outputFile,
                        bool createDarkCalFile)
{
    DataSource::DataSourceRunner runner(std::move(source));

    runner.Start();

    SensorPacketsChunk chunk;
    while (runner.IsActive() && !controller->ExitRequested())
    {
        if(runner.PopChunk(chunk, std::chrono::milliseconds{100}))
        {
            break;
        }
    }
    if (controller->ExitRequested())
    {
        runner.RequestExit();
        return false;
    }

    const auto& stats = AnalyzeChunk(chunk, runner.GetUnitCellProperties());

    assert(stats.mean.shape()[0] == stats.variance.shape()[0]);
    assert(stats.mean.shape()[1] == stats.variance.shape()[1]);

    const uint32_t numRows = stats.mean.shape()[0];
    const uint32_t numCols = stats.mean.shape()[1];
    std::unique_ptr<FrameStatsFile> out;
    if(createDarkCalFile)
    {
        out = std::make_unique<File::CalibrationFile>(outputFile, numRows, numCols);
    }
    else
    {
        out = std::make_unique<File::LoadingFile>(outputFile, numRows, numCols);
    }

    if (controller->ExitRequested())
    {
        return false;
    }
    out->SetFrameMean(stats.mean);

    if (controller->ExitRequested())
    {
        return false;
    }
    out->SetFrameVariance(stats.variance);

    if (controller->ExitRequested())
    {
        return false;
    }
    FrameStatsFile::Attributes attr;
    attr.frameRate = runner.MovieInformation().frameRate;
    attr.movieNum = movieNum;
    attr.numFrames = runner.NumFrames();
    attr.photoelectronSensitivity = runner.MovieInformation().photoelectronSensitivity;
    attr.configWord = 0;
    //TODO missing plumbing to populate these!
    attr.startIndex = 0;
    attr.timeStamp = 0;
    out->SetAttributes(attr);

    return true;
}

// Eigen::Matrix has fortran memory layout by default
// Unsigned one byte data is interpreted as _signed_ int8 to simplify 
// conversion to and comparison with floating point results
typedef Eigen::Matrix<int16_t, Eigen::Dynamic, Eigen::Dynamic> MatrixXs;
typedef Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> MatrixXub;

Eigen::MatrixXf convM2Float(Eigen::Map<const MatrixXs>& map)
{
    return map.cast<float>();
}

Eigen::MatrixXf convM2Float(Eigen::Map<const MatrixXub>& map)
{
    return (map.array() - UINT8_MAX - 1).cast<float>();
}


template<typename M, typename S> 
boost::multi_array<float, 2> CalcChunkMoments(const DataSource::SensorPacketsChunk& chunk)
{
    const auto& layout = (chunk.cbegin())->Layout();
    auto firstChunkZmw = (chunk.cbegin())->StartZmw();
    auto lastChunkZmw  = (chunk.cend() - 1)->StopZmw();
    auto zmwPerChunk   = lastChunkZmw - firstChunkZmw;
    auto framesPerBlock = layout.NumFrames();

    boost::multi_array<float, 2> chunkMoms(boost::extents[2][zmwPerChunk], boost::c_storage_order());

    for (const SensorPacket& batch : chunk)
    {
        // Find packet parameters
        auto batchStartZmw  = batch.StartZmw();
        auto batchEndZmw    = batch.StopZmw();
        auto zmwPerBatch    = batchEndZmw - batchStartZmw;

        // Setup source data
        auto srcDataPtr = reinterpret_cast<const S*>(batch.BlockData(0).Data());
        Eigen::Map<const M> mapBatchView(srcDataPtr, zmwPerBatch, framesPerBlock);
        Eigen::MatrixXf batchMat = convM2Float(mapBatchView);

        // Setup destination data
        auto dstMomPtr = chunkMoms[boost::indices[0][batchStartZmw]].origin();
        Eigen::Map<Eigen::MatrixXf> batchMomView(dstMomPtr, zmwPerBatch, 2);

        // Find moments
        batchMomView.col(0) = batchMat.rowwise().mean().array();
        auto varTmp = (batchMat.colwise() - batchMomView.col(0)).array();
        batchMomView.col(1) = varTmp.square().rowwise().sum().array() / (framesPerBlock - 1);
    }

    return chunkMoms;
}

FrameStats AnalyzeChunk(const DataSource::SensorPacketsChunk& chunk,
                        const std::vector<DataSource::DataSourceBase::UnitCellProperties>& props)
{
    auto firstChunkZmw = (chunk.cbegin())->StartZmw();
    auto lastChunkZmw  = (chunk.cend() - 1)->StopZmw();
    auto zmwPerChunk   = lastChunkZmw - firstChunkZmw;

    auto numZmwX = props[0].x;
    auto numZmwY = props[0].y;
    for (size_t i=0; i < props.size(); i++)
    {
        numZmwX = std::max(numZmwX, props[i].x);
        numZmwY = std::max(numZmwY, props[i].y);
    }
     // Convert indices to amounts ...
    numZmwX++; numZmwY++;
    // ... and ensure their correctness
    assert(size_t(numZmwX*numZmwY) == zmwPerChunk);
    assert(props.size() == zmwPerChunk);

    const auto& layout = (chunk.cbegin())->Layout();
     
    auto chunkMoms = (layout.Encoding() == PacketLayout::INT16) ? 
        CalcChunkMoments<MatrixXs,  MatrixXs::Scalar>(chunk) :
        CalcChunkMoments<MatrixXub, MatrixXub::Scalar>(chunk);

    // Stat moments have to be returned as separate memory blocks
    // They also may be rugged so reshaping is not allowed
    boost::multi_array<float, 2> chunkMean(boost::extents[numZmwX][numZmwY], boost::c_storage_order());
    boost::multi_array<float, 2>  chunkVar(boost::extents[numZmwX][numZmwY], boost::c_storage_order());
    auto outMoms = std::make_tuple(&chunkMean, &chunkVar);

    for (size_t i=0; i < zmwPerChunk; i++)
    {
        (*std::get<0>(outMoms))[props[i].x][props[i].y] = chunkMoms[0][i];
        (*std::get<1>(outMoms))[props[i].x][props[i].y] = chunkMoms[1][i];
    }

    return FrameStats { chunkMean, chunkVar };;
}

} // namespace PacBio::Calibration
