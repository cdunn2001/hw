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

#include <pa-cal/PaCalProgressMessage.h>

using namespace PacBio::DataSource;
using namespace PacBio::File;

namespace PacBio::Calibration {


bool AnalyzeSourceInput(std::unique_ptr<DataSource::DataSourceBase> source,
                        std::shared_ptr<Threading::IThreadController> controller,
                        uint32_t movieNum,
                        std::string outputFile,
                        bool createDarkCalFile,
                        PaCalStageReporter& reporter)
{
    DataSource::DataSourceRunner runner(std::move(source));

    runner.Start();

    PBLOG_INFO << "Waiting for data chunk ... ";
    SensorPacketsChunk chunk;
    while (runner.IsActive() && !controller->ExitRequested())
    {
        if(runner.PopChunk(chunk, std::chrono::milliseconds{100}))
        {
            PBLOG_INFO << "Received data chunk, frames:[" << chunk.StartFrame() << ","
                << chunk.StopFrame() << "). Has full coverage:" << chunk.HasFullPacketCoverage();
            break;
        }
        reporter.Update(0); // keep the heartbeats alive.
    }
    reporter.ForceNextUpdate();
    if (controller->ExitRequested())
    {
        runner.RequestExit();
        return false;
    }

    const auto& stats = AnalyzeChunk(chunk, runner.Pedestal(), runner.GetUnitCellProperties(), reporter);

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
    PBLOG_INFO << "Calibration file " << outputFile << " written and closed.";

    return true;
}

// Eigen::Matrix has fortran memory layout by default
typedef Eigen::Matrix<int16_t, Eigen::Dynamic, Eigen::Dynamic> MatrixXs;
typedef Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> MatrixXub;
typedef Eigen::Map<Eigen::MatrixXf, Eigen::Unaligned, Eigen::OuterStride<Eigen::Dynamic>> StridedMapMatrixXf;

template <typename M>
Eigen::MatrixXf convM2Float(const M& map)
{
    return map.template cast<float>();
}

template<typename M, typename S> 
boost::multi_array<float, 2> CalcChunkMoments(const DataSource::SensorPacketsChunk& chunk, 
                                              int16_t pedestal,
                                              PaCalStageReporter& reporter)
{
    auto zmwPerChunk   = chunk.NumZmws();

    boost::multi_array<float, 2> chunkMoms(boost::extents[2][zmwPerChunk], boost::c_storage_order());

    uint32_t ipacket = 0;
    for (const SensorPacket& packet : chunk)
    {
        ipacket++;
        PBLOG_DEBUG << "CalcChunkMoments of packet " << packet.PacketID() << " of " << chunk.NumPackets();

        // Find packet parameters
        size_t startZmw       = packet.StartZmw();
        PacketLayout pkLayout = packet.Layout();
        size_t numBlocks      = pkLayout.NumBlocks();
        size_t zmwPerBlock    = pkLayout.BlockWidth();
        size_t framesPerBlock = pkLayout.NumFrames();

        // #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < numBlocks; ++i)
        {
            // Setup source data
            SensorPacket::ConstDataView blockView = packet.BlockData(i);
            auto srcDataPtr = reinterpret_cast<const S*>(blockView.Data());
            Eigen::Map<const M> srcBlockMap(srcDataPtr, zmwPerBlock, framesPerBlock);

            // Setup destination data
            size_t blockStartZmw = startZmw + i * zmwPerBlock;
            auto dstMomPtr = chunkMoms[boost::indices[0][blockStartZmw]].origin();
            StridedMapMatrixXf dstMomMap(dstMomPtr, zmwPerBlock, 2, Eigen::OuterStride<Eigen::Dynamic>(zmwPerChunk));

            // Find moments (transpose to traverse frame dimention faster) ...
            Eigen::MatrixXf blockMat = convM2Float(srcBlockMap.transpose());
            auto mom0Tmp = blockMat.colwise().mean();
            auto mom1Tmp = (blockMat.rowwise() - mom0Tmp).array();

            // And store them
            dstMomMap.col(1) = mom1Tmp.square().colwise().sum().array() / (framesPerBlock - 1);
            dstMomMap.col(0) = mom0Tmp.array() - pedestal; // Adjust mean for the offset
        }
        if (ipacket == chunk.NumPackets()) reporter.ForceNextUpdate();
        reporter.Update(1);
    }

    return chunkMoms;
}

FrameStats AnalyzeChunk(const DataSource::SensorPacketsChunk& chunk, int16_t pedestal,
                        const std::vector<DataSource::DataSourceBase::UnitCellProperties>& props,
                        PaCalStageReporter& reporter)
{
    auto zmwPerChunk   = chunk.NumZmws();

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
    if (size_t(numZmwX*numZmwY) != zmwPerChunk)
    {
        throw PBException("numZmwX("+std::to_string(numZmwX) + ")*numZmwY(" + std::to_string(numZmwY) + " != zmwPerChunk(" + std::to_string(zmwPerChunk));
    }
    assert(props.size() == zmwPerChunk);

    const auto& dataType = (chunk.cbegin())->Layout().Encoding();
     
    auto chunkMoms = (dataType == PacketLayout::INT16) ?
        CalcChunkMoments<MatrixXs,  MatrixXs::Scalar>(chunk, pedestal, reporter) :
        CalcChunkMoments<MatrixXub, MatrixXub::Scalar>(chunk, pedestal, reporter);

    // Stat moments have to be returned as separate memory blocks
    // They also may be ragged so reshaping is not an option
    boost::multi_array<float, 2> chunkMom0(boost::extents[numZmwX][numZmwY], boost::c_storage_order());
    boost::multi_array<float, 2> chunkMom1(boost::extents[numZmwX][numZmwY], boost::c_storage_order());
    decltype(chunkMom0)* outMoms[] = { &chunkMom0, &chunkMom1 };

    for (size_t z=0; z < zmwPerChunk; z++)
    {
        (*outMoms[0])[props[z].x][props[z].y] = chunkMoms[0][z];
        (*outMoms[1])[props[z].x][props[z].y] = chunkMoms[1][z];
    }

    return FrameStats { chunkMom0, chunkMom1 };;
}

} // namespace PacBio::Calibration
