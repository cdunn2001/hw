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

#include <pacbio/datasource/DataSourceRunner.h>
#include <pacbio/file/FrameStatsFile.h>

using namespace PacBio::DataSource;
using namespace PacBio::File;

namespace PacBio::Calibration {

bool AnalyzeSourceInput(std::unique_ptr<DataSource::DataSourceBase> source,
                        std::shared_ptr<Threading::IThreadController> controller,
                        uint32_t movieNum,
                        std::string outputFile)
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

    const auto& props = runner.GetUnitCellProperties();

    const auto& stats = AnalyzeChunk(std::move(chunk), props);

    assert(stats.mean.shape()[0] == stats.variance.shape()[0]);
    assert(stats.mean.shape()[1] == stats.variance.shape()[1]);

    const uint32_t numRows = stats.mean.shape()[0];
    const uint32_t numCols = stats.mean.shape()[1];
    File::CalibrationFile out(outputFile, numRows, numCols);

    if (controller->ExitRequested())
    {
        return false;
    }
    out.SetFrameMean(stats.mean);

    if (controller->ExitRequested())
    {
        return false;
    }
    out.SetFrameVariance(stats.variance);

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
    out.SetAttributes(attr);

    return true;
}

FrameStats AnalyzeChunk(DataSource::SensorPacketsChunk chunk,
                        const std::vector<DataSource::DataSourceBase::UnitCellProperties>& props)
{
    // TODO
    boost::multi_array<float, 2> dummy{boost::extents[1][1]};
    return FrameStats{dummy, dummy};
}

} // namespace PacBio::Calibration
