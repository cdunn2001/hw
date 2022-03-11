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

#include <string>

#include <pacbio/datasource/DataSourceBase.h>
#include <common/ThreadController.h>

namespace PacBio::Calibration
{

/// High level functionality, accepting a data source to generate the data and
/// managing all the computation/threading/output concnerns.
///
/// This function will call `RequestExit` on the supplied threading controller
/// when the computation is complete.
///
/// \param source              A DataSourceBase implementation that will provide the input data
/// \param controller          An IThreadController handle, to help handle threading issues.
/// \param outputFile          The destination on disk for the frame computations
/// \param createDarkCalFile   Boolean to indicate if we create a dark cal (or loading cal) file.
///                            The core contents will be the same, but the different files use different
///                            names for the groups/datasets
/// \return                    A bool indicating success or failure.  Failures include returning
///                            before completion if the controler indicates an early termination
///                            is required
bool AnalyzeSourceInput(std::unique_ptr<DataSource::DataSourceBase> source,
                        std::shared_ptr<Threading::IThreadController> controller,
                        uint32_t movieNum,
                        std::string outputFile,
                        bool createDarkCalFile);


struct FrameStats
{
    boost::multi_array<float, 2> mean;
    boost::multi_array<float, 2> variance;
};

// Note: This may need to become a member function, if we ever relax the
//       constraint that we are processing exactly one chunk;
/// Lower level function for computing the mean/variance of a chunk.
///
/// \param chunk  The input data containing a full chunk of full frame data
/// \param props  The UnitCellProperties associated with the data, containing
///               the mapping back from ZMW index to x/y coordinates
/// \return       A FrameStats instance, containing mean and variance for each
///               pixel on the chip.  The boost::multi_arrays are stored using
///               an arr[row][col] convention.
FrameStats AnalyzeChunk(const DataSource::SensorPacketsChunk& chunk, int16_t pedestal,
                        const std::vector<DataSource::DataSourceBase::UnitCellProperties>& props);

} // namespace PacBio::Calibration
