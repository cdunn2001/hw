#include "CoreDMEstimator.h"

#include <cmath>

#include <Eigen/Core>

#include <pacbio/PBException.h>
#include <pacbio/logging/Logger.h>

#include <common/LaneArray.h>
#include <common/StatAccumulator.h>
#include <dataTypes/configs/BasecallerDmeConfig.h>
#include <dataTypes/configs/AnalysisConfig.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

// Static constants
constexpr unsigned short CoreDMEstimator::nModelParams;
constexpr unsigned int CoreDMEstimator::nFramesMin;

// static
PacBio::Logging::PBLogger CoreDMEstimator::logger_ (boost::log::keywords::channel = "DetectionModelEstimator");

// static
CoreDMEstimator::Configuration CoreDMEstimator::config_;

CoreDMEstimator::CoreDMEstimator(uint32_t poolId, unsigned int poolSize)
    : poolId_ (poolId)
    , poolSize_ (poolSize)
{
}

// static
void CoreDMEstimator::Configure(const Data::AnalysisConfig& analysisConfig)
{
    config_.fallbackBaselineMean     = 0.0f;
    config_.fallbackBaselineVariance = 100.0f;
    config_.shotVarCoeff             = 1.0f;

    const float phes = analysisConfig.movieInfo.photoelectronSensitivity;
    assert(std::isfinite(phes) && phes > 0.0f);
    config_.signalScaler = phes;

    auto& psf = analysisConfig.movieInfo.xtalkPsf;
    auto& xtc = analysisConfig.movieInfo.xtalkCorrection;

    // psf and xtc must be two-dimensional boost::multi_arrays.
    assert(psf.num_dimensions() == 2u);
    assert(xtc.num_dimensions() == 2u);

    // We assume that psf is not larger than xtc and that both dimensions of
    // each are odd (i.e., both have a center element).
    for (unsigned i = 0; i < 2u; ++i)
    {
        const auto szPsf = psf.shape()[i];
        const auto szXtc = xtc.shape()[i];
        if (szXtc < szPsf)
        {
            throw PBException("Xtalk correction kernel cannot be smaller than image PSF.");
        }
        if (szPsf % 2u == 0)
        {
            throw PBException("Both dimensions of PSF must be odd.");
        }
        if (szXtc % 2u == 0)
        {
            throw PBException("Both dimensions of xtalk correction kernel must be odd");
        }
    }

    if (psf.num_elements() > 0)
    {
        // xtc size is greater or equal to psf
        const auto i = (xtc.shape()[0] - psf.shape()[0]) / 2;
        const auto j = (xtc.shape()[1] - psf.shape()[1]) / 2;
        typedef Eigen::Map<const Eigen::MatrixXf, Eigen::Unaligned, Eigen::OuterStride<>> StridedMapMatrixXi;
        StridedMapMatrixXi xtcMat(xtc[boost::indices[i][j]].origin(), psf.shape()[1], psf.shape()[0], Eigen::OuterStride<>(xtc.shape()[1]));
        Eigen::Map<const Eigen::MatrixXf> psfMat(psf.origin(),        psf.shape()[1], psf.shape()[0]);

        config_.shotVarCoeff = xtcMat.cwiseProduct(xtcMat).cwiseProduct(psfMat).sum(); // S =  sum(X .^ 2 .* P, "all")
    }

    PBLOG_INFO << "Pulse signal variance shot noise coefficient = "
               << config_.shotVarCoeff << '.';
}

}}}     // namespace PacBio::Mongo::Basecaller
