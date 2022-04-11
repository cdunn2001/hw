#include "CoreDMEstimator.h"

#include <cmath>

#include <Eigen/Core>

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

float CoreDMEstimator::shotNoiseCoeff = 1.0;

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
    config_.fallbackBaselineMean = 0.0f;
    config_.fallbackBaselineVariance = 100.0f;

    const float phes = analysisConfig.movieInfo.photoelectronSensitivity;
    assert(std::isfinite(phes) && phes > 0.0f);
    config_.signalScaler = phes;

    auto& psf = analysisConfig.movieInfo.xtalkPsf;
    auto& xtc = analysisConfig.movieInfo.xtalkCorrection;

    if (psf.shape()[0] * psf.shape()[1] == 0)
    {
        shotNoiseCoeff = 1;
    }
    else
    {
        // xtc size is greater or equal to psf
        auto i = (xtc.shape()[0] - psf.shape()[0]) / 2;
        auto j = (xtc.shape()[1] - psf.shape()[1]) / 2;
        typedef Eigen::Map<const Eigen::MatrixXf, Eigen::Unaligned, Eigen::OuterStride<>> StridedMapMatrixXi;
        StridedMapMatrixXi xtcMat(xtc[boost::indices[i][j]].origin(), psf.shape()[1], psf.shape()[0], Eigen::OuterStride<>(xtc.shape()[1]));
        Eigen::Map<const Eigen::MatrixXf> psfMat(psf.origin(),        psf.shape()[1], psf.shape()[0]);

        shotNoiseCoeff = xtcMat.cwiseProduct(xtcMat).cwiseProduct(psfMat).sum(); // S =  sum(X .^ 2 .* P, "all")
    }

    // shotNoiseCoeff = 1.2171f;
}

}}}     // namespace PacBio::Mongo::Basecaller
