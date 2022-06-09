#ifndef mongo_basecaller_traceAnalysis_DmeEmHybrid_H_
#define mongo_basecaller_traceAnalysis_DmeEmHybrid_H_

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
//

#include <algorithm>
#include <common/MongoConstants.h>

#include "CoreDMEstimator.h"
#include "DmeEmDevice.h"
#include "DmeEmHost.h"

namespace PacBio {
namespace Mongo {

namespace Basecaller {

/// An implementation that runs both CPU and GPU implementations and compares
/// the results.  Used purely for troubleshooting.
class DmeEmHybrid : public CoreDMEstimator
{
public:     // Static functions
    static void Configure(const Data::BasecallerDmeConfig &dmeConfig,
                          const Data::AnalysisConfig &analysisConfig);

public:
    DmeEmHybrid(uint32_t poolId, unsigned int poolSize);

    PoolDetModel InitDetectionModels(const PoolBaselineStats& blStats) const override;

private:    // Customized implementation
    void EstimateImpl(const PoolHist& hist,
                      const Data::BaselinerMetrics& metrics,
                      PoolDetModel* detModel) const override;

    std::unique_ptr<DmeEmDevice> device_;
    std::unique_ptr<DmeEmHost> host_;
};

}}}     // namespace PacBio::Mongo::Basecaller

#endif // mongo_basecaller_traceAnalysis_DmeEmHybrid_H_
