// Copyright (c) 2019, Pacific Biosciences of California, Inc.
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

#ifndef PACBIO_MONGO_BASECALLER_HOST_SIMULATED_PULSE_ACCUMULATOR_H_
#define PACBIO_MONGO_BASECALLER_HOST_SIMULATED_PULSE_ACCUMULATOR_H_

#include <dataTypes/Pulse.h>
#include <basecaller/traceAnalysis/PulseAccumulator.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

class HostSimulatedPulseAccumulator : public PulseAccumulator
{
public:     // Static functions
    static void Configure(const Data::BasecallerPulseAccumConfig& pulseConfig);
    static void Finalize();

public:
    HostSimulatedPulseAccumulator(uint32_t poolId);
    ~HostSimulatedPulseAccumulator() override;

private:
    std::pair<Data::PulseBatch, Data::PulseDetectorMetrics>
    Process(Data::LabelsBatch trace) override;

    Data::Pulse GeneratePulse(uint32_t pulseNum);
};

}}} // namespace PacBio::Mongo::Basecaller

#endif // PACBIO_MONGO_BASECALLER_HOST_SIMULATED_PULSE_ACCUMULATOR_H_
