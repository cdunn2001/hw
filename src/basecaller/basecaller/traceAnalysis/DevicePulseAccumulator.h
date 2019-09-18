//  Copyright (c) 2019, Pacific Biosciences of California, Inc.
//
//  All rights reserved.
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of Pacific Biosciences nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
//  NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
//  THIS LICENSE.  THIS SOFTWARE IS PROVIDED BY PACIFIC BIOSCIENCES AND ITS
//  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//  PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL PACIFIC BIOSCIENCES OR
//  ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
//  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
//  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
//  BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
//  IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
//  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.

#ifndef MONGO_BASECALLER_DEVICE_PULSE_ACCUMULATOR_H
#define MONGO_BASECALLER_DEVICE_PULSE_ACCUMULATOR_H

#include <basecaller/traceAnalysis/PulseAccumulator.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

template <class LabelManager>
class DevicePulseAccumulator : public PulseAccumulator
{
private:     // Types
    struct AccumImpl;

public:     // Static functions
    static void Configure(const Data::MovieConfig& movieConfig, size_t maxCallsPerZmw);
    static void Finalize();

public:
    DevicePulseAccumulator(uint32_t poolId, uint32_t lanesPerPool);

    ~DevicePulseAccumulator() override;

private:    // Customizable implementation
    std::pair<Data::PulseBatch, Data::PulseDetectorMetrics>
    Process(Data::LabelsBatch labels) override;

    std::unique_ptr<AccumImpl> impl_;
};

}}}     // namespace PacBio::Mongo::Basecaller

#endif //MONGO_BASECALLER_DEVICE_PULSE_ACCUMULATOR_H
