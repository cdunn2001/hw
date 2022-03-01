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

#ifndef PACBIO_SIGNAL_SIMULATOR_H
#define PACBIO_SIGNAL_SIMULATOR_H

#include <array>
#include <string>

#include <pacbio/datasource/DataSourceBase.h>
#include <pacbio/datasource/SensorPacket.h>
#include <pacbio/datasource/ZmwFeatures.h>

#include "PaCalConfig.h"


namespace PacBio::Calibration
{

class DataSourceSimulator : public DataSource::DataSourceBase
{
public:
    DataSourceSimulator(DataSourceBase::Configuration cfg, SimInputConfig simCfg);
    virtual ~DataSourceSimulator() {};

public:
    // PacBio::DataSource::DataSourceBase

    // The set of packet layouts that will be used to span a chunk.  For
    // dense layouts the keys should be a contiguous 0-N, though for sparse
    // layouts that does not need to hold.
    std::map<uint32_t, DataSource::PacketLayout> PacketLayouts() const override;

    std::vector<UnitCellProperties> GetUnitCellProperties() const override;
    std::vector<uint32_t> UnitCellIds() const override;

    size_t NumFrames() const override { return numFrames_; }
    size_t NumZmw()    const override; // Kestrel: 4096 x 6144
    int16_t Pedestal() const override { return simCfg_.Pedestal;  };
    double FrameRate() const override { return 100.0;      };

    // Returns "valid" matrices that can be written as-is to a trace file.
    boost::multi_array<float, 2> CrosstalkFilterMatrix() const override;
    boost::multi_array<float, 2> ImagePsfMatrix() const override;

    Sensor::Platform Platform() const override { return Sensor::Platform::DONT_CARE; }
    std::string InstrumentName() const override { return "Sim DataSource"; }
    DataSource::HardwareInformation GetHardwareInformation() override;
    DataSource::MovieInfo MovieInformation() const override;

    void ContinueProcessing() override;

public:
    // Helpers
    static size_t RoundUp(size_t count, size_t batch)
    {
        return (count + batch - 1) / batch * batch;
    }

    DataSource::SensorPacket GenerateBatch();

    std::pair<int16_t, int16_t> Id2Coords(size_t zmwId) const
    { 
        auto lda = simCfg_.nCols;
        return { zmwId / lda, zmwId % lda };
    }

    std::pair<int16_t, int16_t> Id2Norm(size_t zmwId) const
    {
        auto [y, x] = Id2Coords(zmwId);

        int16_t mring = 2048, sring = 128;
        int32_t lda = simCfg_.nCols, off = simCfg_.Pedestal;
        int16_t mean = 10 * std::abs<int16_t>(x - y) % mring;
        int16_t std  = 5 * std::abs<int16_t>(lda - y + x) % sring + 5;
        return { mean + off, std };
    }

private:
    DataSource::SensorPacketsChunk currChunk_;
    std::map<uint32_t, DataSource::PacketLayout> layouts_;
    std::default_random_engine gnr_;

    size_t numFrames_ = 0;
    size_t batchIdx_  = 0;
    size_t chunkIdx_  = 0;

    SimInputConfig simCfg_;
};

} // namespace PacBio::Calibration

#endif // PACBIO_SIGNAL_SIMULATOR_H