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

#include <map>
#include <string>
#include <chrono>

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
    static size_t RoundUp(size_t count, size_t batch)
    {
        return (count + batch - 1) / batch * batch;
    }

    std::pair<int16_t, int16_t> Id2Norm(size_t zmwId) const
    {
        return simCfg_.dataType == 0 ? 
            Id2NormInt16(zmwId) : Id2NormInt8(zmwId);
    }

private:
    // Helpers
    template<typename T>
    DataSource::SensorPacket GenerateBatch() const;

    std::pair<int16_t, int16_t> Id2Coords(size_t zmwId) const
    { 
        auto lda = simCfg_.nCols;
        return { zmwId / lda, zmwId % lda };
    }

    std::pair<int16_t, int16_t> Id2NormInt16(size_t zmwId) const
    {
        int16_t sring = 43;
        int32_t lda = simCfg_.nCols, off = simCfg_.Pedestal;

        auto [y, x] = Id2Coords(zmwId);
        int16_t std  =  2 * std::abs<int16_t>(lda - y + x) % sring + 1;
        int mring = INT16_MAX - 10*std;
        int16_t mean = 3 * std::abs<int16_t>(x - y) % mring + 5*std;
        return { mean + off, std };
    }

    std::pair<int16_t, int16_t> Id2NormInt8(size_t zmwId) const
    {
        int16_t sring = 5;
        int32_t lda = simCfg_.nCols, off = simCfg_.Pedestal;

        auto [y, x] = Id2Coords(zmwId);
        int16_t std  =  2 * std::abs<int16_t>(lda - y + x) % sring + 1;
        int mring = INT8_MAX - 10*std;
        int16_t mean =  std::abs<int16_t>(x - y) % mring + 5*std;
        return { mean + off, std };
    }

private:
    size_t chunkIdx_  = 0;
    size_t batchIdx_  = 0;
    size_t numFrames_ = 0;
    DataSource::SensorPacketsChunk currChunk_;
    std::map<uint32_t, DataSource::PacketLayout> layouts_;

    SimInputConfig simCfg_;

    std::default_random_engine gnr_;
    std::chrono::time_point<std::chrono::system_clock> waitTill_;
};

} // namespace PacBio::Calibration

#endif // PACBIO_SIGNAL_SIMULATOR_H