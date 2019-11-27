#ifndef SEQUELBASECALLER_PRIMARYCONFIG_H
#define SEQUELBASECALLER_PRIMARYCONFIG_H

// Copyright (c) 2016, Pacific Biosciences of California, Inc.
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
//  Description:
/// \brief  Global configuration for the Primary realtime pipeline. These values
///         may be changed at run time.

#include <pacbio/utilities/Finally.h>
#include <pacbio/process/ConfigurationBase.h>
#include <pacbio/utilities/CpuInfo.h>
#include <pacbio/primary/SequelDefinitions.h>

#include <memory>

namespace PacBio {
namespace Primary {


class PrimaryConfig :  public PacBio::Process::ConfigurationObject
{
public:
    PrimaryConfig(double) : PrimaryConfig() {}; // the bogus argument is to keep accidentally doing PrimaryConfig().foo
    // instead of GetPrimaryConfig().foo, which is what everything but unit tests will want to do.

    ADD_PARAMETER(uint32_t, chunksPerSuperchunk, 16);
    ADD_PARAMETER(uint32_t, framesPerBlock,    1024);
    // 7.9 billion is the number of bases per Sequel chip per hour that we expect to be the maximum worst case
    // rate for estimating disk space. We scale down to seconds per ZMW. This macro value includes
    // chemistry rate as well as ZMW loading.
    ADD_PARAMETER(float,    averageBaseRatePerSecond, 7.9e9/3600/1.1e6);
    ADD_PARAMETER(float,    maxAverageBaseRatePerSecond, 12.0f);
    ADD_ARRAY(std::string,  partitions);
    ADD_PARAMETER(uint32_t, numReadBuffers, 3200 * 3);
    ADD_ARRAY(std::string,  laserNames);
    ADD_PARAMETER(uint32_t, framesPerHFMetricBlock, 1024);
    ADD_PARAMETER(uint32_t, framesPerMFMetricBlock, 4096);
    ADD_PARAMETER(bool,     realtimeActivityLabels, false);
    ADD_PARAMETER_NO_DEFAULT(std::string, layoutName);

public:
    /// These are cached values derived from the configuration values.
    /// Accessing the configuration values is a bit expensive, so this
    /// is faster.
    struct
    {
        uint32_t chunksPerSuperchunk = 0;
        uint32_t framesPerBlock = 0;
        size_t framesPerTranche = 0;
        uint32_t blocksPerTranche = 0;
        uint32_t numColors = 0;
        uint32_t numZMWSperPixelTranche = 0;
        double frameRate = 0;
        bool realtimeActivityLabels = false;

        // aliases:
        uint32_t&  tilesPerTranche          { chunksPerSuperchunk };
        size_t&    framesPerSuperchunk      { framesPerTranche };

        void StreamOut(std::ostream& o)
        {
            o
              << "    chunksPerSuperchunk:" << chunksPerSuperchunk << "\n"
              << "    framesPerBlock:" << framesPerBlock << "\n"
              << "    framesPerTranche:" << framesPerTranche << "\n"
              << "    blocksPerTranche:" << blocksPerTranche << std::endl;
        }
        Json::Value Json() const
        {
            Json::Value v;
            v["chunksPerSuperchunk"] = chunksPerSuperchunk;
            v["framesPerBlock"] = framesPerBlock;
            v["framesPerTranche"] = framesPerTranche;
            v["blocksPerTranche"] = blocksPerTranche;
            return v;
        }
    } cache;


private:
    PrimaryConfig()
    {
        constructing_ = true;
        {
            PacBio::Utilities::Finally final([this](){constructing_ = false;});

            // fixme. partition monitoring should be done, or configured by pa-ws.
            partitions[0] = std::string("/data/pa");
            partitions[1] = std::string("/data/pb");

            laserNames[0] = "topLaser";
            laserNames[1] = "bottomLaser";
        }
        OnChanged();
    }

protected:
    void OnChanged() override
    {
        if (constructing_) return;

        cache.chunksPerSuperchunk = chunksPerSuperchunk;
        cache.framesPerBlock = framesPerBlock;
        cache.realtimeActivityLabels = realtimeActivityLabels;

        cache.framesPerTranche = framesPerTile * cache.tilesPerTranche;
        cache.blocksPerTranche = cache.framesPerTranche / cache.framesPerBlock;
    }

public:
    uint64_t NumDataTilesPerSuperchunk(uint64_t numDataTilesPerChunk) const
    {
        return (numDataTilesPerChunk * cache.chunksPerSuperchunk);
    }
    uint64_t NumAllocatedTilesPerSuperchunk(uint64_t numDataTilesPerChunk) const
    {
        return ((numDataTilesPerChunk + numHeaderTilesPerChunk) * cache.chunksPerSuperchunk);
    }

    std::vector<std::string> GetLaserNames() const
    {
        std::vector<std::string> lasers;
        for(const auto& l : laserNames)
        {
            lasers.push_back(l);
        }
        return lasers;
    }

    void StreamOut(std::ostream& o)
    {
        Json::Value v = this->Json();
        v["__cache"] = cache.Json();
        o << v;
    }

    /// Checks that chipClass is set and platform is set, consistently.
    /// Returns true if ok, false if not ok.
    bool Validate(std::string& message) const
    {
        std::stringstream ss;
        bool ok = true;

        message = ss.str();
        return ok;
    }

    /// throws if the instance is not properly configured
    void Validate()
    {
        std::string message;
        if (!Validate(message)) throw PBException(message);
    }

private:
    bool constructing_ = false;
};

inline std::ostream& operator<<(std::ostream& o, PrimaryConfig& config)
{
    config.StreamOut(o);
    return o;
}

inline PrimaryConfig& privateGetPrimaryConfig(bool flag)
{
    static std::unique_ptr<PrimaryConfig> pConfig;
    if (!pConfig || flag) pConfig = std::make_unique<PrimaryConfig>(0.0);
    return *pConfig;
}

inline PrimaryConfig& ResetPrimaryConfig()
{
    return privateGetPrimaryConfig(true);
}

inline PrimaryConfig& GetPrimaryConfig() { return privateGetPrimaryConfig(false); }


}}



#endif //SEQUELBASECALLER_PRIMARYCONFIG_H
