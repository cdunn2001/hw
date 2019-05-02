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
#include <pacbio/ipc/PoolFactory.h>
#include <pacbio/process/ConfigurationBase.h>
#include <pacbio/utilities/CpuInfo.h>

//#include <pacbio/primary/SequelDefinitions.h>
//#include <pacbio/primary/ipc_config.h>
//#include <pacbio/primary/ChipClass.h>

namespace PacBio {
namespace Primary {


class PrimaryConfig :  public PacBio::Process::ConfigurationObject
{
    ADD_PARAMETER(uint32_t, chunksPerSuperchunk, 16);
    ADD_PARAMETER(uint32_t, framesPerBlock,    1024);
    // 7.9 billion is the number of bases per Sequel chip per hour that we expect to be the maximum worst case
    // rate for estimating disk space. We scale down to seconds per ZMW. This macro value includes
    // chemistry rate as well as ZMW loading.
    ADD_PARAMETER(float,    averageBaseRatePerSecond, 7.9e9/3600/1.1e6);
    ADD_PARAMETER(float,    maxAverageBaseRatePerSecond, 12.0f);
    ADD_ARRAY(std::string,  partitions);
    ADD_PARAMETER(uint32_t, numReadBuffers, 3200 * 3);
    ADD_OBJECT(PacBio::IPC::PoolFactory::PoolConfig, poolConfigAcq);
    ADD_OBJECT(PacBio::IPC::PoolFactory::PoolConfig, poolConfigBw);
//    ADD_ENUM(PacBio::Primary::ChipClass, chipClass, PacBio::Primary::ChipClass::UNKNOWN);
    ADD_ARRAY(std::string,  laserNames);

public:
    /// These are cached values derived from the configuration values.
    /// Accessing the configuration values is a bit expensive, so this
    /// is faster.
    struct
    {
        uint32_t chunksPerSuperchunk = 0;
        uint32_t framesPerBlock = 0;
        size_t numDataTilesPerChunk = 0;
        size_t numDataTilesPerSuperchunk = 0;
        size_t numAllocatedTilesPerSuperchunk = 0;
        size_t framesPerTranche = 0;
        uint32_t blocksPerTranche = 0;
        uint32_t numColors = 0;
        uint32_t numZMWs = 0;
        uint32_t numZMWSperPixelTranche = 0;
        double frameRate = 0;

        // aliases:
        uint32_t&  tilesPerTranche     { chunksPerSuperchunk };
        size_t&    framesPerSuperchunk { framesPerTranche };

        void StreamOut(std::ostream& o)
        {
            o
              << "    chunksPerSuperchunk:" << chunksPerSuperchunk << "\n"
              << "    framesPerBlock:" << framesPerBlock << "\n"
              << "    numDataTilesPerSuperchunk:" << numDataTilesPerSuperchunk << "\n"
              << "    numAllocatedTilesPerSuperchunk:" << numAllocatedTilesPerSuperchunk << "\n"
              << "    framesPerTranche:" << framesPerTranche << "\n"
              << "    blocksPerTranche:" << blocksPerTranche << std::endl;
        }
        Json::Value Json() const
        {
            Json::Value v;
            v["chunksPerSuperchunk"] = chunksPerSuperchunk;
            v["framesPerBlock"] = framesPerBlock;
            v["numDataTilesPerSuperchunk"] = numDataTilesPerSuperchunk;
            v["numAllocatedTilesPerSuperchunk"] = numAllocatedTilesPerSuperchunk;
            v["framesPerTranche"] = framesPerTranche;
            v["blocksPerTranche"] = blocksPerTranche;
            return v;
        }
    } cache;


    PrimaryConfig()
    {
        constructing_ = true;
        {
            PacBio::Utilities::Finally final([this](){constructing_ = false;});

            //partitions.resize(2);
            partitions[0] = std::string("/data/pa");
            partitions[1] = std::string("/data/pb");

//            PacBio::Primary::ChipClass autoClass = PacBio::Primary::ChipClass::UNKNOWN;

//            // fixme. this is is super gross
//            if (chipClass() == PacBio::Primary::ChipClass::Benchy)
//            {
//                PBLOG_NOTICE << "PrimaryConfig - BENCHY!";
//                poolConfigAcq.poolType = PacBio::IPC::PoolType::SHM_XSI;
//                poolConfigBw.poolType = PacBio::IPC::PoolType::SHM_XSI;
//            }
//            else
//            {

//                bool avx512f = PacBio::Utilities::Sysinfo::SupportsAVX512f();
//                bool mic = PacBio::Utilities::Sysinfo::SupportsIntelMIC();

//                if (!avx512f && mic)
//                {
//                    poolConfigAcq.poolType = PacBio::IPC::PoolType::SCIF;
//                    poolConfigBw.poolType = PacBio::IPC::PoolType::SCIF;
//                    autoClass = PacBio::Primary::ChipClass::Sequel;
//                }
//                else if (avx512f && !mic)
//                {
//                    poolConfigAcq.poolType = PacBio::IPC::PoolType::SHM_XSI;
//                    poolConfigBw.poolType = PacBio::IPC::PoolType::SHM_XSI;
//                    autoClass = GuessChipClass();
//                }
//                else if (avx512f && mic)
//                {
//                    PBLOG_WARN
//                        << "This platform supports both AVX512f and Intel MIC coprocessors. Assuming MICs are being used.";
//                    poolConfigAcq.poolType = PacBio::IPC::PoolType::SCIF;
//                    poolConfigBw.poolType = PacBio::IPC::PoolType::SCIF;
//                }
//                else
//                {
//                    PBLOG_DEBUG << "This platform does not support AVX512f or Intel MIC coprocessors.";
//                    poolConfigAcq.poolType = PacBio::IPC::PoolType::UNKNOWN;
//                    poolConfigBw.poolType = PacBio::IPC::PoolType::UNKNOWN;
//                }
//            }

//            poolConfigAcq.node = PacBio::IPC::PoolSCIF_Interface::Node::Host;
//            poolConfigAcq.port = PORT_SCIF_ACQUISITION; // 0x11111111
//            poolConfigAcq.poolName = "acq";

//            poolConfigBw.node = PacBio::IPC::PoolSCIF_Interface::Node::Host;
//            poolConfigBw.port = PORT_SCIF_BASEWRITER; // 0x22222222
//            poolConfigBw.poolName = "basewrit"; // limited to 8 characters

//            if (chipClass() == PacBio::Primary::ChipClass::UNKNOWN &&
//                autoClass != PacBio::Primary::ChipClass::UNKNOWN)
//            {
//                PBLOG_INFO << "Setting chip class to auto detection value of " << autoClass.toString() << " was:" << chipClass().toString();
//                chipClass = autoClass;
//            }
//            else
//            {
//                PBLOG_WARN << "NOT SETTING CHIP CLASS TO AUTO DETECTION VALUE OF " << autoClass.toString();
//            }

            laserNames[0] = "topLaser";
            laserNames[1] = "bottomLaser";
        }
        OnChanged();
    }

    void OnChanged() override
    {
        if (constructing_) return;

        cache.chunksPerSuperchunk = chunksPerSuperchunk;
        cache.framesPerBlock = framesPerBlock;

//        switch(chipClass())
//        {
//        default:
//            PBLOG_DEBUG << "defaulting to Sequel chipClass, " << chipClass().toString() << " not supported in OnChanged()";
//            // allow a fall through here, by design
//        case ChipClass::Sequel:
//            cache.numDataTilesPerChunk = Sequel::numDataTilesPerChunk;
//            cache.numColors = 2;
//            cache.numZMWs   = Sequel::numZMWs;
//            cache.numZMWSperPixelTranche = 16;
//            cache.frameRate = Sequel::defaultFrameRate;
//            break;
//        case ChipClass::Spider:
//            cache.numDataTilesPerChunk = Spider::numDataTilesPerChunk;
//            cache.numColors = 1;
//            cache.numZMWs   = Spider::numZMWs;
//            cache.numZMWSperPixelTranche = 32;
//            cache.frameRate = Spider::defaultFrameRate;
//            break;
//        case ChipClass::Benchy:
//            cache.numDataTilesPerChunk = Benchy::numDataTilesPerChunk;
//            cache.numColors = 1;
//            cache.numZMWs   = Benchy::numZMWs;
//            cache.numZMWSperPixelTranche = 32;
//            cache.frameRate = Benchy::defaultFrameRate;
//            break;
//        }

//        cache.numDataTilesPerSuperchunk = (cache.numDataTilesPerChunk * cache.chunksPerSuperchunk);
//        cache.numAllocatedTilesPerSuperchunk = ((cache.numDataTilesPerChunk + numHeaderTilesPerChunk) *
//                cache.chunksPerSuperchunk);

//        cache.framesPerTranche = framesPerTile * cache.tilesPerTranche;
//        cache.blocksPerTranche = cache.framesPerTranche / cache.framesPerBlock;
    }

//    static ChipClass GuessChipClass()
//    {
//        ChipClass autoClass;
//        PacBio::Utilities::CpuInfo cpuInfo;

//        if (cpuInfo.ModelName() == "Intel(R) Xeon(R) Gold 6130 CPU @ 2.10GHz"     || //e.g. PAC2 pac-54043
//            cpuInfo.ModelName() == "Intel(R) Xeon(R) CPU E5-2680 v3 @ 2.50GHz"    || //e.g. pa-dev01
//            cpuInfo.ModelName() == "Intel(R) Xeon(R) CPU E5-2650 v2 @ 2.60GHz"    || //e.g. pa-poca
//            cpuInfo.ModelName() == "0b/01"                                        || //e.g. MICs which are only on Sequel
//            cpuInfo.ModelName() == "Intel(R) Xeon(R) CPU E5-2643 v3 @ 3.40GHz"     //e.g. CI test machines
//           )
//        {
//            autoClass = PacBio::Primary::ChipClass::Sequel;
//        }
//        else if (cpuInfo.ModelName() == "Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz"      || //e.g. rt-test01
//                 cpuInfo.ModelName() == "Intel(R) Xeon(R) Platinum 8160 CPU @ 2.10GHz"  || //e.g. rt-dev01
//                 cpuInfo.ModelName() == "Intel(R) Xeon(R) Platinum 8180 CPU @ 2.50GHz"
//                )
//        {
//            autoClass = PacBio::Primary::ChipClass::Spider;
//        }
//        else
//        {
//            PBLOG_WARN << "Can't determine ChipClass from CpuInfo ModelName:" + cpuInfo.ModelName();
//            autoClass = PacBio::Primary::ChipClass::UNKNOWN;
//        }
//        return autoClass;
//    }

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
private:
    bool constructing_ = false;
};

inline std::ostream& operator<<(std::ostream& o, PrimaryConfig& config)
{
    config.StreamOut(o);
    return o;
}

inline PrimaryConfig& GetPrimaryConfig()
{
   static PrimaryConfig config;
   return config;
}

#define primaryConfig GetPrimaryConfig()


}}



#endif //SEQUELBASECALLER_PRIMARYCONFIG_H
