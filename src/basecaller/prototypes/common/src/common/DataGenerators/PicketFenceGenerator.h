#ifndef PICKET_FENCE_GENERATOR_H
#define PICKET_FENCE_GENERATOR_H


#include <common/ZmwDataManager.h>
#include <common/DataGenerators/GeneratorBase.h>

#include <vector_types.h>

#include <algorithm>
#include <chrono>
#include <list>
#include <string>
#include <cassert>
#include <cstddef>
#include <cmath>

namespace PacBio {
namespace Cuda {
namespace Data {

struct PicketFenceParams
{
    static constexpr size_t MaxSignals = 64;

    bool validate = false;
    size_t numSignals = 32;
    uint16_t pulseWidth = 5;
    uint16_t pulseIpd = 10;
    float pulseWidthRate = 0.2f;
    float pulseIpdRate = 0.1f;
    bool generatePoisson = false;
    short baselineSignalLevel = 200;
    short baselineSigma = 20;
    std::vector<short> pulseSignalLevels = { 200 };

    PicketFenceParams& Validate(bool val)               { validate              = val; return *this; }
    PicketFenceParams& NumSignals(size_t val)           { numSignals            = val; return *this; }
    PicketFenceParams& PulseWidth(uint16_t val)         { pulseWidth            = val; return *this; }
    PicketFenceParams& PulseIpd(uint16_t val)           { pulseIpd              = val; return *this; }
    PicketFenceParams& PulseWidthRate(float val)        { pulseWidthRate        = val; return *this; }
    PicketFenceParams& PulseIpdRate(float val)          { pulseIpdRate          = val; return *this; }
    PicketFenceParams& GeneratePoisson(bool val)        { generatePoisson       = val; return *this; }
    PicketFenceParams& BaselineSignalLevel(short val)   { baselineSignalLevel   = val; return *this; }
    PicketFenceParams& BaselineSigma(short val)         { baselineSigma         = val; return *this; }
    PicketFenceParams& PulseSignalLevels(const std::list<std::string>& val)
    {
        std::vector<short> levels;
        for (const auto& s : val)
        {
            try
            {
                levels.push_back(std::stoi(s));
            }
            catch (...)
            {
                throw PBException("Error parsing pulse levels");
            }
        }

        pulseSignalLevels = levels;
        return *this;
    }
};

class PicketFenceGenerator : public GeneratorBase<int16_t>
{
public:
    static constexpr size_t MaxFrames = 16384;
    static size_t MaxBlocks(size_t blockLen, size_t numBlocks)
    {
        return std::min(MaxFrames / blockLen, numBlocks);
    }

    PicketFenceGenerator(const DataManagerParams& params, const PicketFenceParams& dataParams);

private:
    void PopulateBlock(size_t laneIdx,
                       size_t blockIdx,
                       std::vector<int16_t>& v) override;
public:

    bool ValidateGeneratedSignals() const;

    bool Validate() const
    { return dataParams_.validate; }

private:
    std::vector<short> GetPulseSignal(uint16_t frames) const;

    std::vector<short> GetBaselineSignal(uint16_t frames) const;

    uint16_t GetPulseWidth() const;

    uint16_t GetPulseIpd() const;

private:

    DataManagerParams params_;
    PicketFenceParams dataParams_;

    std::vector<std::vector<uint16_t>> generateIpds_;
    std::vector<std::vector<uint16_t>> generatedPws_;
    std::vector<std::vector<short>> generatedSignals_;
};


}}}

#endif
