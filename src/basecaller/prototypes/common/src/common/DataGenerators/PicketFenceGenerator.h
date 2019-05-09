#ifndef PICKET_FENCE_GENERATOR_H
#define PICKET_FENCE_GENERATOR_H


#include <common/ZmwDataManager.h>
#include <common/DataGenerators/GeneratorBase.h>
#include <common/cuda/memory/UnifiedCudaArray.h>

#include <vector_types.h>

#include <algorithm>
#include <chrono>
#include <random>
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

class PicketFenceGenerator : public GeneratorBase<short2>
{
public:
    static constexpr size_t MaxFrames = 16384;
    static size_t MaxBlocks(size_t blockLen, size_t numBlocks)
    {
        return std::min(MaxFrames / blockLen, numBlocks);
    }

    PicketFenceGenerator(const DataManagerParams& params, const PicketFenceParams& dataParams)
            : GeneratorBase(params.blockLength,
                            params.gpuLaneWidth,
                            MaxBlocks(params.blockLength, params.numBlocks),
                            dataParams.numSignals)
            , params_(params)
            , dataParams_(dataParams)
    {
        for (size_t i = 0; i < dataParams_.numSignals; ++i)
        {
            std::vector<short> signal;
            std::vector<uint16_t> ipds;
            std::vector<uint16_t> pws;

            size_t numFrames = std::max(params_.blockLength * params_.numBlocks, MaxFrames);
            size_t frame = 0;
            while (frame < numFrames)
            {
                // Generate IPD followed by PW.
                uint16_t pulseIpd = GetPulseIpd();
                uint16_t nIpdFrames = std::min(numFrames - frame, static_cast<size_t>(pulseIpd));
                ipds.push_back(nIpdFrames);
                std::vector<short> baselineSignal = GetBaselineSignal(pulseIpd);
                signal.insert(std::end(signal), std::begin(baselineSignal),
                                      std::begin(baselineSignal) + nIpdFrames);
                frame += nIpdFrames;

                uint16_t pulseWidth = GetPulseWidth();
                uint16_t nPwFrames = std::min(numFrames - frame, static_cast<size_t>(pulseWidth));
                pws.push_back(nPwFrames);
                std::vector<short> pulseSignal = GetPulseSignal(pulseWidth);
                signal.insert(std::end(signal), std::begin(pulseSignal),
                                   std::begin(pulseSignal) + nPwFrames);
                frame += nPwFrames;
            }
            assert(signal.size() == numFrames);

            generateIpds_.push_back(ipds);
            generatedPws_.push_back(pws);
            generatedSignals_.push_back(signal);
        }
    }

private:
    void PopulateBlock(size_t laneIdx,
                       size_t blockIdx,
                       std::vector<short2>& v) override
    {
        auto signal = generatedSignals_[laneIdx % generatedSignals_.size()];
        for (size_t i = 0; i < params_.blockLength; ++i)
        {
            size_t f = i + (blockIdx * params_.numBlocks);
            short tmp = signal[f % signal.size()];
            short2 val = {tmp, tmp};
            auto ptr1 = v.data() + i * params_.gpuLaneWidth;
            auto ptr2 = ptr1 + params_.gpuLaneWidth;
            std::fill(ptr1, ptr2, val);
        }
    }
public:

    bool ValidateGeneratedSignals() const
    {
        int totalBaselineLevel = 0;
        int totalBaselineAndPulseLevel = 0;
        size_t numIpds = 0;
        size_t numPulses = 0;
        size_t totalBaselineFrames = 0;
        size_t totalPulseFrames = 0;

        for (size_t i = 0; i < generatedSignals_.size(); ++i)
        {
            const auto& ipds = generateIpds_[i];
            const auto& pws = generatedPws_[i];
            const auto& signal = generatedSignals_[i];

            size_t currIpd = 0;
            size_t currPw = 0;
            size_t frame = 0;
            while (frame < signal.size())
            {
                short ipd = ipds[currIpd++];
                totalBaselineLevel = std::accumulate(signal.begin() + frame,
                                                      signal.begin() + frame + ipd,
                                                      totalBaselineLevel);
                totalBaselineFrames += ipd;
                numIpds++;
                frame += ipd;

                short pw = pws[currPw++];
                totalBaselineAndPulseLevel = std::accumulate(signal.begin() + frame,
                                                   signal.begin() + frame + pw,
                                                   totalBaselineAndPulseLevel);
                totalPulseFrames += pw;
                numPulses++;
                frame += pw;
            }
        }

        auto compareExpected = [](float val, float expectedVal, float bounds, const std::string& name)
        {
            if (std::abs(val - expectedVal) > bounds)
                throw PBException("Expected " + name + " = " + std::to_string(expectedVal) +
                                  " bounds = " + std::to_string(bounds) +
                                  " actual = " + std::to_string(val));
        };

        // Compare to expected values.
        float meanBaselineLevel = totalBaselineLevel / static_cast<float>(totalBaselineFrames);
        compareExpected(meanBaselineLevel, dataParams_.baselineSignalLevel, dataParams_.baselineSigma, "baseline level");

        if (dataParams_.pulseSignalLevels.size() == 1)
        {
            float meanBaselineAndPulseLevel = totalBaselineAndPulseLevel / static_cast<float>(totalPulseFrames);
            short expectedBaselineAndPulseLevel = dataParams_.baselineSignalLevel + dataParams_.pulseSignalLevels[0];
            compareExpected(meanBaselineAndPulseLevel, expectedBaselineAndPulseLevel, dataParams_.baselineSigma, "pulse level");
        }

        float meanIpd = totalBaselineFrames / static_cast<float>(numIpds);
        uint16_t expectedMeanIpd = dataParams_.generatePoisson ? (1/dataParams_.pulseIpdRate) : dataParams_.pulseIpd;
        compareExpected(meanIpd, expectedMeanIpd, 1, "mean IPD");

        float meanPw = totalPulseFrames / static_cast<float>(numPulses);
        uint16_t expectedPulseWidth = dataParams_.generatePoisson ? (1/dataParams_.pulseWidthRate) : dataParams_.pulseWidth;
        compareExpected(meanPw, expectedPulseWidth, 1, "mean PW");

        return true;
    }

    bool Validate() const
    { return dataParams_.validate; }

private:
    std::vector<short> GetPulseSignal(uint16_t frames) const
    {
        std::uniform_int_distribution<> rng(0, dataParams_.pulseSignalLevels.size() - 1);
        short pulseLevel = dataParams_.pulseSignalLevels[rng(gen)];
        std::vector<short> baselineSignal = GetBaselineSignal(frames);

        std::vector<short> pulseSignal;
        for (size_t f = 0; f < frames; ++f)
        {
            pulseSignal.push_back(baselineSignal[f] + pulseLevel);
        }
        return pulseSignal;
    }

    std::vector<short> GetBaselineSignal(uint16_t frames) const
    {
        std::normal_distribution<> rng(dataParams_.baselineSignalLevel, dataParams_.baselineSigma);
        std::vector<short> baselineSignal;
        for (size_t f = 0; f < frames; ++f)
        {
            baselineSignal.push_back(std::floor(rng(gen)));
        }
        return baselineSignal;
    }

    uint16_t GetPulseWidth() const
    {
        if (dataParams_.generatePoisson)
        {
            std::exponential_distribution<> rng(dataParams_.pulseWidthRate);
            return std::ceil(rng(gen));
        }
        else
        {
            return dataParams_.pulseWidth;
        }
    }

    uint16_t GetPulseIpd() const
    {
        if (dataParams_.generatePoisson)
        {
            std::exponential_distribution<> rng(dataParams_.pulseIpdRate);
            return std::ceil(rng(gen));
        }
        else
        {
            return dataParams_.pulseIpd;
        }
    }

private:
    static std::random_device rd;
    static std::mt19937 gen;

    DataManagerParams params_;
    PicketFenceParams dataParams_;

    std::vector<std::vector<uint16_t>> generateIpds_;
    std::vector<std::vector<uint16_t>> generatedPws_;
    std::vector<std::vector<short>> generatedSignals_;
};


}}}

#endif
