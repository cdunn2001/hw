#include <common/DataGenerators/PicketFenceGenerator.h>

#include <random>

namespace PacBio {
namespace Cuda {
namespace Data {

namespace {

// Global variables for use in this file only.  Used to be static
// members of PicketFenceGenerator, but they caused NVCC to choke
// on certain builds because it doesn't seem to know some intrinsics
// that the <random> header ends up using in that case.  Host
// compilation was of course fine.  But for whatever reason, when
// compiling the *device* code in a .cu file, the cuda compiler
// still wants to parse and understand the host side code.  Moving
// these into a .cpp file was the only way to avoid this bug in the
// nvcc compiler while still allowing avx512 compilation for intel
// host code.
std::random_device rd;
std::mt19937 gen(rd());

}

constexpr size_t PicketFenceGenerator::MaxFrames;

PicketFenceGenerator::PicketFenceGenerator(const DataManagerParams& params,
                                           const PicketFenceParams& dataParams)
        : GeneratorBase(params.blockLength,
                        params.laneWidth,
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

void PicketFenceGenerator::PopulateBlock(size_t laneIdx,
                                         size_t blockIdx,
                                         std::vector<int16_t>& v)
{
    auto signal = generatedSignals_[laneIdx % generatedSignals_.size()];
    for (size_t i = 0; i < params_.blockLength; ++i)
    {
        size_t f = i + (blockIdx * params_.numBlocks);
        short val = signal[f % signal.size()];
        auto ptr1 = v.data() + i * params_.laneWidth;
        auto ptr2 = ptr1 + params_.laneWidth;
        std::fill(ptr1, ptr2, val);
    }
}

bool PicketFenceGenerator::ValidateGeneratedSignals() const
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
        short expectedBaselineAndPulseLevel = static_cast<short>(dataParams_.baselineSignalLevel + dataParams_.pulseSignalLevels[0]);
        compareExpected(meanBaselineAndPulseLevel, expectedBaselineAndPulseLevel, dataParams_.baselineSigma, "pulse level");
    }

    float meanIpd = totalBaselineFrames / static_cast<float>(numIpds);
    uint16_t expectedMeanIpd = static_cast<uint16_t>(dataParams_.generatePoisson ? (1/dataParams_.pulseIpdRate) : dataParams_.pulseIpd);
    compareExpected(meanIpd, expectedMeanIpd, 1, "mean IPD");

    float meanPw = totalPulseFrames / static_cast<float>(numPulses);
    uint16_t expectedPulseWidth = static_cast<uint16_t>(dataParams_.generatePoisson ? (1/dataParams_.pulseWidthRate) : dataParams_.pulseWidth);
    compareExpected(meanPw, expectedPulseWidth, 1, "mean PW");

    return true;
}

std::vector<short> PicketFenceGenerator::GetPulseSignal(uint16_t frames) const
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

std::vector<short> PicketFenceGenerator::GetBaselineSignal(uint16_t frames) const
{
    std::normal_distribution<> rng(dataParams_.baselineSignalLevel, dataParams_.baselineSigma);
    std::vector<short> baselineSignal;
    for (size_t f = 0; f < frames; ++f)
    {
        baselineSignal.push_back(std::floor(rng(gen)));
    }
    return baselineSignal;
}

uint16_t PicketFenceGenerator::GetPulseWidth() const
{
    if (dataParams_.generatePoisson)
    {
        std::exponential_distribution<> rng(dataParams_.pulseWidthRate);
        return static_cast<uint16_t>(std::ceil(rng(gen)));
    }
    else
    {
        return dataParams_.pulseWidth;
    }
}

uint16_t PicketFenceGenerator::GetPulseIpd() const
{
    if (dataParams_.generatePoisson)
    {
        std::exponential_distribution<> rng(dataParams_.pulseIpdRate);
        return static_cast<uint16_t>(std::ceil(rng(gen)));
    }
    else
    {
        return dataParams_.pulseIpd;
    }
}

}}}
