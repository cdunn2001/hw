#ifndef mongo_basecaller_traceAnalysis_BaselinerParams_H_
#define mongo_basecaller_traceAnalysis_BaselinerParams_H_

#include <cstddef>
#include <initializer_list>
#include <vector>

#include <dataTypes/BasecallerConfig.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

class BaselinerParams
{
public:
    struct Strides
    {
        Strides(std::initializer_list<size_t> list) : data(list) {}
        std::vector<size_t> data;
    };

    struct Widths
    {
        Widths(std::initializer_list<size_t> list) : data(list) {}
        std::vector<size_t> data;
    };

public:
    BaselinerParams(const Strides& s, const Widths& w, float sigma, float mean)
        : strides_(s.data)
        , widths_(w.data)
        , cSigmaBias_(sigma)
        , cMeanBias_(mean)
    { }

    const std::vector<size_t>& Strides() const { return strides_; }
    const std::vector<size_t>& Widths() const { return widths_; }

    float SigmaBias() const { return cSigmaBias_; }
    float MeanBias() const { return cMeanBias_; }

    /// Computes number of frames of latent data this
    /// configuration will require to be stored
    size_t LatentSize() const
    {
        size_t cumStride = 1;
        size_t minSize = 0;
        for (size_t i = 0; i < strides_.size(); ++i)
        {
            cumStride *= strides_[i];
            minSize += widths_[i] * cumStride;
        }
        size_t ret = 1;
        while (ret < minSize)
        {
            ret *= 2;
        }
        return ret;
    }

    /// Computes the full amount of downsampling that will occur after running
    /// the filter at these settings.
    size_t AggregateStride() const
    {
        size_t ret = 1;
        for (const auto& v : strides_) ret *= v;
        return ret;
    }

private:
    std::vector<size_t> strides_;
    std::vector<size_t> widths_;
    float cSigmaBias_;
    float cMeanBias_;

};

BaselinerParams FilterParamsLookup(const Data::BasecallerBaselinerConfig::MethodName& method);

}}}    // namespace PacBio::Mongo::Basecaller

#endif // mongo_basecaller_traceAnalysis_BaselinerParams_H_
