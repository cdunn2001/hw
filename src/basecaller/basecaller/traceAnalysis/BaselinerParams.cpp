
#include "BaselinerParams.h"

#include <pacbio/logging/Logger.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

BaselinerParams FilterParamsLookup(const Data::BasecallerBaselinerConfig::MethodName& method)
{
    auto logChoice = [](const BaselinerParams& params) {
        std::stringstream ss;
        ss << "Baseline Filter parameters selected:";
        ss << "\n\tStrides: [";
        for (const auto& stride : params.Strides()) ss << stride << " ";
        ss << "]";
        ss << "\n\tWidths: [";
        for (const auto& width : params.Widths()) ss << width << " ";
        ss << "]";
        PBLOG_DEBUG << ss.str();
    };
    
    using Strides = typename BaselinerParams::Strides_t;
    using Widths = typename BaselinerParams::Widths_t;

    // Intentionally not placing a 'default' statement, so we can at least get
    // a compiler warning if this switch is ever incomplete.
    switch (method)
    {
        case Data::BasecallerBaselinerConfig::MethodName::MultiScaleLarge:
        {
            BaselinerParams ret(Strides{1, 2, 8}, Widths{11, 17, 61}, 2.64f, 0.78f);
            logChoice(ret);
            return ret;
        }
        case Data::BasecallerBaselinerConfig::MethodName::MultiScaleMedium:
        {
            BaselinerParams ret(Strides{1, 2, 8}, Widths{9, 17, 31}, 2.41f, 0.69f);
            logChoice(ret);
            return ret;
        }
        case Data::BasecallerBaselinerConfig::MethodName::MultiScaleSmall:
        {
            BaselinerParams ret(Strides{1, 2, 8}, Widths{7, 17, 17}, 2.15f, 0.63f);
            logChoice(ret);
            return ret;
        }
        case Data::BasecallerBaselinerConfig::MethodName::TwoScaleLarge:
        {
            BaselinerParams ret(Strides{2, 8}, Widths{11, 61}, 2.71f, 0.60f);
            logChoice(ret);
            return ret;
        }
        case Data::BasecallerBaselinerConfig::MethodName::TwoScaleMedium:
        {
            BaselinerParams ret(Strides{2, 8}, Widths{9, 31}, 2.44f, 0.50f);
            logChoice(ret);
            return ret;
        }
        case Data::BasecallerBaselinerConfig::MethodName::TwoScaleSmall:
        {
            BaselinerParams ret(Strides{2, 8}, Widths{7, 17}, 2.07f, 0.41f);
            logChoice(ret);
            return ret;
        }
        case Data::BasecallerBaselinerConfig::MethodName::DeviceMultiScale:
        {
            // TODO: Need to fill in parameters.
            BaselinerParams ret(Strides{}, Widths{}, 0, 0);
            logChoice(ret);
            return ret;
        }
        case Data::BasecallerBaselinerConfig::MethodName::NoOp:
            // Should never get here...
            assert(false);
    }

    // Nor here
    assert(false);
    return {Strides{}, Widths{}, 0, 0};
}

}}}     // namespace PacBio::Mongo::Basecaller

