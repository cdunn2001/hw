#include <gtest/gtest.h>
#include <pbbam/RunMetadata.h>

#include "test_data_config.h"

using namespace PacBio::BAM;

const std::string METADATA      = std::string(PacBio::PaPpaTestConfig::cmakeCurrentListDir) + "/data/testcfg.metadata.xml";
const std::string CTRL_METADATA = std::string(PacBio::PaPpaTestConfig::cmakeCurrentListDir) + "/data/testcfg_control.metadata.xml";

TEST(ConfigMetadata, minSnr)
{
    const auto c = RunMetadata::Collection(METADATA);

    ASSERT_DOUBLE_EQ(1.5, c.AutomationParameters().SNRCut());
}

TEST(ConfigMetadata, adapterSequences)
{
    const auto c = RunMetadata::Collection(METADATA);

    ASSERT_TRUE(c.HasTemplatePrepKit());
    const auto& tpk = c.TemplatePrepKit();

    ASSERT_EQ("ATCTCTCTCAACAACAACAACGGAGGAGGAGGAAAAGAGAGAGAT", tpk.LeftAdaptorSequence());
    ASSERT_EQ("ATCTCTCTCAACAACAACAACGGAGGAGGAGGAAAAGAGAGAGAT", tpk.RightAdaptorSequence());
}

TEST(ConfigMetadata, controls)
{
    const auto c = RunMetadata::Collection(CTRL_METADATA);

    ASSERT_TRUE(c.HasControlKit());
    const auto& ck = c.ControlKit();

    ASSERT_EQ("TAGAGAGAGAAAAGGAGGAGGAGGCAACAACAACAACTCTCTCTA", ck.LeftAdapter());
    ASSERT_EQ("TAGAGAGAGAAAAGGAGGAGGAGGCAACAACAACAACTCTCTCTA", ck.RightAdapter());
    ASSERT_EQ(1966, ck.Sequence().size());
}

TEST(ConfigMetadata, hqrfMethod)
{
    const auto c = RunMetadata::Collection(METADATA);

    ASSERT_TRUE(c.AutomationParameters().HasHQRFMethod());
    ASSERT_EQ("SEQUEL", c.AutomationParameters().HQRFMethod());
}
