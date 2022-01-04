#include <gtest/gtest.h>

#include <algorithm>
#include <stdlib.h>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iostream>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>
#include <zlib.h>
#include <unistd.h>

#include <bazio/FastaEntry.h>
#include <bazio/FastaUtilities.h>

#include <postprimary/controlfinder/ControlFilter.h>

#include "test_data_config.h"

using namespace PacBio::Primary;
using namespace PacBio::Primary::Postprimary;


// Test-data files
const std::string data = std::string(PacBio::PaPpaTestConfig::cmakeCurrentListDir) + "/data";
const std::string CONTROL_FASTA         = data + "/Control.fasta";
const std::string CONTROL_ADAPTER_FASTA = data + "/ControlAdapter.fasta";
const std::string EASY_TRUE_FASTA       = data + "/ControlFilterEasyTrue.fasta";
const std::string EASY_TRUE_2_FASTA     = data + "/ControlFilterEasyTrue2.fasta";
const std::string EASY_FALSE_FASTA      = data + "/ControlFilterEasyFalse.fasta";
const std::string HARD_TRUE_FASTA       = data + "/ControlFilterHardTrue.fasta";
const std::string HARD_FALSE_FASTA      = data + "/ControlFilterHardFalse.fasta";
const std::string SPLIT_EASY_FASTA      = data + "/SplitControlFilterEasyTrue.fasta";
    
// Read the control sequence(s) into memory and instantiate our filter object
std::string controlFasta(CONTROL_FASTA);
std::string controlAdapterFasta(CONTROL_ADAPTER_FASTA);
auto controls = std::make_shared<std::vector<FastaEntry>>(
        FastaUtilities::ParseSingleFastaFile(controlFasta));
auto controlAdapters = std::make_shared<std::vector<FastaEntry>>(
        FastaUtilities::ParseSingleFastaFile(controlAdapterFasta));
ControlFilter controlFilter(controls, Platform::RSII, controlAdapters, false);
ControlFilter splitControlFilter(controls, Platform::SEQUELII, controlAdapters);

// Test whether the HotStart barcode recognition works at on the ideal
//  case - All HQ region, known good ZMWs with strong hits
TEST(ControlFilter, EasyTrueControls)
{
    const auto READS = FastaUtilities::ParseSingleFastaFile(EASY_TRUE_FASTA);

    for (const auto& read : READS)
    {
        ControlHit hit = controlFilter.AlignToControl(read.sequence);
        ASSERT_TRUE(hit.accuracy > 0);
        ASSERT_TRUE(hit.span > 0);
    }
}

TEST(ControlFilter, EasyTrueControls2)
{
    const auto READS = FastaUtilities::ParseSingleFastaFile(EASY_TRUE_2_FASTA);

    for (const auto& read : READS)
    {
        ControlHit hit = controlFilter.AlignToControl(read.sequence);

        ASSERT_TRUE(hit.accuracy >= 90.0f);
        EXPECT_NEAR(static_cast<double>(hit.span), static_cast<double>(read.Length()), 100);
    }
}

TEST(ControlFilter, EasyNonControls)
{
    const auto READS = FastaUtilities::ParseSingleFastaFile(EASY_FALSE_FASTA);

    for (const auto& read : READS)
    {
        ControlHit hit = controlFilter.AlignToControl(read.sequence);

        ASSERT_TRUE(hit.accuracy == 0);
        ASSERT_TRUE(hit.span == 0);
    }
}

TEST(ControlFilter, HardTrueControls)
{
    const auto READS = FastaUtilities::ParseSingleFastaFile(HARD_TRUE_FASTA);

    for (const auto& read : READS)
    {
        ControlHit hit = controlFilter.AlignToControl(read.sequence);
        ASSERT_TRUE(hit.accuracy > 0);
        ASSERT_TRUE(hit.span > 0);
    }
}

TEST(ControlFilter, HardNonControls)
{
    const auto READS = FastaUtilities::ParseSingleFastaFile(HARD_FALSE_FASTA);

    for (const auto& read : READS)
    {
        ControlHit hit = controlFilter.AlignToControl(read.sequence);
        ASSERT_TRUE(hit.accuracy == 0);
        ASSERT_TRUE(hit.span == 0);
    }
}

TEST(ControlFilter, SplitWorkflow_EasyTrueControls)
{
    const auto READS = FastaUtilities::ParseSingleFastaFile(EASY_TRUE_FASTA);

    for (const auto& read : READS)
    {
        ControlHit hit = splitControlFilter.AlignToControl(read.sequence);
        ASSERT_TRUE(hit.accuracy > 0);
        ASSERT_TRUE(hit.span > 0);
    }
}

TEST(ControlFilter, SplitWorkflow_EasyTrueControls2)
{
    const auto READS = FastaUtilities::ParseSingleFastaFile(EASY_TRUE_2_FASTA);

    for (const auto& read : READS)
    {
        ControlHit hit = splitControlFilter.AlignToControl(read.sequence);

        ASSERT_TRUE(hit.accuracy >= 90.0f);
        EXPECT_NEAR(static_cast<double>(hit.span), static_cast<double>(read.Length()), 100);
    }
}

TEST(ControlFilter, SplitWorkflow_EasyNonControls)
{
    const auto READS = FastaUtilities::ParseSingleFastaFile(EASY_FALSE_FASTA);

    for (const auto& read : READS)
    {
        ControlHit hit = controlFilter.AlignToControl(read.sequence);

        ASSERT_TRUE(hit.accuracy == 0);
        ASSERT_TRUE(hit.span == 0);
    }
}

TEST(ControlFilter, SplitWorkflow_HardTrueControls)
{
    const auto READS = FastaUtilities::ParseSingleFastaFile(HARD_TRUE_FASTA);

    for (const auto& read : READS)
    {
        ControlHit hit = controlFilter.AlignToControl(read.sequence);
        ASSERT_TRUE(hit.accuracy > 0);
        ASSERT_TRUE(hit.span > 0);
    }
}

TEST(ControlFilter, SplitWorkflow_HardNonControls)
{
    const auto READS = FastaUtilities::ParseSingleFastaFile(HARD_FALSE_FASTA);

    for (const auto& read : READS)
    {
        ControlHit hit = controlFilter.AlignToControl(read.sequence);
        ASSERT_TRUE(hit.accuracy == 0);
        ASSERT_TRUE(hit.span == 0);
    }
}

TEST(ControlFilter, SplitWorkflow_Test1)
{
    ControlFilter scFilter(controls, Platform::SEQUELII, controlAdapters, true, 400);

    const auto READS = FastaUtilities::ParseSingleFastaFile(SPLIT_EASY_FASTA);

    for (const auto& read : READS)
    {
        ControlHit hit = scFilter.AlignToControl(read.sequence);
        ASSERT_TRUE(hit.accuracy == 100);
        // first ~9000 and last ~9000 bases shoul match control so that
        // the entire read matches but we get some mismatches.
        ASSERT_TRUE(hit.span == read.sequence.size() - 4);
    }

}

TEST(ControlFilter, SplitWorkflow_Test2)
{
    // Align directly to control but using different split read length.
    ControlFilter scFilter(controls, Platform::SEQUELII, controlAdapters, true, 500);

    const auto READS = FastaUtilities::ParseSingleFastaFile(CONTROL_FASTA);

    for (const auto& read : READS)
    {
        ControlHit hit = scFilter.AlignToControl(read.sequence);
        ASSERT_TRUE(hit.accuracy == 100);
        ASSERT_TRUE(hit.span == read.sequence.size() - 4);
    }
}
