#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <ctime>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <utility>
#include <vector>
#include <zlib.h>

#include <bazio/RegionLabel.h>
#include <bazio/FastaEntry.h>
#include <bazio/FastaUtilities.h>

#include <postprimary/adapterfinder/AdapterLabeler.h>
#include <postprimary/adapterfinder/AdapterCorrector.h>

#include "DefaultAdapter.h"
#include "TestUtils.h"
#include "Timing.h"
#include "test_data_config.h"

using namespace PacBio::Primary;
using namespace PacBio::Primary::Postprimary;

const std::string data = std::string(PacBio::PaPpaTestConfig::cmakeCurrentListDir) + "/data";
const std::string ALIGNER_FASTA = data + "/AdapterAlignerTest.fasta";
const std::string SPACER_FASTA = data + "/AdapterSpacerTest.fasta";
const std::string RUNTIME_FASTA = data + "/SampleSequencesAsym.fasta";
const std::string OVERLAP_FASTA = data + "/AdapterOverlappingTest.fasta";
const ScoringScheme ADAPTER_SCORING(Platform::RSII, ScoringScope::ADAPTER_FINDING);
const ScoringScheme SPIDER_ADAPTER_SCORING(Platform::SEQUELII, ScoringScope::ADAPTER_FINDING);

// Test #1: Does the aligner produce the expected S-W scores for 
//    a known adapter?
TEST(AdapterTest, aligner)
{
    // Create SMRT-bell adapter
    auto adapterList = CreateAdapterlist();
    EXPECT_EQ(adapterList->size(), 1);
    const auto& adapter = adapterList->at(0);
    
    // Get polymerase reads
    const auto polyReads = FastaUtilities::ParseSingleFastaFile(ALIGNER_FASTA);

    const auto& firstRead = polyReads[0];

    // Fill matrix
    const auto matrix = PairwiseAligner::SWComputeMatrix(
        adapter.sequence, firstRead.sequence, ADAPTER_SCORING);

    // Identify last row
    const int32_t M = adapter.Length() + 1;
    const int32_t N = firstRead.Length() + 1;
    const int32_t beginLastRow = (M - 1) * N;

    // Convert the last row of the matrix to String
    std::ostringstream lastRow;
    for (int j = 1; j < N; ++j)
    {
        lastRow << std::setw(4) << matrix[beginLastRow + j] << ", ";
    }
    std::string lastRowStr = lastRow.str();

    // Declare the expected alignment scores
    std::string expectedLastRow(
            "-304, -293, -282, -271, -278, -271, -260, -264, -255, -245, "
            "-234, -240, -229, -235, -224, -231, -235, -242, -231, -220, "
            "-226, -215, -221, -227, -216, -205, -194, -183, -189, -178, "
            "-167, -156, -163, -167, -174, -181, -185, -192, -199, -205, "
            "-194, -183, -172, -161, -150, -139, -128, -117, -121, -110, "
            " -99, -103,  -92,  -81,  -70,  -59,  -48,  -37,  -26,  -15, "  
            "  -4,    7,   18,   29,   40,   51,   62,   73,   84,   95, " 
            " 106,  117,  128,  139,  135,  128,  121,  114,  110,  106, "  
            "  99,   92,   88,   81,   74,   67,   60,   65,   58,   51, "  
            "  44,   37,   36,   29,   22,   15,   14,    7,    0,   -7, " 
            " -14,  -21,  -28,  -35,  -27,  -16,  -23,  -27,  -34,  -41, "  
            " -45,  -52,  -56,  -63,  -67,  -74,  -81,  -85,  -92,  -99, ");

    // Compare
    EXPECT_EQ(lastRowStr, expectedLastRow);
}

// Test #1.5: Does the aligner produce the expected S-W scores for 
//    a known adapter?
TEST(AdapterTest, spideraligner)
{
    // Create SMRT-bell adapter
    auto adapterList = CreateAdapterlist();
    EXPECT_EQ(adapterList->size(), 1);
    const auto& adapter = adapterList->at(0);
    
    // Get polymerase reads
    const auto polyReads = FastaUtilities::ParseSingleFastaFile(ALIGNER_FASTA);

    const auto& firstRead = polyReads[0];

    // Fill matrix
    const auto matrix = PairwiseAligner::SWComputeMatrix(
        adapter.sequence, firstRead.sequence, SPIDER_ADAPTER_SCORING);

    // Identify last row
    const int32_t M = adapter.Length() + 1;
    const int32_t N = firstRead.Length() + 1;
    const int32_t beginLastRow = (M - 1) * N;

    // Convert the last row of the matrix to String
    std::ostringstream lastRow;
    for (int j = 1; j < N; ++j)
    {
        lastRow << std::setw(4) << matrix[beginLastRow + j] << ", ";
    }
    std::string lastRowStr = lastRow.str();

    // Declare the expected alignment scores
    std::string expectedLastRow(
            "-304, -293, -282, -271, -278, -271, -260, -264, -255, -245, "
            "-234, -240, -229, -235, -224, -231, -235, -242, -231, -220, "
            "-226, -215, -221, -227, -216, -205, -194, -183, -189, -178, "
            "-167, -156, -163, -167, -174, -181, -185, -192, -199, -205, "
            "-194, -183, -172, -161, -150, -139, -128, -117, -121, -110, "
            " -99, -103,  -92,  -81,  -70,  -59,  -48,  -37,  -26,  -15, "
            "  -4,    7,   18,   29,   40,   51,   62,   73,   84,   95, "
            " 106,  117,  128,  139,  135,  128,  121,  114,  110,  106, "
            "  99,   92,   88,   81,   74,   67,   60,   65,   58,   51, "
            "  44,   37,   36,   29,   22,   15,   14,    7,    0,   -7, "
            " -14,  -21,  -28,  -35,  -27,  -16,  -23,  -27,  -34,  -41, "
            " -45,  -52,  -56,  -63,  -67,  -74,  -81,  -85,  -92,  -99, ");

    // Compare
    EXPECT_EQ(lastRowStr, expectedLastRow);
}


// Test #2: Does the back-trace produce the expected path / accuracy?
TEST(AdapterTest, backtrace)
{
    // Create SMRT-bell adapter
    auto adapterList = CreateAdapterlist();
    const auto& adapter = adapterList->at(0);
    EXPECT_EQ(adapterList->size(), 1);

    // Get polymerase reads
    const auto polyReads = FastaUtilities::ParseSingleFastaFile(ALIGNER_FASTA);
    const auto& firstRead = polyReads[0];

    // Fill matrix
    const auto matrix = PairwiseAligner::SWComputeMatrix(
        adapter.sequence, firstRead.sequence, ADAPTER_SCORING);

    // Make a cell for the known adapter end-point
    Cell cell(74, 139);

    // Backtrace to fill in the rest of the Cell's state
    PairwiseAligner::SWBacktracking(matrix, cell, firstRead.Length() + 1,
                                    adapter.sequence, firstRead.sequence,
                                    ADAPTER_SCORING);

    // Compare
    EXPECT_EQ(cell.score,          139);
    EXPECT_EQ(cell.jEndPosition,   static_cast<uint32_t>(74));
    EXPECT_EQ(ReadLength(cell),    static_cast<uint32_t>(44));
    EXPECT_EQ(cell.matches,        static_cast<uint32_t>(42));
    EXPECT_EQ(cell.mismatches,     static_cast<uint32_t>(0));
    EXPECT_EQ(cell.insertions,     static_cast<uint32_t>(2));
    EXPECT_EQ(cell.deletions,      static_cast<uint32_t>(3));
    EXPECT_TRUE(std::abs(Accuracy(cell) - 0.886363) < 0.00001);
}

// Test #3 - Does the SpacedSelector find the correct cells?
TEST(AdapterTest, spacer)
{
    // Create SMRT-bell adapter
    auto adapterList = CreateAdapterlist();
    const auto& adapter = adapterList->at(0);
    EXPECT_EQ(adapterList->size(), 1);

    // Get polymerase reads
    const auto polyReads = FastaUtilities::ParseSingleFastaFile(SPACER_FASTA);
    const auto& firstRead = polyReads[0];

    // Fill matrix
    const auto matrix = PairwiseAligner::SWComputeMatrix(
        adapter.sequence, firstRead.sequence, ADAPTER_SCORING);

    const int32_t N = firstRead.Length() + 1;
    const int32_t beginLastRow = adapter.Length() * N;

    // Convert last row to pairs of position/score
    std::vector<Cell> lastRow;
    lastRow.reserve(N);
    for (int j = 0; j < N; ++j)
    {
        // Filter by adapter score
        if (matrix[beginLastRow + j] > 0)
        {
            lastRow.emplace_back(j, matrix[beginLastRow + j]);
        }
    }
    // No need to waste memory
    lastRow.shrink_to_fit();

    // Find possible adapter end positions with minimal spacing
    auto cells = SpacedSelector::FindEndPositions(
        std::move(lastRow), 40);

    // Convert the last row of the matrix to String
    std::ostringstream spacedCells;
    for (const auto& cell : cells)
    {
        spacedCells << "(" << cell.score << " at " << cell.jEndPosition << "), ";
    }
    std::string spacedCellsStr = spacedCells.str();

    // Declare the expected value of the above String
    std::string spacedCellsExpected(
        "(151 at 136), (92 at 809), (128 at 1430), "
        "(119 at 2031), (88 at 2616), (176 at 3285), "
        "(151 at 3901), (147 at 4551), (136 at 5158), "); 

    // Compare
    EXPECT_EQ(spacedCellsStr, spacedCellsExpected);
}

// Test #4 - Do the cells get converted into the correct RegionLabels?
TEST(AdapterTest, labels)
{
    // Create SMRT-bell adapter
    auto adapterList = CreateAdapterlist();
    EXPECT_EQ(adapterList->size(), 1);

    // Create instance of AdapterLabeler with the SMRT-bell adapter
    // and a minimal adapter score
    AdapterLabeler adapterLabeler(adapterList, Platform::RSII);

    // Get polymerase read
    const auto polyReads = FastaUtilities::ParseSingleFastaFile(SPACER_FASTA);
    const auto& firstRead = polyReads[0];

    auto hqReads = ReadHqRegions(firstRead.id);

    // Compute labels
    auto regionLabels = adapterLabeler.Label(firstRead.sequence, hqReads);
    
    // Convert the identified region labels into a string
    std::ostringstream labels;
    for (const auto& label : regionLabels)
    {
        labels << "(" << label.score << " at " << label.begin << "--" << label.end << "), ";
    }
    std::string regionLabelStr = labels.str();

    // Declare the expected region labels
    std::string regionLabelExpected(
        "(151 at 89--136), (92 at 757--809), (128 at 1381--1430), "
        "(119 at 1990--2031), (88 at 2572--2616), (176 at 3239--3285), "
        "(151 at 3854--3901), (147 at 4506--4551), (136 at 5111--5158), ");

    // Compare
    EXPECT_EQ(regionLabelStr, regionLabelExpected);
}

// Test #5 - Run time test
TEST(AdapterTest, runtime)
{
    // Create SMRT-bell adapter
    auto adapterList = CreateAdapterlist();
    std::cerr << adapterList->at(0).id << std::endl;
    std::cerr << adapterList->at(0).sequence << std::endl;
    EXPECT_EQ(adapterList->size(), 1);

    // Create instance of AdapterLabeler with the SMRT-bell adapter
    // and a minimal adapter score
    AdapterLabeler adapterLabeler(adapterList, Platform::RSII);

    // Get polymerase reads
    auto seqs = FastaUtilities::ParseSingleFastaFile(RUNTIME_FASTA);

    // For each polymerase read
    auto timerBefore = std::chrono::high_resolution_clock::now();

    for (const auto& seq : seqs)
    {
        auto hqReads = ReadHqRegions(seq.id);
        // Compute labels
        auto regionLabels = adapterLabeler.Label(seq.sequence, hqReads);
        // Simplified check that there is a label :)
        EXPECT_EQ(regionLabels.size() >= 1, 1);
    }
    auto timerAfter = std::chrono::high_resolution_clock::now();

    auto t = static_cast<int64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            timerAfter - timerBefore).count() / seqs.size());
    std::cerr << "Average time: " << PrintTime(t) << std::endl;
}

// Test #6 - Overlap test
TEST(AdapterTest, overlap)
{
    std::cerr << "Overlap test" << std::endl;
    // Create SMRT-bell adapter
    auto adapterList = CreateAdapterlist();
    EXPECT_EQ(adapterList->size(), 1);

    // Create instance of AdapterLabeler with the SMRT-bell adapter
    // and a minimal adapter score
    AdapterLabeler adapterLabeler(adapterList, Platform::RSII);

    // Get polymerase reads
    auto seqs = FastaUtilities::ParseSingleFastaFile(OVERLAP_FASTA);
    std::cerr << "# " << seqs.size() << std::endl;
    // For each polymerase read
    for (const auto& seq : seqs)
    {
        auto hqReads = ReadHqRegions(seq.id);
        // Compute labels
        auto regionLabels = adapterLabeler.Label(seq.sequence, hqReads);
        for (const auto& label : regionLabels)
            std::cerr << label.begin << "-" << label.end << std::endl;
    }
}

// Test #7 - Asymmetric overlap test
TEST(AdapterTest, AsymOverlap)
{
    const int ZERO = 0;
    const int ONE = 1;

    std::vector<RegionLabel> intervals1;
    std::vector<RegionLabel> intervals2;

    // I
    intervals1.emplace_back(0,100,10,RegionLabelType::ADAPTER);
    intervals2.emplace_back(10,110,5,RegionLabelType::ADAPTER);

    AdapterLabeler::SimplifyOverlappingIntervals(&intervals1, &intervals2);

    EXPECT_EQ(ONE, intervals1.size());
    EXPECT_EQ(ZERO, intervals2.size());

    intervals1.clear();
    intervals2.clear();

    // II
    intervals1.emplace_back(10,110,5,RegionLabelType::ADAPTER);
    intervals2.emplace_back(0,100,10,RegionLabelType::ADAPTER);

    AdapterLabeler::SimplifyOverlappingIntervals(&intervals1, &intervals2);

    EXPECT_EQ(ZERO, intervals1.size());
    EXPECT_EQ(ONE, intervals2.size());

    intervals1.clear();
    intervals2.clear();

    // III
    intervals1.emplace_back(0,100,10,RegionLabelType::ADAPTER);
    intervals2.emplace_back(100,110,5,RegionLabelType::ADAPTER);

    AdapterLabeler::SimplifyOverlappingIntervals(&intervals1, &intervals2);

    EXPECT_EQ(ONE, intervals1.size());
    EXPECT_EQ(ZERO, intervals2.size());
    
    intervals1.clear();
    intervals2.clear();

    // IV
    intervals1.emplace_back(90,100,10,RegionLabelType::ADAPTER);
    intervals2.emplace_back(80,90,5,RegionLabelType::ADAPTER);

    AdapterLabeler::SimplifyOverlappingIntervals(&intervals1, &intervals2);

    EXPECT_EQ(ONE, intervals1.size());
    EXPECT_EQ(ZERO, intervals2.size());
    
    intervals1.clear();
    intervals2.clear();

    // V
    intervals1.emplace_back(90,100,10,RegionLabelType::ADAPTER);
    intervals2.emplace_back(90,100,5,RegionLabelType::ADAPTER);

    AdapterLabeler::SimplifyOverlappingIntervals(&intervals1, &intervals2);

    EXPECT_EQ(ONE, intervals1.size());
    EXPECT_EQ(ZERO, intervals2.size());
    
    intervals1.clear();
    intervals2.clear();

    // VI
    intervals1.emplace_back(0,100,10,RegionLabelType::ADAPTER);
    intervals2.emplace_back(20,40,5,RegionLabelType::ADAPTER);

    AdapterLabeler::SimplifyOverlappingIntervals(&intervals1, &intervals2);

    EXPECT_EQ(ONE, intervals1.size());
    EXPECT_EQ(ZERO, intervals2.size());
    
    intervals1.clear();
    intervals2.clear();

    // VII
    intervals1.emplace_back(20,40,5,RegionLabelType::ADAPTER);
    intervals2.emplace_back(0,100,10,RegionLabelType::ADAPTER);

    AdapterLabeler::SimplifyOverlappingIntervals(&intervals1, &intervals2);

    EXPECT_EQ(ZERO, intervals1.size());
    EXPECT_EQ(ONE, intervals2.size());
    
    intervals1.clear();
    intervals2.clear();

    // VIII
    intervals1.emplace_back(20,40,11,RegionLabelType::ADAPTER);
    intervals2.emplace_back(0,100,10,RegionLabelType::ADAPTER);

    AdapterLabeler::SimplifyOverlappingIntervals(&intervals1, &intervals2);

    EXPECT_EQ(ONE, intervals1.size());
    EXPECT_EQ(ZERO, intervals2.size());
    
    intervals1.clear();
    intervals2.clear();

    // VIII
    intervals1.emplace_back(20,40,11,RegionLabelType::ADAPTER);
    intervals2.emplace_back(0,100,10,RegionLabelType::ADAPTER);

    AdapterLabeler::SimplifyOverlappingIntervals(&intervals1, &intervals2);

    EXPECT_EQ(ONE, intervals1.size());
    EXPECT_EQ(ZERO, intervals2.size());
    
    intervals1.clear();
    intervals2.clear();

    // IX
    // -----     -----     xxxxx     ------
    //         xxx            ----
    intervals1.emplace_back(0,10,10,RegionLabelType::ADAPTER);
    intervals1.emplace_back(20,30,10,RegionLabelType::ADAPTER);
    intervals1.emplace_back(40,50,10,RegionLabelType::ADAPTER); // DOA
    intervals1.emplace_back(60,70,10,RegionLabelType::ADAPTER);
    intervals2.emplace_back(15,20,9,RegionLabelType::ADAPTER); // DOA
    intervals2.emplace_back(45,55,11,RegionLabelType::ADAPTER);

    AdapterLabeler::SimplifyOverlappingIntervals(&intervals1, &intervals2);

    EXPECT_EQ(3, intervals1.size());
    EXPECT_EQ(ONE, intervals2.size());
    
    intervals1.clear();
    intervals2.clear();

    // X
    // ----     -----     xxxxx  xxxxx
    //        xxx            ------
    intervals1.emplace_back(0,10,10,RegionLabelType::ADAPTER);
    intervals1.emplace_back(20,30,10,RegionLabelType::ADAPTER);
    intervals1.emplace_back(40,50,10,RegionLabelType::ADAPTER); // DOA
    intervals1.emplace_back(52,70,10,RegionLabelType::ADAPTER); // DOA
    intervals2.emplace_back(15,20,9,RegionLabelType::ADAPTER); // DOA
    intervals2.emplace_back(45,55,11,RegionLabelType::ADAPTER);

    AdapterLabeler::SimplifyOverlappingIntervals(&intervals1, &intervals2);

    EXPECT_EQ(2, intervals1.size());
    EXPECT_EQ(ONE, intervals2.size());
    
    intervals1.clear();
    intervals2.clear();

    // XI
    // ----     -----     -----     -----     -----
    //        xxx            
    intervals1.emplace_back(0,10,10,RegionLabelType::ADAPTER);
    intervals1.emplace_back(20,30,10,RegionLabelType::ADAPTER);
    intervals1.emplace_back(40,50,10,RegionLabelType::ADAPTER);
    intervals1.emplace_back(60,70,10,RegionLabelType::ADAPTER);
    intervals1.emplace_back(80,90,10,RegionLabelType::ADAPTER);
    intervals2.emplace_back(15,20,9,RegionLabelType::ADAPTER); // DOA

    AdapterLabeler::SimplifyOverlappingIntervals(&intervals1, &intervals2);

    EXPECT_EQ(5, intervals1.size());
    EXPECT_EQ(ZERO, intervals2.size());
    
    intervals1.clear();
    intervals2.clear();

    // XI
    // ----     -----     -----     -----     -----
    //        xxx            
    intervals1.emplace_back(0,10,10,RegionLabelType::ADAPTER);
    intervals1.emplace_back(20,30,10,RegionLabelType::ADAPTER);
    intervals1.emplace_back(40,50,10,RegionLabelType::ADAPTER);
    intervals1.emplace_back(60,70,10,RegionLabelType::ADAPTER);
    intervals1.emplace_back(80,90,10,RegionLabelType::ADAPTER);
    intervals2.emplace_back(15,20,9,RegionLabelType::ADAPTER); // DOA

    AdapterLabeler::SimplifyOverlappingIntervals(&intervals2, &intervals1);

    EXPECT_EQ(5, intervals1.size());
    EXPECT_EQ(ZERO, intervals2.size());
    
    intervals1.clear();
    intervals2.clear();
}

TEST(AdapterTest, LengthDetector)
{
    const auto hqr = RegionLabel{0, 14000, 100, RegionLabelType::HQREGION};

    // Normal usage: no adapters missing
    std::vector<RegionLabel> adapters;
    adapters.emplace_back(1000, 1050, 10, RegionLabelType::ADAPTER);
    adapters.emplace_back(3000, 3050, 10, RegionLabelType::ADAPTER);
    adapters.emplace_back(5000, 5050, 10, RegionLabelType::ADAPTER);
    adapters.emplace_back(7000, 7050, 10, RegionLabelType::ADAPTER);
    adapters.emplace_back(9000, 9050, 10, RegionLabelType::ADAPTER);
    adapters.emplace_back(11000, 11050, 10, RegionLabelType::ADAPTER);
    adapters.emplace_back(13000, 13050, 10, RegionLabelType::ADAPTER);
    std::vector<int> expFlagged{};
    auto obsFlagged = FlagSubreadLengthOutliers(adapters, hqr);
    EXPECT_EQ(expFlagged, obsFlagged);

    // Missed adapter usage: some adapters missing
    adapters.clear();
    //adapters.emplace_back(1000, 1050, 10, RegionLabelType::ADAPTER);
    adapters.emplace_back(3000, 3050, 10, RegionLabelType::ADAPTER);
    adapters.emplace_back(5000, 5050, 10, RegionLabelType::ADAPTER);
    //adapters.emplace_back(7000, 7050, 10, RegionLabelType::ADAPTER);
    adapters.emplace_back(9000, 9050, 10, RegionLabelType::ADAPTER);
    adapters.emplace_back(11000, 11050, 10, RegionLabelType::ADAPTER);
    //adapters.emplace_back(13000, 13050, 10, RegionLabelType::ADAPTER);
    expFlagged = {0, 2, 4};
    obsFlagged = FlagSubreadLengthOutliers(adapters, hqr);
    EXPECT_EQ(expFlagged, obsFlagged);

    // Bimodal usage: correctly called incorporatum or addendum case:
    adapters.clear();
    // Adapter 1
    adapters.emplace_back(1000, 1050, 10, RegionLabelType::ADAPTER);
    // before
    adapters.emplace_back(2800, 2850, 10, RegionLabelType::ADAPTER);
    // Adapter 2
    adapters.emplace_back(3000, 3050, 10, RegionLabelType::ADAPTER);
    // after
    adapters.emplace_back(3200, 3250, 10, RegionLabelType::ADAPTER);
    // Adapter 1
    adapters.emplace_back(5000, 5050, 10, RegionLabelType::ADAPTER);
    // before
    adapters.emplace_back(6800, 6850, 10, RegionLabelType::ADAPTER);
    // Adapter 2
    adapters.emplace_back(7000, 7050, 10, RegionLabelType::ADAPTER);
    // after
    adapters.emplace_back(7200, 7250, 10, RegionLabelType::ADAPTER);
    // Adapter 1
    adapters.emplace_back(9000, 9050, 10, RegionLabelType::ADAPTER);
    // before
    adapters.emplace_back(10800, 10850, 10, RegionLabelType::ADAPTER);
    // Adapter 2
    adapters.emplace_back(11000, 11050, 10, RegionLabelType::ADAPTER);
    // after
    adapters.emplace_back(11200, 11250, 10, RegionLabelType::ADAPTER);
    // Adapter 1
    adapters.emplace_back(13000, 13050, 10, RegionLabelType::ADAPTER);
    // For now we expect all subreads to be flagged, so the RC candidate can be
    // detected (We don't expect to correctly call incorporatum cases prior to
    // FlagSubreadLengthOutliers at this point)
    expFlagged.clear();
    expFlagged.resize(adapters.size() + 1);
    std::iota(expFlagged.begin(), expFlagged.end(), 0);
    obsFlagged = FlagSubreadLengthOutliers(adapters, hqr);
    EXPECT_EQ(expFlagged, obsFlagged);

    // Trimodal usage: incorrectly called incorporatum or addendum case:
    adapters.clear();
    // Adapter 1
    adapters.emplace_back(1000, 1050, 10, RegionLabelType::ADAPTER);
    // before
    adapters.emplace_back(2800, 2850, 10, RegionLabelType::ADAPTER);
    // Adapter 2
    adapters.emplace_back(3000, 3050, 10, RegionLabelType::ADAPTER);
    // after
    //adapters.emplace_back(3200, 3250, 10, RegionLabelType::ADAPTER);
    // Adapter 1
    adapters.emplace_back(5000, 5050, 10, RegionLabelType::ADAPTER);
    // before
    adapters.emplace_back(6800, 6850, 10, RegionLabelType::ADAPTER);
    // Adapter 2
    adapters.emplace_back(7000, 7050, 10, RegionLabelType::ADAPTER);
    // after
    //adapters.emplace_back(7200, 7250, 10, RegionLabelType::ADAPTER);
    // Adapter 1
    adapters.emplace_back(9000, 9050, 10, RegionLabelType::ADAPTER);
    // before
    adapters.emplace_back(10800, 10850, 10, RegionLabelType::ADAPTER);
    // Adapter 2
    adapters.emplace_back(11000, 11050, 10, RegionLabelType::ADAPTER);
    // after
    //adapters.emplace_back(11200, 11250, 10, RegionLabelType::ADAPTER);
    // Adapter 1
    adapters.emplace_back(13000, 13050, 10, RegionLabelType::ADAPTER);
    expFlagged.clear();
    expFlagged.resize(adapters.size() + 1);
    std::iota(expFlagged.begin(), expFlagged.end(), 0);
    obsFlagged = FlagSubreadLengthOutliers(adapters, hqr);
    EXPECT_EQ(expFlagged, obsFlagged);

    // Concatemer usage:  probably a dimer case:
    adapters.clear();
    adapters.emplace_back(100, 150, 10, RegionLabelType::ADAPTER);
    adapters.emplace_back(200, 250, 10, RegionLabelType::ADAPTER);
    adapters.emplace_back(300, 350, 10, RegionLabelType::ADAPTER);
    adapters.emplace_back(400, 450, 10, RegionLabelType::ADAPTER);
    adapters.emplace_back(500, 550, 10, RegionLabelType::ADAPTER);
    adapters.emplace_back(600, 650, 10, RegionLabelType::ADAPTER);
    adapters.emplace_back(700, 750, 10, RegionLabelType::ADAPTER);
    adapters.emplace_back(800, 850, 10, RegionLabelType::ADAPTER);
    adapters.emplace_back(900, 950, 10, RegionLabelType::ADAPTER);
    adapters.emplace_back(1000, 1050, 10, RegionLabelType::ADAPTER);
    adapters.emplace_back(1100, 1150, 10, RegionLabelType::ADAPTER);
    adapters.emplace_back(1200, 1250, 10, RegionLabelType::ADAPTER);
    adapters.emplace_back(1300, 1350, 10, RegionLabelType::ADAPTER);
    expFlagged.clear();
    obsFlagged = FlagSubreadLengthOutliers(adapters, hqr);
    EXPECT_EQ(expFlagged, obsFlagged);
}

TEST(AdapterTest, palindromeTrieSplitter)
{
    int kmerSize = 3;
    int centerSampleSize = 100;
    // Noisy but fairly balanced:
    std::string pali("ACGGACGATATCGCCGTC");
    int hqrOffset = 0;
    int expectedBreakpoint = 8;
    int center;
    std::vector<RegionLabel> adapters;

    const auto& centers = PalindromeTrieCenter(pali, adapters, hqrOffset,
                                               kmerSize, centerSampleSize);
    ASSERT_EQ(centers.size(), 1);
    center = centers[0];
    EXPECT_EQ(expectedBreakpoint, center);

    // At the end but perfect
    //                              ||
    //                              \/
    //                 0123456789012345678
    std::string pali3("GGCTCACGGACGATATCGT");
    expectedBreakpoint = 13;
    const auto& centers3 = PalindromeTrieCenter(pali3, adapters, hqrOffset,
                                                kmerSize, centerSampleSize);
    ASSERT_EQ(centers3.size(), 1);
    center = centers3[0];
    EXPECT_EQ(expectedBreakpoint, center);

    // At the beginning but perfect
    //                    ||
    //                    \/
    //                 0123456789012
    std::string pali4("CGATATCGTCCGT");
    expectedBreakpoint = 3;
    const auto& centers4 = PalindromeTrieCenter(pali4, adapters, hqrOffset,
                                                kmerSize, centerSampleSize);
    ASSERT_EQ(centers4.size(), 1);
    center = centers4[0];
    EXPECT_EQ(expectedBreakpoint, center);

    // alignment broken by burst
    //                                ||
    //                                \/
    //                 0123456789012345678901234
    std::string pali5("AGCTTTTTTTGACGATATCGTCCGT");
    expectedBreakpoint = 15;
    const auto& centers5 = PalindromeTrieCenter(pali5, adapters, hqrOffset,
                                                kmerSize, centerSampleSize);
    ASSERT_EQ(centers5.size(), 1);
    center = centers5[0];
    EXPECT_EQ(expectedBreakpoint, center);

    kmerSize = 4;
    // slightly off center
    //                           ||
    //                           \/
    //                 01234567890123456789
    std::string pali6("AGCGACGACATATGTCCCGT");
    expectedBreakpoint = 10;
    const auto& centers6 = PalindromeTrieCenter(pali6, adapters, hqrOffset,
                                                kmerSize, centerSampleSize);
    ASSERT_EQ(centers6.size(), 1);
    center = centers6[0];
    EXPECT_EQ(expectedBreakpoint, center);
}

TEST(AdapterTest, HashKmer)
{
    int kmerSize = 3;
    std::vector<int> powlut(kmerSize + 1, 1l);
    for (int offset = 1; offset <= kmerSize; ++offset)
        powlut[offset] = powlut[offset - 1] * 4;
    std::string pali("CAT");
    int fdindex, rcindex;
    std::tie(fdindex, rcindex) = HashKmer(pali, 0, kmerSize, powlut);
    EXPECT_EQ(19, fdindex);
    EXPECT_EQ(14, rcindex);

    // update the hash
    std::string pali2("CATA");
    std::tie(fdindex, rcindex) = HashKmer(pali2, 0, kmerSize, powlut);
    EXPECT_EQ(19, fdindex);
    EXPECT_EQ(14, rcindex);
    std::tie(fdindex, rcindex) = HashKmer(pali2, 1, kmerSize, powlut);
    EXPECT_EQ(12, fdindex);
    EXPECT_EQ(51, rcindex);
    // reset the fdindex/rcindex to pre-update values
    std::tie(fdindex, rcindex) = HashKmer(pali2, 0, kmerSize, powlut);
    std::tie(fdindex, rcindex) = UpdateKmerHash(pali2, 1, kmerSize, powlut,
                                                fdindex, rcindex);
    EXPECT_EQ(12, fdindex);
    EXPECT_EQ(51, rcindex);
}

