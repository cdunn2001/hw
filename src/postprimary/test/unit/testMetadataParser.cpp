//
// Created by jnguyen on 2/15/17.
//

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <memory>

#include <pacbio/logging/Logger.h>

#include <postprimary/bam/RuntimeMetaData.h>
#include <postprimary/application/MetadataParser.h>

#include "test_data_config.h"

using namespace PacBio::Primary;
using namespace PacBio::Primary::Postprimary;

const std::string METADATA    = std::string(PacBio::PaPpaTestConfig::cmakeCurrentListDir) + "/data/metadata.xml";
const std::string ICSMETADATA = std::string(PacBio::PaPpaTestConfig::cmakeCurrentListDir) + "/data/ics_metadata.xml";

TEST(MetadataParser, ParseRMD)
{
PacBio::Logging::LogSeverityContext context(PacBio::Logging::LogLevel::WARN);

    auto rmd = MetadataParser::ParseRMD(METADATA);
    EXPECT_EQ("ATCTCTCTCttttcctcctcctccgttgttgttgttGAGAGAGAT", rmd->leftAdapter);
    EXPECT_EQ("ATCTCTCTCttttcctcctcctccgttgttgttgttGAGAGAGAT", rmd->rightAdapter);

    EXPECT_EQ("TAGAGAGAGAAAAGGAGGAGGAGGCAACAACAACAACTCTCTCTA", rmd->leftAdapterControl);
    EXPECT_EQ("TAGAGAGAGAAAAGGAGGAGGAGGCAACAACAACAACTCTCTCTA", rmd->rightAdapterControl);
    EXPECT_EQ("TGTCTAGGTCATCTCAACGTAGCTTTGACATATAACTTATCTAAAGTAATCCCTGCACACCTGTATGCATTATGCTTGCTATACACGCGGACACAGGCATATCATTTATTTTTTGCCATGTCCATTAATTGTTCAATAATTTTACTCACGGTATTTAATTTGATGTTGTGTTATATAGAATTGGAATTAAACTTATAAGGATGCTTAGACGTTGCATTATAAAAGTTTATGTACTAAGTATTTAAGACATTGGCATATGATTATAGCTTGACATTATTAAAAATTAATTAATTAAATCTCACACAATACTTATTCAAGACATTTTTACTAAGATAACCAAAGGAATGCGAACAAAATAATACTTAAAATATAAGACTTAGAAGTAATATGATCCAATAGTTACATATAGTACACTAAGTTCCTAAATTATATAACTTTAAAACAAAGTTACGAAATTTGGAAATAATTTTATTTAATCATATTTTCATAATAATGAAATACTGTTTATTTCAGTGGCGAAAAGAGATAATACGATTTTATAGTGATAGAATATCCTTGAAATATCTAAAGATAAAATTAGAAACTTTCTCTTTTCGCTGTAAAGCTATATGACTTAAAAATAACTTATACGCAAAGTATATTGCAGTGGAAACCCAAGAGTATAGTAGCCATGTAATCTCGGGTTCGAAACTACACGCCGCGCACGTAGTCAGATGGTCTGAAACTTGTCTGGGGCTGTTTGTTGACGGATGGAGACTTCACTAAGTGGCGTCAGGCGATGCGCACACACGGGACTCAATCCCGTAGCATGTTATGTGTCGTTCGAAACTCGTGCGTTCGAGATTTACGCCACATTGCCGGCTGGTCCAAGGACGTTATCTACCAGATGATACGGTCCAATTCGTAAGTTTGACTCACATAGTCGCGAACCGCGGAGCTGGAGAACAATAATTACCGGATGATTAGTTGACCATACGCACTATCATGCTCCGTGACTCAGTTTCCGCCATGGAGTTCTCACAGCCCCGTGTGTACCATAACTGCAGTAAGTAAGGACCTTGTTCGGAGGCCGACTCGTATTTCATATGATCTTAGTCTCGCCACCTTATCGCACGAATTGGGGGTGTCTTTTAGCCGACTCCGGCACGATCCGCCGGGAAGTTACTCGACCAGTTGCGGGACGCCCTAGTATGTTCGTATTACGTTCGATGCGTAAGCACCCCAGAGATTTTTGGCGGACGTTTCGGTAAATCATAGTAGAACCGGAGCGGTAAAGCTATTGATAACACGCAGGGACGAGCCAGTCGTCTAAGCTCCTCAGGGGTACCGTTCGCCGGACTACAGCCTGTCCCCGGCGGCCGCAACTGGGCTGCGATCCAGCCCCCGCTCCAAAAGGATGACTCGACCTTGCGCCTCGCGTACTCTGCTCTCGAGCTGTCTCCGTGGGCAATGCCGGCTCACGCTGTGGGGAACCCTGGACGCCCGGGCCGAGCCGACGTGGCCCCGCCCAGGCCTTTTCGTCGATCGCAGCTATGTACCCTGTGCTGGCCAGCGCTACTGCGCCGGCCATTAGCGGTGCGCTCTCGACTCGGCCCCAACGTAGACGGCGTCGCTGGCCGGATTCAAAGAAGTGAGCTACTACCATCGCGTGACGCCCTGCGGGCCTGAGTAACCGTGCACGAAGGACACCCCGTTCGTGGCGGGGGTTGCCTCCGCGACGGTCGCCAACGTTGGGGGTCGGTGCATTCAGGCGACGAGGGACCGCTGGTTTCCGGAGAGCGGCCTGTGCTCACACAGGTGCGGTCCATGGGGCCTGTGGATCCGGTTCTCCCACGCGTAGCGCCGGCGTTAGCATGGACGCTAAATAAGTATACGCCGGCAAAGGGAGTGTAGGCCGGCCCGAGGGCAATCGCGGTTACCGGGGTGGGGGAGCTCCCCGCACCAGCCTTGATGTGGTGTGCGAGCG", rmd->control);
}

TEST(MetadataParser, ParseOrigRMD)
{
    PacBio::Logging::LogSeverityContext context(PacBio::Logging::LogLevel::WARN);

    auto rmd = MetadataParser::ParseRMD(ICSMETADATA);
    EXPECT_EQ("ATCTCTCTCAATTTTTTTTTTTTTTTTTTTTTTTAAGAGAGAGAT", rmd->leftAdapter);
    EXPECT_EQ("ATCTCTCTCAACAACAACAACGGAGGAGGAGGAAAAGAGAGAGAT", rmd->rightAdapter);
}

TEST(MetadataParser,TheRest)
{
    PacBio::Logging::LogSeverityContext context(PacBio::Logging::LogLevel::WARN);

    auto rmd = MetadataParser::ParseRMD(ICSMETADATA);
    EXPECT_EQ("4.0.1",rmd->schemaVersion);
    EXPECT_TRUE(PacBio::Text::String::StartsWith(rmd->dataSetCollection,
                                                 "<Collections xmlns=\"http://pacificbiosciences.com/PacBioCollectionMetadata.xsd\">\n"));
    EXPECT_TRUE(PacBio::Text::String::Contains(rmd->dataSetCollection,
                                               "AutomationParameter"));
    EXPECT_TRUE(PacBio::Text::String::EndsWith(rmd->dataSetCollection,
                                                 "</Collections>\n"));
    EXPECT_EQ( "<ExternalResource UniqueId=\"af723687-00e9-473a-89a1-42b5a147d953\" MetaType=\"ExternalResource\" TimeStampedName=\"Inst1234_ExternalResource_02/25/2017 01:53:34\" />\n",
               rmd->externalResource);
    EXPECT_EQ("c1cc4c94-3454-4dde-a308-d14ec15e3cd6",rmd->subreadSet.uniqueId);
    EXPECT_EQ("Inst1234_SubreadSetCollection_02/25/2017 01:53:34",rmd->subreadSet.timeStampedName);
    EXPECT_EQ("2017-02-25T01:53:34.705915Z",rmd->subreadSet.createdAt);
    EXPECT_EQ("Sample Name-Cell1",rmd->subreadSet.name);
    EXPECT_EQ("",rmd->subreadSet.tags); // shouldn't have any tags...
}

