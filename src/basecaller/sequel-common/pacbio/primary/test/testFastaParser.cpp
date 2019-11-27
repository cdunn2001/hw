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

#include <pacbio/primary/SequenceUtilities.h>
#include <pacbio/primary/FastaEntry.h>
#include <pacbio/primary/FastaUtilities.h>

#include "pacbio-primary-test-config.h"

using namespace PacBio::Primary;

// Barcode test-data
const std::string BARCODES_FASTA = std::string(PacBio::Primary::CommonTest::cmakeCurrentListDir) + "/data/BarcodeReferencesSym.fasta";

const uint32_t NUM_BARCODES = 96;

// String test-data
const std::string ID = "Test";
const std::string SEQUENCE = "ATGCATGCATGCATGCATGC";
char const* SEQUENCE_PTR   = "ATGCATGCATGCATGCATGC";
const std::string REV_COMP = "GCATGCATGCATGCATGCAT";
char const* REV_COMP_PTR   = "GCATGCATGCATGCATGCAT";

// Numerical test-data
const int32_t SEQUENCE_SIZE = SEQUENCE.size();
const uint32_t FASTA_NUMBER = 42;

TEST(FastaParser, SequenceUtilities)
{
    // Test the string reverse-complement function
    std::string revComp = SequenceUtilities::ReverseCompl(SEQUENCE);
    EXPECT_EQ(revComp, REV_COMP);

    // ... and that it's reversible
    std::string sequence = SequenceUtilities::ReverseCompl(revComp);
    EXPECT_EQ(sequence, SEQUENCE);
    
    // Test the char* reverse-complement function
    char* revCompPtr = new char[SEQUENCE_SIZE]; 
    SequenceUtilities::ReverseCompl(SEQUENCE_PTR, SEQUENCE_SIZE, revCompPtr);
    EXPECT_STREQ(revCompPtr, REV_COMP_PTR);

    // ... and that it's reversible
    char* sequencePtr = new char[SEQUENCE_SIZE]; 
    SequenceUtilities::ReverseCompl(revCompPtr, SEQUENCE_SIZE, sequencePtr);
    EXPECT_STREQ(sequencePtr, SEQUENCE_PTR);
}

TEST(FastaParser, FastaEntry)
{
    // Test the creation of a FastaEntry from scratch
    FastaEntry test(ID, SEQUENCE, FASTA_NUMBER);
    EXPECT_EQ(test.id, ID);
    EXPECT_EQ(test.sequence, SEQUENCE);
    EXPECT_EQ(test.number, FASTA_NUMBER);
    EXPECT_EQ(test.Length(), SEQUENCE_SIZE);

    // Test the in-place reverse-complement function
    test.ReverseCompl();
    EXPECT_EQ(test.id, ID);
    EXPECT_EQ(test.sequence, REV_COMP);
    EXPECT_EQ(test.number, FASTA_NUMBER);
    EXPECT_EQ(test.Length(), SEQUENCE_SIZE);

    // ... and that it's reversible
    test.ReverseCompl();
    EXPECT_EQ(test.id, ID);
    EXPECT_EQ(test.sequence, SEQUENCE);
    EXPECT_EQ(test.number, FASTA_NUMBER);
    EXPECT_EQ(test.Length(), SEQUENCE_SIZE);
}

TEST(FastaParser, FastaUtilities)
{
    std::vector<FastaEntry> reads = FastaUtilities::ParseSingleFastaFile(BARCODES_FASTA);

    // Test that we read the correct number of records
    EXPECT_EQ(reads.size(), NUM_BARCODES);

    // Test that the records are in the correct order with the correct data
    FastaEntry& read0 = reads.at(0);
    EXPECT_EQ(read0.id,              "0001_Forward");
    EXPECT_EQ(read0.sequence,    "TCAGACGATGCGTCAT");
    EXPECT_EQ(read0.number,  static_cast<size_t>(0));
    FastaEntry& read42 = reads.at(42);
    EXPECT_EQ(read42.id,              "0043_Forward");
    EXPECT_EQ(read42.sequence,    "TACGCGTGTACGCAGA");
    EXPECT_EQ(read42.number, static_cast<size_t>(42));
    FastaEntry& read95 = reads.at(95);
    EXPECT_EQ(read95.id,              "0096_Forward");
    EXPECT_EQ(read95.sequence,    "GTGTGCACTCACACTC");
    EXPECT_EQ(read95.number, static_cast<size_t>(95));

    std::vector<FastaEntry> writeReads;
    writeReads.emplace_back(ID, SEQUENCE);
    const std::string fastaWritePath = std::string(PacBio::Primary::CommonTest::cmakeCurrentBinaryDir) + "/out.fasta";
    FastaUtilities::WriteSingleFastaFile(writeReads, fastaWritePath);
    std::vector<FastaEntry> readReads = FastaUtilities::ParseSingleFastaFile(fastaWritePath);

    FastaEntry& read = readReads.at(0);
    EXPECT_EQ(ID, read.id);
    EXPECT_EQ(SEQUENCE, read.sequence);

}
