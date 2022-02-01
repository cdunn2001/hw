#include <gtest/gtest.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>
#include <thread>
#include <chrono>
#include <random>
#include <string>
#include <assert.h>

#include <postprimary/test/SentinelVerification.h>

using namespace PacBio::Primary::Postprimary;

TEST(verification, Encoding)
{
    SentinelVerification sv;

    unsigned int zmwHn1 = 10944688;
    std::string ts1 = "AAAGGGTTTCCCGTACGCGCTGACGACAGCTC";

    // Exact match.
    EXPECT_TRUE(sv.CheckZMW(ts1, zmwHn1));

    // Trailing sequence doesn't match.
    EXPECT_FALSE(sv.CheckZMW(ts1 + "ACGT", zmwHn1));

    // Trailing sequence matches.
    EXPECT_TRUE(sv.CheckZMW(ts1 + "AAA", zmwHn1));

    // Prefix with random sequence.
    EXPECT_TRUE(sv.CheckZMW("TATATATATATATA" + ts1, zmwHn1));

    // Prefix with random sequence but trailing sequence doesn't match.
    EXPECT_FALSE(sv.CheckZMW("TATATATATATATA" + ts1 + "ACGT", zmwHn1));

    // Multiple copies with random prefix sequence.
    EXPECT_TRUE(sv.CheckZMW("CGCGCGCGCGCGC" + ts1 + ts1 + ts1, zmwHn1));

    // Longer matching.
    unsigned int nCopies = 10;
    std::string seq1;
    for (unsigned int i = 0; i < nCopies; i++) seq1 += ts1;
    EXPECT_TRUE(sv.CheckZMW(seq1, zmwHn1));
    EXPECT_TRUE(sv.CheckZMW(seq1 + "A", zmwHn1));
    EXPECT_TRUE(sv.CheckZMW("TTTTTTTTTTTTTTTTTT" + seq1, zmwHn1));

    // Check some more numbers.
    unsigned int zmwHn2 = 74908671;
    std::string ts2 = "AAAGGGTTTCCCGTAGCGATGCTGTGACTATA";
    EXPECT_TRUE(sv.CheckZMW(ts2, zmwHn2));
    EXPECT_FALSE(sv.CheckZMW(ts2 + "GGG", zmwHn2));

    unsigned int zmwHn3 = 50921932;
    std::string ts3 = "AAAGGGTTTCCCGTAGTCTGCTCGTGCGCTCT";
    EXPECT_TRUE(sv.CheckZMW(ts3, zmwHn3));
    EXPECT_FALSE(sv.CheckZMW(ts3 + ts2 + ts3, zmwHn3));

    unsigned int zmwHn4 = 27722628;
    std::string ts4 = "AAAGGGTTTCCCGTACTGCTAGAGACGCACGT";
    EXPECT_TRUE(sv.CheckZMW(ts4, zmwHn4));
    EXPECT_TRUE(sv.CheckZMW("AAAAAAAAAAAAAAA" + ts4, zmwHn4));
}

