#pragma once

#include <chrono>
#include <cmath>
#include <ctime>
#include <iostream>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>
#include <zlib.h>

namespace PacBio {
namespace Primary {

/// Provides methods to parse a FASTA file.
class SequenceUtilities
{

public:  // Static Methods

/// Reverse complement a DNA sequence in string format
/// \param  seq  The string of DNA bases to be reverse-complemented
/// \return  A string with the reverse complement of the input
static std::string ReverseCompl(const std::string& seq)
{
    int8_t rc_table[128] = {
        4, 4, 4, 4, 4, 4, 4,  4,  4, 4, 4, 4, 4,  4,  4, 4,  4, 4, 4, 4,  4, 4,  4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4,  4,  4, 4, 4, 4, 4,  4,  4, 4,  4, 4, 4, 4,  4, 4,  4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4,  4,  4, 4, 4, 4, 4,  84, 4, 71, 4, 4, 4, 67, 4, 4,  4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 65, 65, 4, 4, 4, 4, 4,  4,  4, 4,  4, 4, 4, 84, 4, 71, 4, 4, 4, 67,
        4, 4, 4, 4, 4, 4, 4,  4,  4, 4, 4, 4, 65, 65, 4, 4,  4, 4, 4, 4,  4, 4,  4, 4};
    std::string reverseCompl(seq.length(), 'N');
    for (uint32_t i = 0; i < seq.length(); ++i)
    {
        reverseCompl[seq.length()-i-1] = (char)rc_table[(int8_t)seq[i]];
    }
    return reverseCompl;
}

/// Compute reverse complementary of given char* and place it into a second 
///    provided char*
/// \param seq forward sequence
/// \param end length of the forward sequence
/// \param rc requested rev compl sequence
static void ReverseCompl(const char* seq, int32_t end, char* rc)
{
    int32_t start = 0;
    int8_t rc_table[128] = {
        4, 4, 4, 4, 4, 4, 4,  4,  4, 4, 4, 4, 4,  4,  4, 4,  4, 4, 4, 4,  4, 4,  4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4,  4,  4, 4, 4, 4, 4,  4,  4, 4,  4, 4, 4, 4,  4, 4,  4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4,  4,  4, 4, 4, 4, 4,  84, 4, 71, 4, 4, 4, 67, 4, 4,  4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 65, 65, 4, 4, 4, 4, 4,  4,  4, 4,  4, 4, 4, 84, 4, 71, 4, 4, 4, 67,
        4, 4, 4, 4, 4, 4, 4,  4,  4, 4, 4, 4, 65, 65, 4, 4,  4, 4, 4, 4,  4, 4,  4, 4};
    rc[end] = '\0';
    --end;
    while ((start < end))
    {
        rc[start] = (char)rc_table[(int8_t)seq[end]];
        rc[end] = (char)rc_table[(int8_t)seq[start]];
        ++start;
        --end;
    }
    if (start == end) rc[start] = (char)rc_table[(int8_t)seq[start]];
}

// /// Compute reverse complementary of given string.
// /// \param seq forward sequence string
// /// \return requested rev compl sequence
// static std::string ReverseCompl(const std::string& seq)
// {
//     // Buffer char*
//     char* reverseChar = new char[seq.size() + 1];
//     // Compute
//     ReverseCompl(seq.c_str(), seq.size(), reverseChar);
//     // Convert char* to string
//     std::string reverseString(reverseChar);
//     // Free heap memory
//     delete [] reverseChar; 
//     // Return rev compl string
//     return std::move(reverseString);
// }

}; // SequenceUtilities

}}
