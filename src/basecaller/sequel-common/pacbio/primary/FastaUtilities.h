#pragma once

#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>
#include <zlib.h>

#include <pacbio/primary/kseq.h>
#include <pacbio/primary/FastaEntry.h>

KSEQ_INIT(gzFile, gzread)

namespace PacBio {
namespace Primary {

/// Provides methods to parse a FASTA file.
class FastaUtilities
{

public:  // Static methods

/// Parses a fasta file, given the filePath.
/// \param  filePath  A string of the path to the Fasta file to be parsed
/// \return A vector of FastaEntry objects
static std::vector<FastaEntry> ParseSingleFastaFile(const std::string& filePath)
{
    std::vector<FastaEntry> output;
    gzFile fp;
    kseq_t *seq;
    int l;
    fp = gzopen(filePath.c_str(), "r");
    seq = kseq_init(fp);
    uint32_t number = 0;
    while ((l = kseq_read(seq)) >= 0) 
    {
        for (uint32_t i = 0; i < seq->seq.l; ++i)
        {
            seq->seq.s[i] = static_cast<char>(std::toupper(seq->seq.s[i]));
        }
        output.emplace_back(seq->name.s, seq->seq.s, number++);
    }
    kseq_destroy(seq);
    gzclose(fp);
    return output;
}

static bool WriteSingleFastaFile(const std::vector<FastaEntry>& fastaList, const std::string& filePath)
{
    std::ofstream fh(filePath);
    if (fh)
    {
        for (const auto& f : fastaList)
        {
            fh << ">" << f.id << "\n";
            fh << f.sequence << "\n";
        }
        fh.close();
        if (fh.fail())
            return false;
    }
    else
    {
        return false;
    }

    return true;
}


}; // FastaUtilities

}}

