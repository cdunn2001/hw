// Copyright (c) 2014-2016, Pacific Biosciences of California, Inc.
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted (subject to the limitations in the
// disclaimer below) provided that the following conditions are met:
//
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
//  * Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
//  * Neither the name of Pacific Biosciences nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE
// GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY PACIFIC
// BIOSCIENCES AND ITS CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL PACIFIC BIOSCIENCES OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
// USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
// OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGE.

// Programmer: Armin Töpfer

#pragma once

#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <limits>

#include <postprimary/bam/BarcodeStrategy.h>

namespace PacBio {
namespace Primary {
namespace Postprimary {

/// Command-line parameters
struct UserParameters
{
    UserParameters() {}
    UserParameters(int argc, const char* const* const argv)
    {
        std::stringstream ss;
        for(int i=0;i<argc;i++) ss << argv[i] << " ";
        originalCommandLine = ss.str();
    }

    std::string originalCommandLine;

    std::vector<std::string> inputFilePaths;
    std::string fileListPath;
    std::string uuid;
    std::string outputPrefix;
    std::string subreadsetFilePath;
    std::string runtimeMetaDataFilePath;
    bool zmq = false;
    std::string fakeHQ;

    int         threads = 1;
    int         bamthreads = 4;
    bool        polymeraseread = false;
    bool        outputFasta = false;
    bool        outputFastq = false;

    // Read filter
    static constexpr float minSnrSequel = 3.75f;
    static constexpr float minSnrSpider = 2.0f;
    uint32_t    minPolymerasereadLength = 50;
    uint32_t    minSubreadLength = 100;
    float       minSnr = 3.75;
    uint32_t    minEmptyTime = 0;
    uint32_t    emptyOutlierTime = 3;

    // Adapter finding parameters
    bool        disableAdapterFinding = false;
    std::string adaptersFilePath;
    int         flankLength = -1;
    bool        localAlnFlanking = false;
    float       minSoftAccuracy = -1.0;
    float       minHardAccuracy = -1.0;
    float       minFlankingScore = -1.0;
    bool        sensitiveCorrector = true;
    bool        correctAdapters = true;
    bool        lookForLoop = false;
    bool        trimToLoop = false;
    int         stemLength = 0;
#ifdef DIAGNOSTICS
    bool        emitAdapterMetrics = false;
#endif

    // Barcode calling parameters
    BarcodeStrategy scoreMode;

    // Control filtering parameters
    bool        disableControlFiltering = false;
    std::string controlFilePath;
    std::string controlAdaptersFilePath;
    bool        saveControls = false;
    bool        useSplitControlWorkflow = true;
    uint32_t    splitReadLength = 10000;
    uint32_t    splitReferenceLength = 15000;
#ifdef DIAGNOSTICS
    bool        emitControlMetrics = false;
#endif

    bool        hqonly = false;
    bool        nobam = false;
    bool        saveScraps = true;
    bool        savePbi = true;
    bool        inlinePbi = false;

    bool        noClassifyInsertions = false;
    uint32_t    minInsertionLength = 50;

    bool        silent = false;
    bool        noStats = false;
    bool        noStatsH5 = false;
    bool        diagStatsH5 = false;

    // HQRF parameters
    bool        hqrf = true;
    std::string hqrfMethod;
    bool        ignoreBazAL = false;

    bool streamBam = false;

    // White listing
    std::vector<uint32_t> whiteListZmwNumbers;
    std::vector<uint32_t> whiteListZmwIds;
    std::vector<int32_t> whiteListHoleNumbers;

    // for profiling
    uint32_t maxNumZmwsToProcess{std::numeric_limits<uint32_t>::max()};

    // Reports
    bool        runReport = false;

    // Knobs to help control disk IO
    size_t zmwHeaderBatchMB = 30;
    size_t zmwBatchMB = 1000;
    size_t maxInputQueueMB = 10000;

    // Knob to control output queue for memory
    size_t maxOutputQueueMB = 30000;

    size_t defaultZmwBatchSize = 100;
};

}}} // ::PacBio::Primary::Postprimary
