// Copyright (c) 2015-2018, Pacific Biosciences of California, Inc.
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

// Programmer: Martin Smith

#pragma once

#include <iostream>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include <postprimary/application/UserParameters.h>
#include <postprimary/bam/EventData.h>

#include "InsertState.h"

namespace PacBio {
namespace Primary {
namespace Postprimary {

// Base class specifying the general interface for insert classification
class InsertFinder
{
public: /// Client API
    virtual std::vector<InsertState> ClassifyInserts(const BazEventData& packets) const = 0;

    virtual ~InsertFinder() {};
};

// Classifier for identifying burst inserts.  Only operates on bases.  If any
// pulses are present (internal mode), they will remain as `EX_SHORT_PULSE`
// in the output.
class BurstInsertFinder : public InsertFinder
{
public: // structors
    // Primary constructor
    BurstInsertFinder(uint32_t minInsertionLength)
    : minInsertionLength_(minInsertionLength)
    {};

    // No move/copy
    BurstInsertFinder(BurstInsertFinder&&) = delete;
    BurstInsertFinder(const BurstInsertFinder&) = delete;
    BurstInsertFinder& operator=(BurstInsertFinder&&) = delete;
    BurstInsertFinder& operator=(const BurstInsertFinder&) = delete;

    ~BurstInsertFinder() override = default;

public: /// Client API

    std::vector<InsertState> ClassifyInserts(const BazEventData& packets) const override;

private:
    
    uint32_t minInsertionLength_;
};

class SimpleInsertFinder : public InsertFinder
{
public: // structors
    // Primary constructor
    SimpleInsertFinder()= default;

    // No Copy/move
    SimpleInsertFinder(SimpleInsertFinder&&) = delete;
    SimpleInsertFinder(const SimpleInsertFinder&) = delete;
    SimpleInsertFinder& operator=(SimpleInsertFinder&&) = delete;
    SimpleInsertFinder& operator=(const SimpleInsertFinder&) = delete;

    ~SimpleInsertFinder() override = default;

public: /// Client API

    std::vector<InsertState> ClassifyInserts(const BazEventData& read) const override;
};

// Helper class to return a concrete implementation of InsertFinder
std::unique_ptr<InsertFinder>
InsertFinderFactory(const UserParameters& user);

}}} // ::PacBio::Primary::Postprimary


