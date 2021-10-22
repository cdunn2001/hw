// Copyright (c) 2021, Pacific Biosciences of California, Inc.
//
// All rights reserved.
//
// THIS SOFTWARE CONSTITUTES AND EMBODIES PACIFIC BIOSCIENCES' CONFIDENTIAL
// AND PROPRIETARY INFORMATION.
//
// Disclosure, redistribution and use of this software is subject to the
// terms and conditions of the applicable written agreement(s) between you
// and Pacific Biosciences, where "you" refers to you or your company or
// organization, as applicable.  Any other disclosure, redistribution or
// use is prohibited.
//
// THIS SOFTWARE IS PROVIDED BY PACIFIC BIOSCIENCES AND ITS CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL PACIFIC BIOSCIENCES OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
// OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
// ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef BAZIO_FILE_RUN_LENGTH_ENC_H
#define BAZIO_FILE_RUN_LENGTH_ENC_H

#include <functional>

#include <json/json.h>

namespace PacBio::BazIO
{

// Converts the run length specified in the JSON to hex.
Json::Value RunLengthEncLUTHexJson(const std::vector<std::pair<uint32_t, uint32_t>>& input);

// Converts the sequence into a JSON array consisting of elements starting number and run length.
template <typename TInput>
std::vector<std::pair<uint32_t, uint32_t>> RunLengthEncLUT(const std::vector<TInput>& input,
                                                           const std::function<bool(uint32_t,uint32_t)>& cmp
                                                           = [](uint32_t val1, uint32_t val2) { return val1 == val2 + 1; })
{
    std::vector <std::pair<uint32_t, uint32_t>> rleLut;
    if (!input.empty())
    {
        uint32_t startNumber = input[0];
        uint32_t currentNumber = input[0];
        uint32_t currentCount = 1;
        for (size_t i = 1; i < input.size(); ++i)
        {
            if (cmp(input[i], currentNumber))
            {
                currentNumber = input[i];
                ++currentCount;
            }
            else
            {
                rleLut.emplace_back(startNumber, currentCount);
                startNumber = input[i];
                currentNumber = input[i];
                currentCount = 1;
            }
        }
        rleLut.emplace_back(startNumber, currentCount);
    }

    return rleLut;
}

// Encodes a run with the same value.
template <typename TInput>
std::vector<std::pair<uint32_t, uint32_t>> RunLengthSameEncLUT(const std::vector<TInput>& input)
{
    return RunLengthEncLUT(input, [](uint32_t val1, uint32_t val2) { return val1 == val2; });
}

// Decodes the RLE in JSON format with the start number specified in hex.
std::vector<uint32_t> RunLengthDecLUTHexJson(const Json::Value& node,
                                             const std::function<uint32_t(uint32_t,uint32_t)>& ins
                                             = [](uint32_t val, uint32_t rl) { return val + rl; });

// Decodes a run with the same value.
std::vector<uint32_t> RunLengthSameDecLUTHexJson(const Json::Value& node);

} // PacBio::BazIO

#endif // BAZIO_FILE_RUN_LENGTH_ENC_H
