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

#include <json/json.h>

namespace PacBio::BazIO
{

Json::Value RunLengthEncLUTJson(const std::vector<std::pair<uint32_t, uint32_t>>& input);
Json::Value RunLengthEncLUTHexJson(const std::vector<std::pair<uint32_t, uint32_t>>& input);

template <typename TInput>
std::vector<std::pair<uint32_t, uint32_t>> RunLengthEncLUT(const std::vector<TInput>& input)
{
    std::vector <std::pair<uint32_t, uint32_t>> rleLut;
    if (!input.empty())
    {
        uint32_t startNumber = input[0];
        uint32_t currentNumber = input[0];
        uint32_t currentCount = 1;
        for (size_t i = 1; i < input.size(); ++i)
        {
            if (input[i] == currentNumber + 1)
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

std::vector<uint32_t> RunLengthDecLUTHexJson(const Json::Value& node);

} // PacBio::BazIO

#endif // BAZIO_FILE_RUN_LENGTH_ENC_H
