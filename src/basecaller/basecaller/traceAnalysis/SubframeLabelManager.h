//  Copyright (c) 2019, Pacific Biosciences of California, Inc.
//
//  All rights reserved.
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of Pacific Biosciences nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
//  NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
//  THIS LICENSE.  THIS SOFTWARE IS PROVIDED BY PACIFIC BIOSCIENCES AND ITS
//  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//  PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL PACIFIC BIOSCIENCES OR
//  ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
//  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
//  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
//  BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
//  IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
//  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.

#ifndef MONGO_BASECALLER_SUBFRAME_LABEL_MANAGER
#define MONGO_BASECALLER_SUBFRAME_LABEL_MANAGER

#include <common/MongoConstants.h>
#include <common/cuda/CudaFunctionDecorators.h>
#include <common/cuda/utility/CudaArray.h>

#include <dataTypes/Pulse.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

// This class manages all the frame label logic for the subframe model.
// There is a sort of duck-typing API going on, where alternate viterbi
// state models can use the same frame labeling and pulse accumulation
// code as long as they define a similar LabelManager class that exposes
// the same functions and members.
class SubframeLabelManager
{
public:
    static constexpr int numStates = 13;

    // Would be nice to not have to use this shadowing declaration to
    // avoid signed/unsigned comparisons.  Does the other version really
    // need to be unsigned?
    static constexpr int numAnalogs = Mongo::numAnalogs;

    using NucleotideLabel = Data::Pulse::NucleotideLabel;

    // Retrieves the up/down/full labels for each analog
    CUDA_ENABLED static constexpr int FullState(int i) { return i+1; }
    CUDA_ENABLED static constexpr int UpState(int i) { return i+1 + numAnalogs; }
    CUDA_ENABLED static constexpr int DownState(int i) { return i+1 + 2*numAnalogs; }

    template <typename Label>
    CUDA_ENABLED static auto IsPulseUpState(const Label& label)
    {
        return (label > numAnalogs) & (label <= 2*numAnalogs);
    }

    template <typename Label>
    CUDA_ENABLED static auto IsPulseDownState(const Label& label)
    {
        return (label > 2*numAnalogs);
    }

    template <typename Label1, typename Label2>
    CUDA_ENABLED static auto IsNewSegment(const Label1& prev, const Label2& next)
    {
        return IsPulseUpState(next) | ((next == BaselineLabel()) & (prev != BaselineLabel()));
    }

    CUDA_ENABLED static short BaselineLabel()
    {
        return 0;
    }

    // analogMap should list analogs in order of descending relative amplitude
    CUDA_ENABLED SubframeLabelManager(const Cuda::Utility::CudaArray<Data::Pulse::NucleotideLabel, numAnalogs>& analogMap)
        : analogMap_(analogMap)
    {}

    CUDA_ENABLED NucleotideLabel Nucleotide(short label) const
    {
        if (IsPulseDownState(label)) label -= 2*numAnalogs;
        if (IsPulseUpState(label)) label -= numAnalogs;

        assert(label > 0 && label <= numAnalogs);
        return analogMap_[label-1];
    }
private:
    Cuda::Utility::CudaArray<Data::Pulse::NucleotideLabel, numAnalogs> analogMap_;
};

}}}

#endif // MONGO_BASECALLER_SUBFRAME_LABEL_MANAGER
