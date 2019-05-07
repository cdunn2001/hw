// Copyright (c) 2019, Pacific Biosciences of California, Inc.
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
//

#ifndef PACBIO_CUDA_MEMORY_DATA_MANAGER_KEY_H
#define PACBIO_CUDA_MEMORY_DATA_MANAGER_KEY_H

#include <common/cuda/CudaFunctionDecorators.h>

namespace PacBio {
namespace Cuda {
namespace Memory {

// Move to own file
namespace detail {

// This file warrants some discussion and justification
//
// There are a number of quirks the mongo data managment classes
// try to accomidate.  For example, gpu device allocations appear
// on the host just as a regular pointer.  They are illegal to
// dereference on the host even when not null.  Thus gpu allocations
// are strongly encapsulated, so they remain an opaque type on the
// host and can only actually be seen when on the gpu.
//
// Another example is that some allocations are associated with a
// specific shape/layout, e.g. corresponding to our trace data layout.
// Once such an allocation is made, the data management classes
// make it impossible (or at least very difficult) to get ahold
// of the raw allocation without the associated layout metadata,
// thus enforcing safe and correct access patterns.
//
// Both of these examples are made more complicated by the fact that
// allocation lifetime management must occur in host-side code, but
// any function callable on the gpu side must have a __device__
// decorator and must reside in a .cu or .cuh file.  This requirement
// has led to a sort of reverse-pimpl pattern, where a class is
// declared in a normal .h file with data (protected or private)
// declared but no real member functions.  Then in a .cuh file it is
// augmented by a class that provides no additional data but
// does provide __device__ functions for use on the GPU.  This allows
// pure C++ code suitable for any vanilla .cpp translation unit to
// safely pass around and control the lifecycle of obects on the gpu,
// while still allowing strong and high level abstractions to be
// used on the gpu.
//
// As an aside, one might think we can merely compile the entire
// project via nvcc and have everything in .cu files.  That works
// in theory, but in practice the nvcc compilation process isn't
// perfect, and does choke up on valid (especially complicated)
// c++ code.  Thus it's desirable to limit .cu files to gpu
// code and host code that invokes gpu kernels.  Everything that
// can be compiled in a vanilla .cpp translation unit probably
// should be.
//
// This unusual separation functionality and data layout can make
// it difficult to maintain strong encapsulation.  Things like
// friendship and inheritance can be used to bridge the gap between
// the pure C++ data class and the gpu implementation class while
// still maintaining tight encapsulation, but both approaches can be
// dissatisfactory.  Friendship gives carte blanche access and
// inheritance is meant to represent something else.  Both are brittle
// and require explicitly enumerating who has access to what, which
// sounds good at first, but can lead to weird inheritance structures
// or explicit lists of forward declares that are hard to maintain.
//
// This class works around all that, by implementing a sort of passkey.
// Trusted classes can inherit from DataManager, to gain the ability to
// call the protected static function and construct a DataManagerKey
// object.  Any class that is not a tightly integrated part of this
// data management framework should *not* inherit from DataMaanger.
// Now any class that wishes to provide a function that should only
// be callable by other data management classes, it merely needs
// accept a DataManagerKey value as one of it's parameters.
//
// This technique does not prevent abuse.  Obviously anyone can
// inherit from this class and thus gain access to "hidden" things.
// But it does make it easier to control who sees what.  When
// necessary, data management classes can "break encapsulation" and
// expose some of it's internal details, and as long as that function
// is guarded by a DataManagerKey parameter, then it can be resonably
// assured it is only accessed by classes that are closely related
// and have reason to interface with deeper implementation details.
// Strong encapsulation is still enforced with external and
// unrelated code.
class DataManagerKey
{
    friend class DataManager;
private:
    // Not using =default, so we can disable aggregate initialization
    CUDA_ENABLED DataManagerKey() {}
};

class DataManager
{
protected:
    CUDA_ENABLED static DataManagerKey DataKey() { return DataManagerKey{}; }
};

}}}}

#endif // PACBIO_CUDA_MEMORY_DATA_MANAGER_KEY_H
