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

#ifndef PACBIO_CUDA_MEMORY_ALLOCATION_VIEW_H
#define PACBIO_CUDA_MEMORY_ALLOCATION_VIEW_H

#include "DataManagerKey.h"

namespace PacBio {
namespace Cuda {
namespace Memory {

// Non-owning view of gpu compatible host memory.  Can be a subset
// of the full allocation.  *Not* meant for any sort of long term
// storage.
template <typename T>
class HostView
{
public:
    HostView(T* data, size_t len, detail::DataManagerKey)
        : data_(data)
        , len_(len)
    {}

    T& operator[](size_t idx) { return data_[idx]; }
    const T& operator[](size_t idx) const {return data_[idx]; }

    T* Data() { return data_; }
    size_t Size() const { return len_; }
private:
    T* data_;
    size_t len_;
};

// Non-owning host-side representation of device-side data.  Can
// be a subset of the full allocation.  *Not* meant for any
// sort of long term storage.
//
// This class does not expose the underlying memory, and is mostly
// useless on it's own.  Pure C++ host code cannot meaningfully
// interact with the contents of the data, and this class is
// meant merely to provide a handle for passing it around. Cuda
// code should include AllocationViews.cuh to get access to the
// DeviceView class, which does provide array access and which can
// be implicitly constructed from this class.
template <typename T>
class DeviceHandle
{
public:
    DeviceHandle(T* data, size_t len, detail::DataManagerKey key)
        : data_(data)
        , len_(len)
    {}

protected:
    T* data_;
    size_t len_;
};

}}}

#endif // PACBIO_CUDA_MEMORY_ALLOCATION_VIEW_H
