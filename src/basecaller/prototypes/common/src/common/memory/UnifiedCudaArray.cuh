/*
 * UnifiedCudaArray.cuh
 *
 *  Created on: Feb 14, 2019
 *      Author: bbyington
 */

//------------------------------------------------------------
// README: This code is deprecated!
//
// This is part of the original and UnifiedCudaArray
// implementation.  At the point of this files creation,
// the UnifiedCudaArray class is being simplified and
// streamlined.  The more general original implementation,
// which supported various mechanisms for transferring data
// between host and GPU, is *probably* not needed.  Pinned
// memory host allocations with explicit transfers seems the
// way forward.  But I'm preserving the older more flexible
// implementation here for now, just in case we learn something
// new to make us revisit that choice, and want to do some
// experiments
//--------------------------------------------------------------


// Extensions to UnifiedCudaArray available only in cuda compilation units.
// In particular provide array access to device data when in device code

#ifndef UNIFIEDCUDAARRAY_EXP_CUH_
#define UNIFIEDCUDAARRAY_EXP_CUH_

#include "UnifiedCudaArray.h"

namespace PacBio {
namespace Cuda {
namespace Memory {
namespace Experimental {

template <typename T>
class DeviceView : public DeviceHandle<T>
{
    using Parent = DeviceHandle<T>;
    using Parent::data_;
    using Parent::len_;
public:
    DeviceView(const DeviceHandle<T>& handle) : DeviceHandle<T>(handle) {}
    template <typename Policy>
    DeviceView(UnifiedCudaArray<T, Policy>& ptr) : DeviceHandle<T>(ptr) {}

    __device__
    T& operator[](size_t idx) { return data_[idx]; }
    __device__
    const T& operator[](size_t idx) const {return data_[idx]; }

    __device__
    T* Data() { return data_; }
    __device__
    size_t Size() const { return len_; }

};


}}}}

#endif /* UNIFIEDCUDAARRAY_CUH_ */
