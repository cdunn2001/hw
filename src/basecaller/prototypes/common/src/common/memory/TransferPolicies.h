/*
 * TransferPolicies.h
 *
 *  Created on: Feb 14, 2019
 *      Author: bbyington
 */

#ifndef TRANSFERPOLICIES_EXP_H_
#define TRANSFERPOLICIES_EXP_H_

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

#include <pacbio/cuda/PBCudaRuntime.h>

namespace PacBio {
namespace Cuda {
namespace Memory {
namespace Experimental {

// Determines how device memory will be allocated in
// regards to host memory.  It can always be present,
// only present when necessary, and only present when
// necessary with memory selected from a pool.
enum class BackingStrategy
{
    ALWAYS,
    TEMPORARY,
    POOLED
};

// Controls automatic data synchronization direction.
enum class SyncDirection
{
    HostWriteDeviceRead,  // Write on host read on device
    HostReadDeviceWrite,    // Write on device, read on host
    Symmetric  // Read/write either direction
};

template <typename T>
struct CudaPointers
{
    T* hostMem_ = nullptr;
    T* deviceMem_ = nullptr;
};

// Separate host and device memory
// Host memory always backed by device memory
// Host memory *not* managed by cuda
struct BasicPolicy
{
    template <typename T>
    static CudaPointers<T> Allocate(size_t s)
    {
        CudaPointers<T> ret;
        ret.hostMem_ = new T[s];
        ret.deviceMem_ = CudaMalloc<T>(s);
        return ret;
    }

    template <typename T>
    static void Deallocate(CudaPointers<T>& data)
    {
        delete [] data.hostMem_;
        CudaFree(data.deviceMem_);
        data = CudaPointers<T>();
    }

	template <typename T>
    static void CopyToDevice(CudaPointers<T>& data, size_t count)
	{
	    CudaCopyDevice(data.deviceMem_, data.hostMem_, count);
	}
	template <typename T>
    static void CopyToHost(CudaPointers<T>& data, size_t count)
	{
	    CudaCopyHost(data.hostMem_, data.deviceMem_, count);
	}

	static constexpr BackingStrategy backing = BackingStrategy::ALWAYS;
};

// Separate host and device memory
// Host memory always backed by device memory
// Host memory *is* managed by cuda (for faster transfer)
struct PinnedPolicy
{
    template <typename T>
    static CudaPointers<T> Allocate(size_t s)
    {
        CudaPointers<T> ret;
        ret.hostMem_ = CudaMallocHost<T>(s);
        ret.deviceMem_ = CudaMalloc<T>(s);
        return ret;
    }

    template <typename T>
    static void Deallocate(CudaPointers<T>& data)
    {
        CudaFreeHost(data.hostMem_);
        CudaFree(data.deviceMem_);
        data = CudaPointers<T>();
    }

	template <typename T>
    static void CopyToDevice(CudaPointers<T>& data, size_t count)
	{
	    CudaCopyDevice(data.deviceMem_, data.hostMem_, count);
	}
	template <typename T>
    static void CopyToHost(CudaPointers<T>& data, size_t count)
	{
	    CudaCopyHost(data.hostMem_, data.deviceMem_, count);
	}

	static constexpr BackingStrategy backing = BackingStrategy::ALWAYS;
};

// Unified host and device memory
// Not currently for use on Jetson!  That architecture does not support concurrent
// managed access, meaning we cannot write to a unified memory arena if any GPU
// kernels are running, *regardless* of if that kernel would access the same
// managed allocation.
struct UnifiedPolicy
{
    template <typename T>
    static CudaPointers<T> Allocate(size_t s)
    {
        CudaPointers<T> ret;
        ret.hostMem_ = CudaMallocManaged<T>(s);
        ret.deviceMem_ = ret.hostMem_;
        return ret;
    }

    template <typename T>
    static void Deallocate(CudaPointers<T>& data)
    {
        CudaFree(data.hostMem_);
        data = CudaPointers<T>();
    }

	template <typename T>
    static void CopyToDevice(CudaPointers<T>& data, size_t count)
	{
	    // Nothing do to
	}
	template <typename T>
    static void CopyToHost(CudaPointers<T>& data, size_t count)
	{
	    // Nothing to do
	}

	static constexpr BackingStrategy backing = BackingStrategy::ALWAYS;
};

struct ZeroCopyPolicy
{
    template <typename T>
    static CudaPointers<T> Allocate(size_t s)
    {
        CudaPointers<T> ret;
        ret.hostMem_ = CudaMallocHostZero<T>(s);
        ret.deviceMem_ = CudaHostGetDevicePtr(ret.hostMem_);
        return ret;
    }

    template <typename T>
    static void Deallocate(CudaPointers<T>& data)
    {
        CudaFreeHost(data.hostMem_);
        data = CudaPointers<T>();
    }

	template <typename T>
    static void CopyToDevice(CudaPointers<T>& data, size_t count)
	{
	    // Nothing do to
	}
	template <typename T>
    static void CopyToHost(CudaPointers<T>& data, size_t count)
	{
	    // Nothing to do
	}

	static constexpr BackingStrategy backing = BackingStrategy::ALWAYS;
};

using DefaultPolicy = PinnedPolicy;

}}}}

#endif /* TRANSFERPOLICIES_EXP_H_ */
