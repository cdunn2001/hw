#ifndef PACBIO_CUDA_MEMORY_UNIFIED_CUDA_ARRAY_EXP_H
#define PACBIO_CUDA_MEMORY_UNIFIED_CUDA_ARRAY_EXP_H

#include <cassert>

#include <pacbio/cuda/memory/TransferPolicies.h>

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


namespace PacBio {
namespace Cuda {
namespace Memory {
namespace Experiemental {

// Non-owning host-side view of the memory.  Can be a subset of
// the full allocation.  *Not* meant for any sort of long term
// storage.  Obtaining an instance of this class will cause
// memory to be synchronized to the host side (no performance
// penalty if it already is), and will only remain valid until
// a view of the device side is requested.
template <typename T>
class HostView
{
private:
    template <typename U, typename Policy>
    friend class UnifiedCudaArray;
    HostView(T* start, size_t idx, size_t len)
        : data_(start+idx)
        , len_(len)
    {}

public:
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
// sort of long term storage.  Obtaining an instance of this
// class will cause memory to be synchronized to the device side
// (no performance penalty if it already is), and will only
// remain valid until a view of the host side is requested.
//
// This class does not expose the underlying memory, and is mostly
// useless on it's own.  Pure C++ host code cannot meaningfully
// interact with the contents of the data, and this class is
// meant merely to provide a handle for passing it around. Cuda
// code should include UnifiedCudaArray.cuh to get access to the
// DeviceView class, which does provide array access and which can
// be implicitly constructed from this class.
template <typename T>
class DeviceHandle
{
private:
    template <typename U, typename Policy>
    friend class UnifiedCudaArray;
    DeviceHandle(T* start, size_t idx, size_t len)
        : data_(start+idx)
        , len_(len)
        , origIndex_(idx)
    {}

protected:
    T* data_;
    size_t len_;
    size_t origIndex_;
};

// TODO handle pitched allocations for multidimensional data?
// TODO the API here evolved into a cross between a smart pointer and
//      a managed array.  It should probably decide what it actually is!
template <typename T, typename Policy = DefaultPolicy>
class UnifiedCudaArray {
public:
    UnifiedCudaArray(size_t count, SyncDirection dir)
        : activeOnHost_(true)
        , count_(count)
        , syncDir_(dir)
    {
        data_ = Policy::template Allocate<T>(count_);
        for (size_t i = 0; i < count; ++i)
            data_.hostMem_[i] = T();
    }

    UnifiedCudaArray(const UnifiedCudaArray&) = delete;
    UnifiedCudaArray(UnifiedCudaArray&& other)
    {
        data_ = other.data_;
        count_ = other.count_;
        activeOnHost_ = other.activeOnHost_;
        syncDir_ = other.syncDir_;
        other.data_ = CudaPointers<T>();
    }
    UnifiedCudaArray& operator=(const UnifiedCudaArray&) = delete;
    UnifiedCudaArray& operator=(UnifiedCudaArray&& other)
    {
        Deallocate();
        data_ = other.data_;
        count_ = other.count_;
        activeOnHost_ = other.activeOnHost_;
        syncDir_ = other.syncdir_;
        other.deviceMem_ = nullptr;
        other.hostMem_ = nullptr;
        return *this;
    }

    ~UnifiedCudaArray() { Deallocate(); }

    size_t Size() const { return count_; }
    bool ActiveOnHost() const { return activeOnHost_; }

    HostView<T> GetHostView(size_t idx, size_t len)
    {
        if (!activeOnHost_) CopyImpl(true, false);
        assert(idx + len <= count_);
        return HostView<T>(data_.hostMem_, idx, len);
    }
    HostView<T> GetHostView() { return GetHostView(0, count_); }
    operator HostView<T>() { return GetHostView(); }

    DeviceHandle<T> GetDeviceHandle(size_t idx, size_t len)
    {
        if (activeOnHost_) CopyImpl(false, false);
        assert(idx + len <= count_);
        return DeviceHandle<T>(data_.deviceMem_, idx, len);
    }
    DeviceHandle<T> GetDeviceHandle() { return GetDeviceHandle(0, count_); }
    operator DeviceHandle<T>() { return GetDeviceHandle(); }

    void CopyToDevice()
    {
        CopyImpl(false, true);
    }
    void CopyToHost()
    {
        CopyImpl(true, true);
    }

private:
    void Deallocate()
    {
        Policy::Deallocate(data_);
    }

    void CopyImpl(bool toHost, bool manual)
    {
        if (toHost)
        {
            activeOnHost_ = true;
            if (manual || syncDir_ != SyncDirection::HostWriteDeviceRead)
            {
                // TODO See if there is a better strategy than synchronizing twice.
                //      We need to synchronize afterwards, to make sure the transfer is
                //      complete before we access the data.  However synchronizing before
                //      seems helpful (so far) because transfers seem to be completed in the
                //      order they are issued regardless of stream, so issuing too early
                //      may set up false data transfer dependencies that stall execution.
                // TODO If this stays, also set up an asynchronous transfer request system, so the
                //      host doesn't necessarily *have* to stall for the entirety of the transfer
                CudaSynchronizeDefaultStream();
                Policy::CopyToHost(data_, count_);
                CudaSynchronizeDefaultStream();
            }
        } else {
            activeOnHost_ = false;
            if (manual || syncDir_ != SyncDirection::HostReadDeviceWrite)
                Policy::CopyToDevice(data_, count_);
        }
    }

    CudaPointers<T> data_;

	bool activeOnHost_;
	size_t count_;
	SyncDirection syncDir_;
};

}}}} //::Pacbio::Cuda::Memory::Experimental

#endif //PACBIO_CUDA_MEMORY_UNIFIED_CUDA_ARRAY_EXP_H
