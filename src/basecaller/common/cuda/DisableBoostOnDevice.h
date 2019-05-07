// This is slightly experimental and probably slightly dangerous.
//
// Parts of boost try to support running on the gpu by decorating their functions with __device__ __host__
// when it detects nvcc is being used.  However support is incomplete so not all boost functions are marked
// this way.  Unfortunately, due to injection via templates, it's very possible that boost functions attempting
// to support cuda call other boost functions that do not, and since it is an error to call a __host__ function
// from a __device__ __host__ function, this causes a hard compilation error.  This error occurs even if only
// the host side call paths would actually be invoked, meaning whole swathes of boost cannot be used at all in .cu
// files, even if they will never be used on the gpu.
//
// This works around that by intercepting the BOOST_GPU_ENABLED macro and tweaking it to only create host code.
// Whenever this is included you will lose the ability to invoke *any* boost from device code, but gain the
// ability to invoke *all* boost from host code.
//
// This file will cause a compilation error if it does not come before all other boost headers.  (Or at least before
// all boost headers that are aware of the host system configuration, which is the only code that would be affected
// anyway).  This should be safe to use within a single translation unit.  I would not be surprised if it's possible
// to get in a situation where different compilation units disagree on how functions are declared.  This header
// is mainly meant as a backup strategy.  *Only* use it if you are fighting odd device compiler errors in a file
// that uses boost, and even then consider re-organizing code so that all host code utilizing boost remains in
// a normal .cpp file.

#ifdef BOOST_CONFIG_HPP
#error DisableBoostOnDevice.h must be included before any boost files (preferrably first inclusion in a .cu file)
#endif

#include <boost/config.hpp>
#ifdef BOOST_GPU_ENABLED
#undef BOOST_GPU_ENABLED
#define BOOST_GPU_ENABLED __host__
#endif
