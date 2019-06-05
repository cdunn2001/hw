// Use this #define to mark functions as capable of executing on the gpu as well as the
// host.  This extra indirection is necessary so that the extra directives are hidden
// when being compiled by anything other than nvcc.
//
// Note that decorating a function with CUDA_ENABLED just means nvcc will try and compile
// it for device side, not that it will be valid code.  There are a number of limitations
// as to what can go into device code.


#ifndef CUDAFUNCTIONDECORATORS_H_
#define CUDAFUNCTIONDECORATORS_H_

#ifdef __CUDACC__
#define DEVICE_CODE 1
#define CUDA_ENABLED __host__ __device__
#else
#define DEVICE_CODE 0
#define CUDA_ENABLED
#endif




#endif /* CUDAFUNCTIONDECORATORS_H_ */
