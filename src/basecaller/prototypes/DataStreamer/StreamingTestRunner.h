#ifndef PACBIO_CUDA_STREAMING_TEST_RUNNER_H
#define PACBIO_CUDA_STREAMING_TEST_RUNNER_H

#include <common/ZmwDataManager.h>

namespace PacBio {
namespace Cuda {

void RunTest(const Data::DataManagerParams& params, size_t simulKernels);

}} // ::PacBio::Cuda

#endif // PACBIO_CUDA_STREAMING_TEST_RUNNER_H
