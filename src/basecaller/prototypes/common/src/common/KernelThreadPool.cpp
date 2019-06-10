#include "KernelThreadPool.h"

namespace PacBio {
namespace Cuda {

std::atomic<size_t> ThreadRunner::instanceCount_{0};

}}
