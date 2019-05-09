#include <common/DataGenerators/PicketFenceGenerator.h>

namespace PacBio {
namespace Cuda {
namespace Data {

std::random_device PicketFenceGenerator::rd;
std::mt19937 PicketFenceGenerator::gen(rd());
constexpr size_t PicketFenceGenerator::MaxFrames;

}}}
