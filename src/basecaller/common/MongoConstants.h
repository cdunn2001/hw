#ifndef mongo_common_MongoConstants_H_
#define mongo_common_MongoConstants_H_

namespace PacBio {
namespace Mongo {

static constexpr unsigned int cudaThreadsPerWarp = 32u;
static constexpr unsigned int cacheLineSize = 64u;
static constexpr unsigned int laneSize = 64u;
static constexpr unsigned int numAnalogs = 4u;

}}      // namespace PacBio::Mongo

#endif // mongo_common_MongoConstants_H_
