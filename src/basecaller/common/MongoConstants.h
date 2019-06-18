#ifndef mongo_common_MongoConstants_H_
#define mongo_common_MongoConstants_H_

namespace PacBio {
namespace Mongo {

static constexpr unsigned int laneSize = 64u;
static constexpr unsigned int cudaThreadsPerWarp = 32u;

}}      // namespace PacBio::Mongo

#endif // mongo_common_MongoConstants_H_
