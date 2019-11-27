
#define DECLARE_SEQUELH5_SUPPORT()
#define DECLARE_ZMWSTAT_START_GROUP(name)
#define DECLARE_ZMWSTATDATASET_1D(prefix,name,type,units,description,dim0)           type prefix##name | units | description |
#define DECLARE_ZMWSTATDATASET_2D(prefix,name,type,units,description,dim0,dim1)      type prefix##name [ dim1 ]| units | description |
#define DECLARE_ZMWSTATDATASET_3D(prefix,name,type,units,description,dim0,dim1,dim2) type prefix##name [ dim1 ][ dim2 ] | units | description |
#define DECLARE_ZMWSTAT_END_GROUP(name) | | |
#define DECLARE_ZMWSTAT_ENUM(...)  | |  __VA_ARGS__ |

#include "ZmwStatsFileDefinition.h"

#undef DECLARE_SEQUELH5_SUPPORT
#undef DECLARE_ZMWSTAT_START_GROUP
#undef DECLARE_ZMWSTATDATASET_1D
#undef DECLARE_ZMWSTATDATASET_2D
#undef DECLARE_ZMWSTATDATASET_3D
#undef DECLARE_ZMWSTAT_END_GROUP
#undef DECLARE_ZMWSTAT_ENUM

