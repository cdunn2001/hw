//
// Created by mlakata on 1/23/19.
//

#ifndef SEQUELACQUISITION_FRAMECLASS_H
#define SEQUELACQUISITION_FRAMECLASS_H

#include <pacbio/utilities/SmartEnum.h>

namespace PacBio  {
namespace Primary {

// Defines the frame format coming from the sensor via Aurora PDUs, as well as stored in movie files.
// basically, the magic word of the PDU as well as the bit-depth of the pixels.

#if 1
SMART_ENUM(FrameClass,
    UNKNOWN = 0, // default value when unset
    Irrelevant,  // For USB for example
    Format2C2A,
    Format1C4A
);
#else
enum class FrameClass
{
    UNKNOWN = 0,
    Format2C2A,
    Format1C4A
};
#endif

}} //namespaces

#endif //SEQUELACQUISITION_FRAMECLASS_H
