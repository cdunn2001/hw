//
// Created by mlakata on 10/28/19.
//

#ifndef PA_WS_SEQUELTESTROIS_H
#define PA_WS_SEQUELTESTROIS_H

#include <pacbio/primary/SequelROI.h>

namespace PacBio {
namespace Primary {

inline SequelRectangularROI SequelROI_SequelAlphaFull()
{
    return SequelRectangularROI(0, 0, SequelLayout::maxPixelRows, SequelLayout::maxPixelCols,
                                SequelSensorROI::SequelAlpha());
}

}}

#endif //PA_WS_SEQUELTESTROIS_H
