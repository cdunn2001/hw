#ifndef PA_CAL_PACALPROGRESSMESSAGE_H
#define PA_CAL_PACALPROGRESSMESSAGE_H

#include <app-common/ProgressMessage.h>
#include <pacbio/utilities/SmartEnum.h>

namespace PacBio::Calibration {

SMART_ENUM(PaCalStages,
           StartUp,
           Analyze,
           Shutdown);

using PaCalProgressMessage = PacBio::IPC::ProgressMessage<PaCalStages>;
using PaCalStageReporter = PaCalProgressMessage::StageReporter;

inline PaCalProgressMessage::Table& PaCalProgressStages()
{
    static PaCalProgressMessage::Table stages = {
        { "StartUp",    { false, 0, 10 } },
        { "Analyze",    {  true, 1, 80 } },
        { "Shutdown",   { false, 2, 10 } }
    };
    return stages;
}

} // namespace

#endif
