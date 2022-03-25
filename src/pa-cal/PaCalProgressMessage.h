#ifndef PA_CAL_PACALPROGRSSMESSAGE_H
#define PA_CAL_PACALPROGRSSMESSAGE_H

#include <app-common/ProgressMessage.h>
#include <pacbio/utilities/SmartEnum.h>

namespace PacBio::Calibration {

SMART_ENUM(PaCalStages,
           StartUp,
           Analyze,
           Shutdown);

using PaCalProgressMessage = PacBio::IPC::ProgressMessage<PaCalStages>;
using PaCalStageReporter = PaCalProgressMessage::StageReporter;

}

#endif
