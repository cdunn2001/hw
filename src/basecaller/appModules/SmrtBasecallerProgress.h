#ifndef APPMODULES_SMRTBASECALLERPROGRESS_H
#define APPMODULES_SMRTBASECALLERPROGRESS_H

#include <pacbio/utilities/SmartEnum.h>
#include <app-common/ProgressMessage.h>

SMART_ENUM(SmrtBasecallerStages,
           StartUp,
           BazCreation,
           Analyze,
           Shutdown);

using SmrtBasecallerProgressMessage =  PacBio::IPC::ProgressMessage<SmrtBasecallerStages>;
using SmrtBasecallerStageReporter = SmrtBasecallerProgressMessage::ThreadSafeStageReporter;

#endif
