//
// Created by mlakata on 9/26/19.
//

#ifndef SEQUELACQUISITION_INTERFACES_H
#define SEQUELACQUISITION_INTERFACES_H

namespace PacBio {
namespace Primary {

/// Linux process exit codes.
enum ExitCode
        : int
{
    BIST_Pass = 0,
    NormalExit = 0,
    // Linux reserves exit codes 1 and 2, as well as >= 126
    BIST_Fail = 3,
    H5Exception = 4,
    StdException = 5,
    UncaughtException = 6,
    CommandParsingException = 7,
    BIST_Exception = 8,
    DefaultUnknownFailure = 9
};

}} //namespaces

#endif //SEQUELACQUISITION_INTERFACES_H
