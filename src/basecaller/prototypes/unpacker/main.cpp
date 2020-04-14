//
// Created by mlakata on 3/20/20.
//

#include <pacbio/logging/Logger.h>
#include <pacbio/utilities/Time.h>

#include "unpacker.h"
#include "hdf5_12bitter.h"

using namespace PacBio::Primary;

int main(int argv, char* argc[])
{
    try
    {
        std::string run = (argv >= 2 ? argc[1] : "");
        if (run == "" || run == "unpacker")
        {
            double t0 = PacBio::Utilities::Time::GetMonotonicTime();
            Unpacker::test();
            double t1 = PacBio::Utilities::Time::GetMonotonicTime();
            PBLOG_INFO << "unpacker time = " << (t1-t0) << " seconds";
        }
        if (run == "" || run == "hdf5")
        {
            double t0 = PacBio::Utilities::Time::GetMonotonicTime();
            Hdf5_12bitter::test();
            double t1 = PacBio::Utilities::Time::GetMonotonicTime();
            PBLOG_INFO << "unpacker time = " << (t1-t0) << " seconds";
        }
    }
    catch(const std::exception& ex)
    {
        PBLOG_ERROR << "Exception caught: " << ex.what();
    }
    return 0;
}
