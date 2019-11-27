// Copyright (c) 2018, Pacific Biosciences of California, Inc.
//
// All rights reserved.
//
// THIS SOFTWARE CONSTITUTES AND EMBODIES PACIFIC BIOSCIENCES' CONFIDENTIAL
// AND PROPRIETARY INFORMATION.
//
// Disclosure, redistribution and use of this software is subject to the
// terms and conditions of the applicable written agreement(s) between you
// and Pacific Biosciences, where "you" refers to you or your company or
// organization, as applicable.  Any other disclosure, redistribution or
// use is prohibited.
//
// THIS SOFTWARE IS PROVIDED BY PACIFIC BIOSCIENCES AND ITS CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL PACIFIC BIOSCIENCES OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
// OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
// ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//  Description:
/// \brief  HDF5 support for EventObjects for mov.h5 and trc.h5

#include <pacbio/logging/Logger.h>
#include <sstream>
#include <pacbio/PBException.h>
#include <pacbio/utilities/ISO8601.h>

#include <pacbio/primary/HDF5cpp.h>
#include <pacbio/primary/EventObject.h>
#include <pacbio/primary/SequelHDF5.h>
#include <pacbio/primary/SequelMovieEventsHDF5.h>

namespace PacBio {
namespace Primary {

/// open file for read. load all attributes and dataset definitions.
void SequelMovieEventsHDF5::OpenForRead(H5::H5File& hdf5file)
{
    try
    {
        {
            LOCK_HDF5();
            gEvents = hdf5file.openGroup("/Events");
        }
        laserPowerChanges.OpenForRead(gEvents);
    }
    catch (H5::Exception&)
    {
        PBLOG_WARN << "Can't open /Events";
    }
}

void SequelMovieEventsHDF5::CreateForWrite(H5::H5File& hdf5file, const std::vector<std::string>& laserNames)
{
    {
        LOCK_HDF5();
        gEvents = hdf5file.createGroup("/Events");
    }
    laserPowerChanges.CreateForWrite(gEvents,laserNames);
}

//////////////////

void SequelMovieLaserPowerEventsHDF5::OpenForRead(H5::Group& gEvents)
{
    nL = 0;
    nLPC = 0;

    try
    {
        LOCK_HDF5();

        gLaserPowerChanges = gEvents.openGroup("LaserPowerChanges");

        dsTimeStamp     = gLaserPowerChanges.openDataSet("TimeStamp");
        dsFrameInterval = gLaserPowerChanges.openDataSet("FrameInterval");
        dsPower         = gLaserPowerChanges.openDataSet("Power");
        atLaserName     = dsPower.openAttribute("LaserName");

        auto dims = GetDims(dsTimeStamp);
        nLPC = dims[0];

        dims = GetDims(atLaserName);
        nL = dims[0];

        atLaserName >> laserNames_;
    }
    catch (H5::Exception&)
    {
        PBLOG_WARN << "Can't open LaserPowerChanges";
    }
}

const std::vector<std::string>& SequelMovieLaserPowerEventsHDF5::GetLasers() const
{
    return laserNames_;
}


void SequelMovieLaserPowerEventsHDF5::CreateForWrite(H5::Group& gEvents, const std::vector<std::string>& laserNames)
{
    laserNames_ = laserNames;
    nL = laserNames.size();
    nLPC = 0;

    LOCK_HDF5();

    hsize_t hspace1_start[1]= {0};
    hsize_t hspace1_max[1]={H5S_UNLIMITED};
    H5::DataSpace hspace_nLPC(1, hspace1_start, hspace1_max);

    hsize_t hspace2_start[2]={0,2};
    hsize_t hspace2_max[2]={H5S_UNLIMITED,2};
    H5::DataSpace hspace_nLPC_x_2(2, hspace2_start, hspace2_max);

    hsize_t hspace3_start[3]={0,nL,2};
    hsize_t hspace3_max[3]={H5S_UNLIMITED,nL,2};
    H5::DataSpace hspace_nLPC_x_nL_x_2(3, hspace3_start, hspace3_max);

    hsize_t hspace4_start[1]={nL};
    H5::DataSpace hspace_nL(1, hspace4_start);

    // H5S_UNLIMITED requires that you give a chunking dimension to the property list,
    // even if it is meaningless.
    H5::DSetCreatPropList propList1;
    H5::DSetCreatPropList propList2;
    H5::DSetCreatPropList propList3;
    hsize_t chunk_dims[3]{1,1,1};
    propList1.setChunk(1, chunk_dims);
    propList2.setChunk(2, chunk_dims);
    propList3.setChunk(3, chunk_dims);

    H5::DataSpace scalar;

    gLaserPowerChanges = gEvents.createGroup("LaserPowerChanges");
    dsTimeStamp     = gLaserPowerChanges.createDataSet("TimeStamp", SeqH5string(), hspace_nLPC         , propList1);
    dsFrameInterval = gLaserPowerChanges.createDataSet("FrameInterval",  uint32(), hspace_nLPC_x_2     , propList2);
    dsPower         = gLaserPowerChanges.createDataSet("Power",         float32(), hspace_nLPC_x_nL_x_2, propList3);
    atLaserName     = dsPower.createAttribute("LaserName",    SeqH5string(), hspace_nL           );

    atLaserName << laserNames;
}

std::vector<EventObject> SequelMovieLaserPowerEventsHDF5::ReadAllEvents()
{
    std::vector<EventObject> events(nLPC);

    for(uint32_t ievent=0; ievent < nLPC; ++ievent)
    {
        GetEvent(ievent, events.at(ievent));
    }

    return events;
}

std::vector<LaserPowerChange> SequelMovieLaserPowerEventsHDF5::ReadAllLaserChanges() const
{
    using std::vector;
    using std::ostringstream;
    using boost::multi_array;

    // The object to be returned.
    vector<LaserPowerChange> lpcs;

    if (nLPC == 0)
    {
        // No laser power change events in trace file.
        return lpcs;
    }

    const auto& laserNames = GetLasers();
    const auto nLasers = laserNames.size();

    vector<std::string> timeStampISO;
    dsTimeStamp >> timeStampISO;
    if (timeStampISO.size() != nLPC)
    {
        ostringstream msg;
        msg << "Bad size of LaserPowerChanges/TimeStamp data set: Read "
            << timeStampISO.size() << ". Expected " << nLPC << '.';
        throw PBException(msg.str());
    }

    multi_array<uint64_t, 2> frameInterval;
    dsFrameInterval >> frameInterval;
    if (frameInterval.shape()[0] != nLPC || frameInterval.shape()[1] != 2)
    {
        ostringstream msg;
        msg << "Bad size of LaserPowerChanges/FrameInterval data set: Read ("
            << frameInterval.shape()[0] << ", " << frameInterval.shape()[1]
            << "). Expected (" << nLPC << ", 2).";
        throw PBException(msg.str());
    }

    multi_array<float, 3> power;
    dsPower >> power;
    if (power.shape()[0] != nLPC
            || power.shape()[1] != nLasers
            || power.shape()[2] != 2)
    {
        throw PBException("Bad size of LaserPowerChanges/Power data set.");
    }

    lpcs.reserve(nLPC);
    for (uint32_t i = 0; i < nLPC; ++i)
    {
        const double t = PacBio::Utilities::ISO8601::EpochTime(timeStampISO[i]);
        LaserPowerChange lpc (t, frameInterval[i], power[i][0], power[i][1]);
        lpcs.push_back(lpc);
    }

    return lpcs;
}


void SequelMovieLaserPowerEventsHDF5::AddEvent(const EventObject& eo)
{
    PBHDF5::Append(dsTimeStamp, eo.timeStamp());


    PBHDF5::Record1D<uint32_t> interval(2);
    interval[0] = static_cast<uint32_t>(eo.startFrame() - firstFrame_);
    interval[1] = static_cast<uint32_t>(eo.stopFrame() - firstFrame_);
    PBHDF5::Append(dsFrameInterval, interval);

    PBHDF5::Record2D<float> power(nL,2);
    for(uint32_t i=0; i< nL; i++)
    {
        power(i, 0) = -1.0;
        power(i, 1) = -1.0;
    }
    for(uint32_t i=0; i< eo.lasers.size(); i++)
    {
        if (eo.lasers[i].name() == LaserPowerObject::LaserName::unknown)
        {
            throw PBException("Laser #" + std::to_string(i) + " name not allowed to be 'unknown'");
        }
        uint32_t dstIndex = static_cast<uint32_t>(eo.lasers[i].name().native());
        if (dstIndex >= nL)
        {
            throw PBException("out of range! dstIndex:" + std::to_string(dstIndex) +
            " nL:" + std::to_string(nL));
        }
        power(dstIndex, 0) = static_cast<float>(eo.lasers[i].startPower_mW());
        power(dstIndex, 1) = static_cast<float>(eo.lasers[i].stopPower_mW());
    }
    PBHDF5::Append(dsPower, power);

    nLPC++;
}

void SequelMovieLaserPowerEventsHDF5::GetEvent(uint32_t index, EventObject& eo)
{
    eo.eventType = EventObject::EventType::laserpower; // fix me. This is not general

    std::string timestamp;
    PBHDF5::Get(dsTimeStamp, index, timestamp);
    eo.timeStamp = timestamp;

    PBHDF5::Record1D<uint32_t> interval(2);
    PBHDF5::Get(dsFrameInterval, index, interval);
    eo.startFrame = interval[0] + firstFrame_;
    eo.stopFrame = interval[1] + firstFrame_;

    PBHDF5::Record2D<float> power(nL,2);
    PBHDF5::Get(dsPower, index, power);
    for(uint32_t i=0; i< nL; i++)
    {
        eo.lasers[i].name = LaserPowerObject::LaserName(laserNames_.at(i));
        eo.lasers[i].startPower_mW = power(i, 0);
        eo.lasers[i].stopPower_mW  = power(i, 1);
    }
}

}};
