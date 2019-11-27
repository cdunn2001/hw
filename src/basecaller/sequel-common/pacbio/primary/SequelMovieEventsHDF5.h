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

#ifndef SEQUELACQUISITION_SEQUELMOVIEEVENTSHDF5_H
#define SEQUELACQUISITION_SEQUELMOVIEEVENTSHDF5_H

#include <stdint.h>
#include <vector>
#include <pacbio/primary/HDF5cpp.h>
#include <pacbio/primary/LaserPowerChange.h>

namespace PacBio {
namespace Primary {

class EventObject;

class SequelMovieLaserPowerEventsHDF5
{
public:
    H5::Group gLaserPowerChanges;

    H5::DataSet dsTimeStamp;
    H5::DataSet dsFrameInterval;
    H5::DataSet dsPower;

    H5::Attribute atLaserName;

    uint32_t nL = 0; // number of lasers (nominally 2)
    uint32_t nLPC = 0; // number of LaserPowerChange events increases from 0

    void OpenForRead(H5::Group& gEvents);
    void CreateForWrite(H5::Group& gEvents, const std::vector<std::string>& laserNames);

    std::vector<LaserPowerChange> ReadAllLaserChanges() const;
    std::vector<EventObject> ReadAllEvents();
    const std::vector<std::string>& GetLasers() const;

    void AddEvent(const EventObject& eo);
    void GetEvent(uint32_t index, EventObject& eo);

    void SetFirstFrame(uint64_t firstFrame) { firstFrame_ = firstFrame;}
private:
    uint64_t firstFrame_ = 0;
    std::vector<std::string> laserNames_;
};


class SequelMovieEventsHDF5
{
public:
    SequelMovieLaserPowerEventsHDF5 laserPowerChanges;

    H5::Group gEvents;
    void OpenForRead(H5::H5File& hdf5file);
    void CreateForWrite(H5::H5File& hdf5file, const std::vector<std::string>& laserNames);
};

}} // namespace

#endif //SEQUELACQUISITION_SEQUELMOVIEEVENTSHDF5_H
