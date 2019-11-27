//
// Created by mlakata on 10/8/19.
//

#ifndef SEQUELACQUISITION_SEQUELMOVIECONFIG_H
#define SEQUELACQUISITION_SEQUELMOVIECONFIG_H

#include <pacbio/primary/ChipClass.h>
#include <pacbio/primary/SequelROI.h>
#include <H5public.h>
#include <memory>

namespace PacBio {
namespace Primary {


/// Used to configure the constructors of various SequelMovieBase derived classes.
struct SequelMovieConfig
{
    /// Defines the HDF5 chunking dimensions for Movie files
    struct MovieChunking {
        hsize_t frame = 0;
        hsize_t row = 0;
        hsize_t col = 0;
    };
    /// Defines the HDF5 chunking dimensions for Trace files
    struct TraceChunking
    {
        hsize_t zmw = 16;
        hsize_t channel = 0; // this is ignored when Setting. Set it to zero. It is only valid after GetChunking()
        hsize_t frame = 512;
    };

    /// Linux file path to write the movie to
    std::string path;

    /// The "file" roi, containing the pixels that are to be written to the file. Even though the SequelMovieConfig
    /// instance only needs to have a lifetime during the construction of the under lying file, here it will own
    /// the ROI.  A converient way of copying (rather than moving) an ROI is to use
    ///   roi.reset(sourceROI.Clone());
    std::unique_ptr <SequelROI> roi;

    /// The chipclass of the file, which will influence for format
    ChipClass chipClass = ChipClass::UNKNOWN;

    /// total number of frames to write, if known ahead of time. Some movie files
    /// can be appended to an infinite number of files. Some movie files
    /// need to be allocated with a fixed number of frames immediately at creation time.
    uint64_t numFrames = 0;

    /// Defines the "SequelMovieFileHDF5" specific parameters.
    struct
    {
        /// Chunking is an HDF5 concept, designed to optimize reads and writes to the data
        MovieChunking chunking;
        /// HDF5 defines different levels of compression. The values of this parameter are not documented here.
        uint32_t compression = 0;
    } movie;

    /// Defines the "SequelTraceFile" specific parameters.
    struct
    {
        /// Chunking is an HDF5 concept, designed to optimize reads and writes to the data
        TraceChunking chunking;
        /// HDF5 defines different levels of compression. The values of this parameter are not documented here.
        uint32_t compression = 0;
    } trace;

    /// assignment operator.
    SequelMovieConfig& operator=(const SequelMovieConfig& mc)
    {
        path = mc.path;
        roi.reset(mc.roi ? mc.roi->Clone() : nullptr);
        chipClass = mc.chipClass;
        numFrames = mc.numFrames;
        movie = mc.movie;
        trace = mc.trace;
        return *this;
    }
};

}}

#endif //SEQUELACQUISITION_SEQUELMOVIECONFIG_H
