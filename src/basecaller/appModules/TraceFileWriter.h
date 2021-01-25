//
// Created by mlakata on 12/2/20.
//

#ifndef MONGO_APPMODULES_TRACEFILEWRITER_H
#define MONGO_APPMODULES_TRACEFILEWRITER_H

#include <pacbio/tracefile/TraceFile.h>
#include <appModules/DataFileWriter.h>

namespace PacBio {
namespace Application {

#if 0
class DataFileEvent
{

};
#endif

class TraceFileWriter : public DataFileWriterInterface
{
public:

    // Create constructor for writing, will create underlying trace contents.
    // Specifying numFrames = 0 indicates the number of frames to be written
    // is not known at creation time.
    TraceFileWriter(const std::string& fileName, size_t numZmws, size_t numFrames=0) :
        traceFile_(new PacBio::TraceFile::TraceFile(fileName, numZmws, numFrames)) {}

public:

    TraceFileWriter(const TraceFileWriter&) = delete;
    TraceFileWriter& operator=(const TraceFileWriter&) = delete;
    TraceFileWriter(TraceFileWriter&&) = default;
    TraceFileWriter& operator=(TraceFileWriter&&) = default;

    ~TraceFileWriter() override {}

public:
    PacBio::TraceFile::EventsData& Events() {return traceFile_->Events();}

    PacBio::TraceFile::ScanData& Scan(){return traceFile_->Scan(); }

    PacBio::TraceFile::TraceData& Traces() {return traceFile_->Traces();}

    //GroundTruthData GroundTruth();

    void OutputSummary(std::ostream& stream) const override { stream << "TraceFileWriter"; }
private:
    std::unique_ptr<PacBio::TraceFile::TraceFile> traceFile_;
};

}}

#endif // MONGO_APPMODULES_TRACEFILEWRITER_H
