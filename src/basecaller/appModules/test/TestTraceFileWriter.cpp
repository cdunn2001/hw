#include <vector>
#include <gtest/gtest.h>

#include <pacbio/dev/gtest-extras.h>
#include <pacbio/dev/TemporaryDirectory.h>
#include <appModules/TraceFileWriter.h>

//#include <pacbio/tracefile/TraceFile.h>
//#include <pacbio/datasource/PacketLayout.h>
//#include <pacbio/datasource/MallocAllocator.h>
//#include <pacbio/sensor/RectangularROI.h>

TEST(TraceFileWriter,A)
{
    PacBio::Dev::TemporaryDirectory tempdir;
    std::string traceFileName = tempdir.DirName() + "/trace.trc.h5";
    const uint32_t numZmws = 32;
    std::unique_ptr<PacBio::Application::DataFileWriterInterface> writer = std::make_unique<PacBio::Application::TraceFileWriter>(traceFileName, numZmws);

    std::stringstream ss;
    writer->OutputSummary(ss);
    TEST_COUT << ss.str() << std::endl;
}
