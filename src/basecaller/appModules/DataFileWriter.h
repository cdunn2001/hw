#ifndef PACBIO_APPLICATION_DATAFILEWRITER_H
#define PACBIO_APPLICATION_DATAFILEWRITER_H

#include <ostream>

namespace PacBio {
namespace Application    {

class DataFileWriterInterface
{
public:
    virtual ~DataFileWriterInterface() {}

    virtual void OutputSummary(std::ostream &stream) const = 0;
};

}} // namespaces

#endif
