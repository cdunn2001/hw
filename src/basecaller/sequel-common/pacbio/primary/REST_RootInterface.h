//
// Created by mlakata on 2/8/18.
//

#ifndef SEQUELACQUISITION_REST_ROOTINTERFACE_H
#define SEQUELACQUISITION_REST_ROOTINTERFACE_H

#include <pacbio/POSIX.h>
#include <pacbio/process/RESTDataModel.h>
#include <pacbio/process/RESTServer.h>

namespace PacBio {
namespace Process {
class RESTServer;

}}

void REST_StartServer();
PacBio::Process::RESTServer& GetRESTServer();
#define stringify(x) stringify2(x)
#define stringify2(x) #x
#define GetREST_Root() GetREST_RootAt(__FILE__  ":" stringify(__LINE__) )
PacBio::Process::RESTDataModelLock GetREST_RootAt(const char* sourceLine);

inline void REST_StartServerBlocking()
{
    REST_StartServer();

    PBLOG_INFO << "GetRESTServer().server is " << (void*) GetRESTServer().server << std::endl;
    uint32_t iterations = 0;
    while(GetRESTServer().server == nullptr)
    {
        PBLOG_INFO << "waiting for server ..."  << std::endl;
        PacBio::POSIX::Sleep(0.5);
        if (iterations > 100) throw PBException("Waited too long for REST server to boot up!");
        iterations++;
    }
    PacBio::POSIX::Sleep(0.5);
}

#endif //SEQUELACQUISITION_REST_ROOTINTERFACE_H
