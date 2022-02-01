//
// Created by mlakata on 4/27/15.
//

#include <cstdlib>
#include <pacbio/process/RESTServer.h>
#include <pacbio/POSIX.h>
#include <pacbio/primary/ipc_config.h>
#include <pacbio/logging/Logger.h>
#include <pacbio/primary/REST_RootInterface.h>

using namespace PacBio::Process;
using namespace PacBio::Primary;
using namespace PacBio;

class RESTServerEx : public RESTServer
{
public:
    RESTServerEx() : RESTServer(PORT_HTTP_PPAD, POSIX::GetCurrentWorkingDirectory())
    {
    }

    ~RESTServerEx()
    {
        {
            auto root = CheckoutRoot("~RESTServerEx");
            root->ModifyString("state", "offline");
        }
        POSIX::usleep(1500000);
    }
};

RESTServerEx restServer;


RESTDataModelLock GetREST_RootAt(const char* lineinfo)
{
    return restServer.CheckoutRoot(lineinfo);
}

void REST_StartServer()
{
    char* restServerRoot = getenv("REST_SERVER_ROOT");
    if (restServerRoot != nullptr)
    {
        restServer.fileSystemRoot = restServerRoot;
    }
    else
    {
#if 0
#ifdef NDEBUG
        restServer.fileSystemRoot = cmakeInstallPrefix + std::string("/ui");
#else
        restServer.fileSystemRoot = cmakeCurrentListDir + std::string("/ui");
#endif
#endif
    }

    restServer.Start();
    PBLOG_INFO << "restserver/webserver: " << restServer.GetRootURL() ;
}
