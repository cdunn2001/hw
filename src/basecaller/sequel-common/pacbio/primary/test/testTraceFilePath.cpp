//
// Created by jnguyen on 11/3/16.
//

#include <boost/filesystem.hpp>


std::string TraceFilePath(const std::string& fn)
{
    std::string roots[] = {
        "/pbi",
        "/home/pbi"
    };

    for (const std::string& r : roots)
    {
        boost::filesystem::path dir(r);
        boost::filesystem::path file(fn);
        boost::filesystem::path p = dir / file;
        if (boost::filesystem::exists(p))
            return p.string();
    }

    return (boost::filesystem::path(roots[0]) / boost::filesystem::path(fn)).string();
}
