#ifndef DASHBOARD_WRAPPER_H
#define DASHBOARD_WRAPPER_H

#include <iostream>
#include <string>

extern char _binary_dashboard_html_start;
extern size_t _binary_dashboard_html_size;

inline std::string GetDashboardString()
{
    const char* begin = &_binary_dashboard_html_start;
    size_t len = reinterpret_cast<size_t>(&_binary_dashboard_html_size);
    std::string s(begin,len);
    return s;
}

#endif

