#include <iostream>
#include <string>

extern char _binary_@FILENAME@_end;
extern char _binary_@FILENAME@_start;
extern size_t _binary_@FILENAME@_size;

// TODO. replace with string_view when C++17 becomes standard
std::string GetDocString()
{
    const char* begin = &_binary_@FILENAME@_start;
    size_t len = reinterpret_cast<size_t>(&_binary_@FILENAME@_size);

    std::string s(begin,len);
    return s;
}
