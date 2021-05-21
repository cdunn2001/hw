//
// Created by jnguyen on 9/6/17.
//

#ifndef PPA_TESTUTILS_H
#define PPA_TESTUTILS_H

#include <string>
#include <vector>
#include <sstream>

#include <bazio/RegionLabel.h>
#include <bazio/RegionLabelType.h>

using namespace PacBio::Primary;

std::vector<std::string> Split(const std::string& s, char delim)
{
    std::vector<std::string> elems;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

RegionLabel ReadHqRegions(const std::string& s)
{
    std::vector<std::string> parts = Split(s, '/');
    std::string hqRegion = parts[2];
    auto readPos = Split(hqRegion, '_');
    size_t hqStart = std::stoi(readPos[0]);
    size_t hqEnd = std::stoi(readPos[1]);
    return RegionLabel(hqStart, hqEnd, 0, RegionLabelType::HQREGION);
}


#endif //PPA_TESTUTILS_H
