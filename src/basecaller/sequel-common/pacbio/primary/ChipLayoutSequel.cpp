// Copyright (c) 2018, Pacific Biosciences of California, Inc.
//
// All rights reserved.
//
// THIS SOFTWARE CONSTITUTES AND EMBODIES PACIFIC BIOSCIENCES' CONFIDENTIAL
// AND PROPRIETARY INFORMATION.
//
// Disclosure, redistribution and use of this software is subject to the
// terms and conditions of the applicable written agreement(s) between you
// and Pacific Biosciences, where "you" refers to you or your company or
// organization, as applicable.  Any other disclosure, redistribution or
// use is prohibited.
//
// THIS SOFTWARE IS PROVIDED BY PACIFIC BIOSCIENCES AND ITS CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL PACIFIC BIOSCIENCES OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
// OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
// ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//  Description:
//  Defines members of class ChipLayoutSequel.

#include "ChipLayoutSequel.h"

#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/copy.hpp>
#ifdef WIN32
#else
#include <boost/iostreams/filter/gzip.hpp>
#endif

namespace PacBio {
namespace Primary {

// Define member functions.
ChipLayoutSequel::ChipLayoutSequel() : ChipLayout([](){
    Params r;
    r.minUnitCellX_ = MinUnitCellX;
    r.minUnitCellY_ = MinUnitCellY;
    r.maxUnitCellX_ = MaxUnitCellX;
    r.maxUnitCellY_ = MaxUnitCellY;
    r.rowPixelsPerZmw_ = 1;
    r.colPixelsPerZmw_ = 2;
    r.unitCellOffsetToSensorX_ = UnitCellOffsetToSensorX;
    r.unitCellOffsetToSensorY_ = UnitCellOffsetToSensorY;
    return r;}())
{

}

ChipLayoutSequel::~ChipLayoutSequel()
{
    if (fullMatrix_) delete[] fullMatrix_;
}

void ChipLayoutSequel::InitFullMatrix()
{
    if (params.minUnitCellX_!=0 || params.minUnitCellY_ !=0)
    {
        throw PBException("Misconfiguration");
    }
    fullMatrix_ = new uint8_t[params.maxUnitCellX_ * params.maxUnitCellY_];
}


void ChipLayoutSequel::Load(std::istream& file)
{

#ifdef WIN32
        throw PBException("not implemented");
#else
    boost::iostreams::filtering_streambuf <boost::iostreams::input> inbuf;
    inbuf.push(boost::iostreams::gzip_decompressor());
    inbuf.push(file);
    //Convert streambuf to istream
    std::istream instream(&inbuf);

    std::string line;
    int32_t row = 0;
    int32_t col = 0;
    while (std::getline(instream, line))
    {
        std::stringstream lineStream(line);
        std::string cell;
        col = 0;
        while (std::getline(lineStream, cell, ','))
        {
            FullMatrix(row,col) = static_cast<uint8_t>(std::stoi(cell));
            col++;
        }
        row++;
    }
    if (row != params.maxUnitCellX_) throw PBException("unexpected number of rows: " + std::to_string(row));
    if (col != params.maxUnitCellY_) throw PBException("unexpected number of cols: " + std::to_string(col));

#endif
}

}}  // PacBio::Primary
