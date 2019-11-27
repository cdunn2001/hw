
// Copyright (c) 2014-2018, Pacific Biosciences of California, Inc.
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
// File Description:
/// \brief  a class used to describe the Sequel chip layout
//
// Programmer: Mark Lakata

#include <pacbio/primary/ChipLayout.h>

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <fstream>
#include <iostream>

#include <pacbio/text/imemstream.h>
#include <pacbio/primary/SequelMovie.h>
#include <pacbio/primary/SequelROI.h>
#include <pacbio/primary/Sparse2DArray.h>
#include <pacbio/primary/ChipLayoutRTO2.h>
#include <pacbio/primary/ChipLayoutRTO3.h>
#include <pacbio/primary/ChipLayoutSpider1.h>
#include <pacbio/primary/ChipLayoutSpider1p0NTO.h>
#include <pacbio/primary/ChipLayoutBenchyDemo.h>
#include <pacbio/primary/Platform.h>

namespace PacBio {
namespace Primary {

  int Excel::GetCol(const char* excelLabel)
  {
      int val = 0;
      int pos = 0;
      for(const char* ptr = excelLabel + strlen(excelLabel) - 1; ptr >= excelLabel; ptr--)
      {
          if (*ptr >= 'A' && *ptr <= 'Z')
          {
              int i = *ptr - 'A' + 1;
              if (pos == 0)
              {
                  val = i;
              }
              else if(pos == 1)
              {
                  val += i* 26;
              } else if (pos == 2)
              {
                  val += i * 26 * 26;
              }
              else
              {
                  throw PBException("too many letters " + std::string(excelLabel));
              }
              pos ++;
          }
      }
      if (val == 0) throw PBException("no letters "  + std::string(excelLabel));
      return val;
  }

  int Excel::GetRow(const char* excelLabel)
  {
      for(const char* ptr = excelLabel ; *ptr != 0; ptr++)
      {
          if (*ptr >= 'A' && *ptr <= 'Z')
          {
          }
          else
          {
              return atoi(ptr);
          }
      }
      throw PBException("not a number " + std::string(excelLabel));
  }

  // excel is 1-based, UnitCells are 0-based
  UnitCell::UnitCell(const char* excelLabel) :
          UnitCell(Excel::GetRow(excelLabel)-1,
                   Excel::GetCol(excelLabel)-1)
  {
  }

  std::string UnitCell::ExcelCell() const
  {
      char s[4];
      int y0= y;
      if (y0 < 26)
      {
          s[0] = (char)('A' + y0);
          s[1] = 0;
      }
      else {
          y0 -= 26;
          if (y0 < 26*26)
          {
              s[0] = (char)('A' + y0/26) ;
              s[1] = (char)('A' + y0 % 26);
              s[2] = 0;
          }
          else
          {
              y0 -= 26*26;
              s[0] = (char)('A' + y0/(26*26)) ;
              s[1] = (char)('A' + (y0 % (26*26)) / 26);
              s[2] = (char)('A' + (y0 % (26*26)) % 26);
              s[3] = 0;
          }
      }
      return s + std::to_string(x+1);
  }


  ChipLayout::~ChipLayout()
  {
  }

  std::vector<UnitCell> ChipLayout::GetUnitCellListByIntType(int unitType, const SequelROI& roi)
  {
      std::vector<UnitCell> results;

      size_t numZmws = roi.CountZMWs();
      for(SequelROI::Enumerator e(roi,0,numZmws);e;e++)
      {
          std::pair<RowPixels, ColPixels> coord = e.UnitCellPixelOrigin();
          UnitCell cell(ConvertAbsoluteRowPixelToUnitCellX(coord.first),
                        ConvertAbsoluteColPixelToUnitCellY(coord.second));

          auto thisType = GetUnitCellIntType(cell);
          if (thisType == unitType)
          {
              results.push_back(cell);
          }
      }

      return results;
  }

  std::vector<UnitCell> ChipLayout::GetUnitCellListByPredicate(std::function<bool(const UnitCell&)> predicate, const SequelROI& roi) const
  {
      std::vector<UnitCell> results;

      size_t numZmws = roi.CountZMWs();
      for(SequelROI::Enumerator e(roi,0,numZmws);e;e++)
      {
          std::pair<RowPixels, ColPixels> coord = e.UnitCellPixelOrigin();
          UnitCell cell(ConvertAbsoluteRowPixelToUnitCellX(coord.first),
                        ConvertAbsoluteColPixelToUnitCellY(coord.second));

          if (predicate(cell))
          {
              results.push_back(cell);
          }
      }

      return results;
  }

  std::vector<UnitCell> ChipLayout::GetUnitCellList( const SequelROI& roi) const
  {
      std::vector<UnitCell> results;

      size_t numZmws = roi.CountZMWs();
      for(SequelROI::Enumerator e(roi,0,numZmws);e;e++)
      {
          std::pair<RowPixels, ColPixels> coord = e.UnitCellPixelOrigin();
          UnitCell cell(ConvertAbsoluteRowPixelToUnitCellX(coord.first),
                        ConvertAbsoluteColPixelToUnitCellY(coord.second));
          results.push_back(cell);
      }

      return results;
  }

  SequelRectangularROI ChipLayout::GetROIFromUnitCellRectangle(UnitX xMin, UnitY yMin, UnitX xHeight, UnitY yWidth)
  {
      int32_t pixelRowMin;
      int32_t pixelColMin = (yMin.Value() - params.unitCellOffsetToSensorY_) * params.colPixelsPerZmw_;

      uint32_t numPixelRows = static_cast<uint32_t>(std::abs(xHeight.Value() * params.rowPixelsPerZmw_));
      uint32_t numPixelCols = static_cast<uint32_t>(std::abs(yWidth.Value() * params.colPixelsPerZmw_));
      if (params.rowPixelsPerZmw_ < 0)
      {
          // invert roi. Use xMax instead of xMin
          pixelRowMin = (xMin.Value()+xHeight.Value()-1 - params.unitCellOffsetToSensorX_) * params.rowPixelsPerZmw_;
      }
      else
      {
          pixelRowMin = (xMin.Value() - params.unitCellOffsetToSensorX_) * params.rowPixelsPerZmw_;
      }
      if (params.colPixelsPerZmw_ < 0)
      {
          throw PBException("not supported");
      }
      if (pixelRowMin <0 || pixelColMin < 0)
      {
          throw PBException("Negative pixel coordinates");
      }
      return SequelRectangularROI(static_cast<uint32_t>(pixelRowMin),
                                  static_cast<uint32_t>(pixelColMin),
                                  numPixelRows, numPixelCols, GetSensorROI());
  }

  UnitX ChipLayout::ConvertAbsoluteRowPixelToUnitCellX(RowPixels x) const
  {
      int32_t unitX = static_cast<int32_t>(x.Value()) / params.rowPixelsPerZmw_ + params.unitCellOffsetToSensorX_;
      if (unitX < params.minUnitCellX_ || unitX > params.maxUnitCellX_)
          throw PBException("x coordinate out of unit cell range: " + std::to_string(x.Value()) );
      return unitX;
  }


  UnitY  ChipLayout::ConvertAbsoluteColPixelToUnitCellY(ColPixels y) const
  {
      int32_t unitY = static_cast<int32_t>(y.Value()) / params.colPixelsPerZmw_ + params.unitCellOffsetToSensorY_;
      if (unitY < params.minUnitCellY_ || unitY > params.maxUnitCellY_)
          throw PBException("y coordinate out of unit cell range: pixelY:" + std::to_string(y.Value()) +
                            ", unitY:" + std::to_string(unitY) +
                            ", maxUnitCellY:" + std::to_string(params.maxUnitCellY_));
      return unitY;
  }

  UnitX ChipLayout::ConvertRelativeRowPixelToUnitCellX(RowPixels x) const
  {
      int32_t unitX = static_cast<int32_t>(x.Value())  / params.rowPixelsPerZmw_;
      return unitX;
  }


  UnitY  ChipLayout::ConvertRelativeColPixelToUnitCellY(ColPixels y) const
  {
      uint32_t unitY = static_cast<int32_t>(y.Value()) / params.colPixelsPerZmw_ ;
      return unitY;
  }


  std::vector<std::pair<int,UnitCell>> ChipLayout::GetUnitCellIntTypeList(const SequelROI& roi) const
  {
     std::vector<std::pair<int,UnitCell>> results;
     size_t numZmws = roi.CountZMWs();
     for(SequelROI::Enumerator e(roi,0,numZmws);e;e++)
     {
         std::pair<RowPixels, ColPixels> coord = e.UnitCellPixelOrigin();
         UnitCell cell(ConvertAbsoluteRowPixelToUnitCellX(coord.first),
                       ConvertAbsoluteColPixelToUnitCellY(coord.second));

         results.push_back(std::make_pair(GetUnitCellIntType(cell),cell));
     }
     return results;
  }

  std::vector<std::pair<ChipLayout::UnitFeature, UnitCell>>
  ChipLayout::GetUnitCellFeatureList(const SequelROI& roi) const
  {
      std::vector<std::pair<UnitFeature, UnitCell>> output;

      auto l = GetUnitCellList(roi);
      for( auto& uc : l)
      {
          output.push_back( std::make_pair(GetUnitCellFeatures(uc),uc));
      }
      return output;
  }


  std::unique_ptr<ChipLayout> ChipLayout::Factory(const std::string& layoutName)
  {
      if (layoutName == "")
      {
          throw PBException("ChipLayout::Factory can not create layout if not specified or if set to \"\".");
      }
      using sup = std::unique_ptr<ChipLayout>;
      if (layoutName == ChipLayoutRTO2::ClassName())
      {
          return sup(new ChipLayoutRTO2());
      }
      if (layoutName == ChipLayoutRTO3::ClassName())
      {
          return sup(new ChipLayoutRTO3());
      }
      if (layoutName == ChipLayoutSpider1::ClassName())
      {
          return sup(new ChipLayoutSpider1());
      }
      if (layoutName == ChipLayoutSpider1p0NTO::ClassName())
      {
          return sup(new ChipLayoutSpider1p0NTO());
      }
      if (layoutName == ChipLayoutBenchyDemo::ClassName())
      {
          return sup(new ChipLayoutBenchyDemo());
      }
      throw PBException("ChipLayout::Factory can't produce " + layoutName);
  }

  const std::vector<std::string>& ChipLayout::GetLayoutNames()
  {
      static std::vector<std::string> layoutNames;
      if (layoutNames.size() == 0)
      {
          layoutNames.push_back(ChipLayoutRTO2::ClassName());
          layoutNames.push_back(ChipLayoutRTO3::ClassName());
          layoutNames.push_back(ChipLayoutSpider1::ClassName());
          layoutNames.push_back(ChipLayoutSpider1p0NTO::ClassName());
          layoutNames.push_back(ChipLayoutBenchyDemo::ClassName());
      }
      return layoutNames;
  }

  SequelSensorROI ChipLayout::GetSensorROI(Platform platform)
  {
      switch(platform)
      {
      case PacBio::Primary::Platform::Sequel1PAC1:
      case PacBio::Primary::Platform::Sequel1PAC2:
          return SequelSensorROI(0, 0, Sequel::maxPixelRows, Sequel::maxPixelCols, Sequel::numPixelRowsPerZmw,
                                 Sequel::numPixelColsPerZmw);
      case PacBio::Primary::Platform::Spider:
          return SequelSensorROI(0, 0, Spider::maxPixelRows, Spider::maxPixelCols, Spider::numPixelRowsPerZmw,
                                 Spider::numPixelColsPerZmw);
      case PacBio::Primary::Platform::UNKNOWN:
          throw PBException("Platform was not defined, can't determined SensorROI ");
      default:
          throw PBException("Unsupported Platform: " + platform.toString());
      }
  }

  ChipLayout::ChipLayout(const Params& initializer) : params(initializer)
  {
      if (params.colPixelsPerZmw_ == 0) throw PBException("bad colPixelsPerZmw, was zero");
      if (params.rowPixelsPerZmw_ == 0) throw PBException("bad rowPixelsPerZmw_,was zero");
  }

}}  // PacBio::Primary
