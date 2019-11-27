// Copyright (c) 2015, Pacific Biosciences of California, Inc.
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
/// \brief  Implementation of ROI handler, including rectangular and sparse variants


#include <string>

#include <json/json.h>

#include <pacbio/POSIX.h>
#include <pacbio/text/String.h>
#include <pacbio/ipc/JSON.h>
#include <pacbio/primary/Tile.h>
#include <pacbio/logging/Logger.h>

#include <pacbio/primary/SequelROI.h>
#include <pacbio/primary/ChipLayout.h>

namespace PacBio
{
 namespace Primary
 {

  std::ostream&   operator<<(std::ostream& s, const PacBio::Primary::SequelSensorROI& roi)
  {
      roi.ToStream(s);
      return s;
  }

 SequelSensorROI::SequelSensorROI(uint32_t rowMin, uint32_t colMin, uint32_t numRows, uint32_t numCols,
         const ChipLayout* chiplayout) :
         SequelSensorROI(rowMin,colMin,numRows,numCols,
                         chiplayout->GetSensorROI().NumPixelRowsPerZmw(),
                         chiplayout->GetSensorROI().NumPixelColsPerZmw())
 {

 }

  void SequelSensorROI::ToStream(std::ostream& s) const
  {
      s << "region:(" << physicalRowOffset_ << "," << physicalColOffset_ << ") - (" << PhysicalRowMax() << "," << PhysicalColMax() << ") size:(" <<
                      physicalRows_ << "," << physicalCols_ << ")";

      s << " unitcell:(" << NumPixelRowsPerZmw() <<" x " << NumPixelColsPerZmw() << ")";
  }

  std::string SequelSensorROI::ToString() const
  {
      std::stringstream ss;
      ToStream(ss);
      return ss.str();
  }

  SequelRectangularROI SequelROI::Null()
  {
      return SequelRectangularROI(0, 0, 0, 0, SequelSensorROI(0, 0, 0, 0, 0 ,0));
  }

  SequelRectangularROI::SequelRectangularROI(const Json::Value& jsonSpecification, const SequelSensorROI& sensorROI) :
          SequelRectangularROI(jsonSpecification[0].asUInt(), jsonSpecification[1].asUInt(), jsonSpecification[2].asUInt(), jsonSpecification[3].asUInt(), sensorROI)
  {
  }

  SequelRectangularROI::SequelRectangularROI(RowPixels rowMin0, ColPixels colMin0, RowPixels numRows0, ColPixels numCols0, const SequelSensorROI& sensorRoi0)
      : SequelROI(sensorRoi0),
      rowMin_(rowMin0),
      colMin_(colMin0),
      rowMax_(rowMin0 + numRows0),
      colMax_(colMin0 + numCols0),
      numRows_(numRows0),
      numCols_(numCols0)
  {
      everything_ = (
              rowMin_.Value() == sensorRoi_.PhysicalRowOffset() &&
              colMin_.Value() == sensorRoi_.PhysicalColOffset() &&
              rowMax_.Value() == sensorRoi_.PhysicalRowMax() &&
              colMax_.Value() == sensorRoi_.PhysicalColMax()
      );

      if (NumPixelRows() > SensorROI().PhysicalRows() ||
          NumPixelCols() > SensorROI().PhysicalCols())
      {
          PBLOG_INFO << *this;
          throw PBException("ROI larger than physical (sensor) size");
      }
      sensorRoi_.CheckProposedROI(rowMin_,colMin_,numRows_,numCols_);

  }
  bool SequelRectangularROI::ContainsTileOffset(uint32_t tileOffset, uint32_t* internalOffset) const
  {
      if (everything_)
      {
          // shortcut if the ROI is the same as the SensorROI
          if (internalOffset) *internalOffset = tileOffset;
          return true;
      }

      int32_t tileStartingRow = tileOffset / (sensorRoi_.PhysicalCols() / Tile::NumPixels);
      int32_t tileStartingCol = (tileOffset % (sensorRoi_.PhysicalCols() / Tile::NumPixels)) * Tile::NumPixels;
      bool inside (tileStartingRow >= RelativeRowPixelMin() && tileStartingRow < RelativeRowPixelMax() &&
              tileStartingCol >= RelativeColPixelMin() && tileStartingCol < RelativeColPixelMax());

      if (inside && internalOffset)
      {
          const uint32_t roiWidthInPixels = RelativeColPixelMax()-RelativeColPixelMin();
          const uint32_t roiWidthInTiles = roiWidthInPixels / Tile::NumPixels;
#if 0
          PBLOG_INFO << "tileStartingCol:" << tileStartingCol << " RelativeColPixelMin():" << RelativeColPixelMin() <<
                            " tileStartingRow:" << tileStartingRow << " RelativeRowPixelMin:" << RelativeRowPixelMin() <<
                            " roiWidthInTiles:" << roiWidthInTiles ; //<< std::endl;
#endif
          *internalOffset = (tileStartingCol - RelativeColPixelMin())/ Tile::NumPixels +
                  (tileStartingRow - RelativeRowPixelMin()) * roiWidthInTiles;
      }
      return inside;
  }

  std::ostream& operator<<(std::ostream& s, const SequelROI& roi)
  {
      roi.ToStream(s);
      return s;
  }

  std::ostream& operator<<(std::ostream& s, const RowPixels x)
  {
      return s << x.Value() << "pix";
  }

  std::ostream& operator<<(std::ostream& s, const ColPixels x)
  {
      return s << x.Value() << "pix";
  }

  std::string SequelRectangularROI::ToString() const
  {
      std::stringstream ss;
      ToStream(ss);
      return ss.str();
  }

  void SequelRectangularROI::ToStream(std::ostream& s) const
  {
      {
          s << "region:(" << rowMin_ << "," << colMin_ << ") - (" << rowMax_ << "," << colMax_ << ") size:(" <<
          numRows_ << "," << numCols_ << ")";
          s << " (Sensor ROI:" << SensorROI() << ")";
      }
  }

  void SequelRectangularROI::CheckROI() const
  {
      if (RelativeRowPixelMin() < 0)
          throw PBException("ROI min pixel row is out of range: " + std::to_string(RelativeRowPixelMin()));
      if (RelativeColPixelMin() < 0)
          throw PBException("ROI min pixel col is out of range: " + std::to_string(RelativeColPixelMin()));
      if (RelativeRowPixelMax() > static_cast<int32_t>(SensorROI().PhysicalRows()))
          throw PBException("ROI max pixel row is out of range: " + std::to_string(RelativeRowPixelMax()) + ", is limited to <= " +
                            std::to_string(static_cast<int32_t>(SensorROI().PhysicalRows())));
      if (RelativeColPixelMax() > static_cast<int32_t>(SensorROI().PhysicalCols()))
          throw PBException("ROI max pixel col is out of range: " + std::to_string(RelativeColPixelMax()));
      if (SensorROI().PhysicalColOffset() % Tile::NumPixels)
          throw PBException("ROI phys pixel col is not divisible by 32: " + std::to_string(SensorROI().PhysicalColOffset()));
      if (NumPixelCols() % Tile::NumPixels)
          throw PBException("ROI NumPixelCols is not divisible by 32: " + std::to_string(NumPixelCols()));
      if (RelativeColPixelMin() % Tile::NumPixels)
          throw PBException("ROI min pixel col is not divisible by 32: " + std::to_string(RelativeColPixelMin()));
      if (RelativeColPixelMax() % Tile::NumPixels)
          throw PBException("ROI max pixel col is not divisible by 32: " + std::to_string(RelativeColPixelMax()));
  };


 void SequelSensorROI::CheckProposedROI(RowPixels minRow, ColPixels minCol, RowPixels numRows, ColPixels numCols) const
 {
     if (minCol.Value() % columnModulo_)
         throw PBException("SequelSensorROI requires minCol to be a multiple of " +
                                        std::to_string(columnModulo_) + " pixels (16 ZMWs), was " +
                           std::to_string(minCol.Value()));
     if (numCols.Value() % columnModulo_)
         throw PBException("SequelSensorROI requires numCols to be a multiple of "  +
                           std::to_string(columnModulo_) + " pixels (16 ZMWs), was " +
                           std::to_string(numCols.Value()));
     if (minRow.Value() < PhysicalRowOffset())
         throw PBException(
                 "ProposedROI minRow is lower than Sensor ROI: minRow=" + std::to_string(minRow.Value()) +
                 " physRowOffset=" + std::to_string(PhysicalRowOffset()));
     if (minRow.Value() + numRows.Value() > PhysicalRowMax())
     {
         std::stringstream ss;
         ss << "ProposedROI maxRow is greater than Sensor ROI physicalRowMax:"
                 << " minRow=" << minRow.Value()
                 << " maxRow=" << (minRow.Value() + numRows.Value())
                 << " physRowMax=" << PhysicalRowMax();
         throw PBException(ss.str());
     }
     if (minCol.Value() < PhysicalColOffset())
         throw PBException("ProposedROI minCol is lower than Sensor ROI: minCol=" + std::to_string(minCol.Value()));

     uint32_t maxCol = minCol.Value() + numCols.Value();
     if (maxCol > PhysicalColMax())
         throw PBException("ProposedROI maxCol (" +
                           std::to_string(maxCol) +
                           ") is greater than Sensor ROI (" +
                           std::to_string(PhysicalColMax())
                           + ")");
}


  SequelROI* SequelRectangularROI::Clone() const
  {
      return new SequelRectangularROI(*this);
  }

  Json::Value SequelRectangularROI::GetJson() const
  {
      Json::Value rect;
      rect.append(AbsoluteRowPixelMin());
      rect.append(AbsoluteColPixelMin());
      rect.append(NumPixelRows());
      rect.append(NumPixelCols());

      return rect;
  }

  bool SequelRectangularROI::operator==(const SequelROI& a) const
  {
      if (a.Type() != ROI_Type_e::Rectangular) return false;

      const SequelRectangularROI* aa = dynamic_cast<const SequelRectangularROI*>(&a);
      return rowMin_ == aa->rowMin_ &&
              colMin_ == aa->colMin_ &&
              rowMax_ == aa->rowMax_ &&
              colMax_ == aa->colMax_ &&
              numRows_ == aa->numRows_ &&
              numCols_ == aa->numCols_ ;
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



  SequelSparseROI::SequelSparseROI(const SequelSensorROI& sensorROI) :
      SequelROI(sensorROI),
      bitmask_(sensorROI.TotalPixels()/32,false), // 32 pixels per SIMD word
      internalOffsets_(sensorROI.TotalPixels()/32,-1),
      totalPixels_(0),
      everything_(false),
      dirty_(true)
  {
  }

  SequelSparseROI::SequelSparseROI(const SequelSparseROI& a) :
      SequelROI(a.SensorROI()),
      bitmask_(a.bitmask_),
      internalOffsets_(a.internalOffsets_),
      totalPixels_(a.totalPixels_),
      everything_(a.everything_),
      json_(a.json_),
      dirty_(true)
  {

  }

  SequelSparseROI::SequelSparseROI(const std::string& jsonSpecification, const SequelSensorROI& sensorROI) :
          SequelSparseROI(sensorROI)
  {
      if (jsonSpecification == "all")
      {
          SelectAll();
      }
      else
      {
          std::string jsonString;
          if (PacBio::POSIX::IsFile(jsonSpecification))
          {
              jsonString = PacBio::Text::String::Slurp(jsonSpecification);
          }
          else
          {
              jsonString = jsonSpecification;
          }
          const Json::Value json = PacBio::IPC::ParseJSON(jsonString);
          ImportJson(json);
      }
  }

  SequelSparseROI::SequelSparseROI(const Json::Value& jsonSpecification, const SequelSensorROI& sensorROI) :
          SequelSparseROI(sensorROI)
  {
      ImportJson(jsonSpecification);
  }

void SequelSparseROI::SelectAll()
{
    dirty_ = true;

    AddRectangle(0,0,SensorROI().PhysicalRows(),SensorROI().PhysicalCols());
}

    void SequelSparseROI::ImportJson(const Json::Value& jsonSpecification)
  {
      dirty_ = true;

      if  (jsonSpecification.size() == 4 &&
              jsonSpecification[0].isNumeric() )
      {
          AddRectangle(jsonSpecification[0].asUInt(), jsonSpecification[1].asUInt(), jsonSpecification[2].asUInt(), jsonSpecification[3].asUInt());
      }
      else
      {
          for (const Json::Value& rect : jsonSpecification)
          {
              AddRectangle(rect[0].asUInt(), rect[1].asUInt(), rect[2].asUInt(), rect[3].asUInt());
          }
      }
  }

  void SequelSparseROI::AddRectangle(RowPixels minRow, ColPixels minCol, RowPixels numRows, ColPixels numCols)
  {
      dirty_ = true;

      sensorRoi_.CheckProposedROI(minRow,minCol,numRows,numCols);

      const uint32_t rowOffset = minRow.Value() - sensorRoi_.PhysicalRowOffset();
      const uint32_t colOffset = minCol.Value() - sensorRoi_.PhysicalColOffset();

      for (uint32_t irow = 0; irow < numRows.Value(); irow++)
      {
          for (uint32_t icol = 0; icol < numCols.Value(); icol += 32)
          {
              uint32_t offset = (irow + rowOffset) * sensorRoi_.PhysicalCols() + (icol + colOffset);
              offset /= 32;
              if (bitmask_[offset] == false)
              {
                  bitmask_[offset] = true;
                  totalPixels_ += 32;
              }
          }
      }
      if (numRows.Value() > 0 && numCols.Value() > 0)
      {
          Json::Value rect;
          rect.append(minRow.Value());
          rect.append(minCol.Value());
          rect.append(numRows.Value());
          rect.append(numCols.Value());
          json_.append(rect);
      }
      everything_ = totalPixels_ == sensorRoi_.TotalPixels();
  }

  bool SequelSparseROI::Contains(const PixelCoord& coord) const
  {
      uint32_t pixelOffset = (coord.row.Value() - sensorRoi_.PhysicalRowOffset()) * sensorRoi_.PhysicalCols() + (coord.col.Value() - sensorRoi_.PhysicalColOffset());
      uint32_t tileOffset = pixelOffset / Tile::NumPixels;

      return ContainsTileOffset(tileOffset);
  }

  void SequelSparseROI::AddZMW(const PixelCoord& coord)
  {
      dirty_ = true;
      // truncate col to lane boundary, and select the whole lane
      uint32_t colTruncated = (coord.col.Value() - coord.col.Value() % sensorRoi_.PixelLaneWidth());
      AddRectangle(coord.row,colTruncated,1,sensorRoi_.PixelLaneWidth());
  }



  std::vector<SequelSparseROI::Rect> SequelSparseROI::CondensedRects() const
  {
      struct Segment
      {
          uint32_t minCol = 0;
          uint32_t maxCol = 0;
      };
      std::vector<Rect> completedRects;
      std::map<uint64_t, Rect> currentRects;
      for (uint32_t irow = 0; irow <= sensorRoi_.PhysicalRows(); irow++)
      {
          if (irow < sensorRoi_.PhysicalRows())
          {
              bool inrect = false;
              Segment currentSegment;
              for (uint32_t icol = 0; icol <= sensorRoi_.PhysicalCols(); icol += 32)
              {
                  bool pixel;
                  if (icol < sensorRoi_.PhysicalCols())
                  {
                      uint32_t offset = irow * sensorRoi_.PhysicalCols() + icol;
                      offset /= 32;
                      pixel = bitmask_[offset];
                  }
                  else
                  {
                      pixel = false;
                  }

                  if (inrect)
                  {
                      if (pixel)
                      {
                          currentSegment.maxCol = icol + 32;
                      }
                      else
                      {
                          uint64_t hash = (uint64_t) currentSegment.maxCol << 32 | currentSegment.minCol;
                          auto&& cr = currentRects.find(hash);
                          if (cr != currentRects.end())
                          {
                              cr->second.maxRow = irow + 1;
                          }
                          else
                          {
                              PBLOG_TRACE << "starting new rect at " << irow << " " << currentSegment.minCol <<
                                              " " << currentSegment.maxCol;
                              Rect newRect;
                              newRect.minRow = irow;
                              newRect.maxRow = irow + 1;
                              newRect.minCol = currentSegment.minCol;
                              newRect.maxCol = currentSegment.maxCol;
                              currentRects[hash] = newRect;
                          }
                          inrect = false;
                      }
                  }
                  else
                  {
                      if (pixel)
                      {
                          PBLOG_TRACE << "starting new seg at " << icol;
                          currentSegment.minCol = icol;
                          currentSegment.maxCol = icol + 32;
                          inrect = true;
                      }
                      else
                      {
                          // do nothing
                      }
                  }
              }
          }
          for (auto r = currentRects.begin(); r != currentRects.end();)
          {
              if (r->second.maxRow != irow + 1)
              {
                  PBLOG_TRACE << "flushing completed rect:" << r->second.minRow << " " << r->second.minCol;
                  completedRects.push_back(r->second);
                  currentRects.erase(r++);
              }
              else
              {
                  ++r;
              }
          }
      }
      return completedRects;
  }

  Json::Value SequelSparseROI::Condense() const
  {
      std::vector<Rect> completedRects = CondensedRects();

      Json::Value newJson;
      newJson.clear();

      PBLOG_TRACE << "new rects: " << completedRects.size();
      for (auto&& r : completedRects)
      {
          Json::Value rect;
          rect.append(r.minRow);
          rect.append(r.minCol);
          rect.append(r.maxRow - r.minRow);
          rect.append(r.maxCol - r.minCol);
          newJson.append(rect);
      }
      return newJson;
  }

  void SequelSparseROI::PostAddRectangle()
  {
      json_ = Condense();
  }

  void SequelSparseROI::Recache() const
  {
      absRowPixelMin_ = 0xFFFFFFFF;
      absColPixelMin_ = 0xFFFFFFFF;
      std::vector<Rect> rects = SequelSparseROI::CondensedRects();
      for(auto&r : rects)
      {
          if (r.minRow < absRowPixelMin_) absRowPixelMin_ = r.minRow;
          if (r.minCol < absColPixelMin_) absColPixelMin_ = r.minCol;
      }

      int32_t internalOffset = 0;
      for(auto& i : internalOffsets_) i = -1;
      const uint32_t maxTileOffset = sensorRoi_.TotalPixels()/Tile::NumPixels;
      for(uint32_t tileOffset = 0; tileOffset < maxTileOffset; tileOffset++)
      {
          if (ContainsTileOffset(tileOffset))
          {
              internalOffsets_[tileOffset] = internalOffset;
              internalOffset++;
          }
      }

      dirty_ = false;
  }

  uint32_t SequelSparseROI::AbsoluteRowPixelMin() const
  {
      if (dirty_)
      {
          Recache();
      }
      return absRowPixelMin_;
  }
  uint32_t SequelSparseROI::AbsoluteColPixelMin() const
  {
      if (dirty_)
      {
          Recache();
      }
      return absColPixelMin_;
  }

  uint32_t SequelSparseROI::TotalPixels() const
  {
      return totalPixels_;
  }

  bool SequelSparseROI::Everything() const
  {
      return everything_;
  }

  bool SequelSparseROI::ContainsTileOffset(uint32_t tileOffset, uint32_t* internalOffset) const
  {
      bool inside = bitmask_[tileOffset];
      if (internalOffset && inside)
      {
          if (dirty_)
          {
              Recache();
          }
          int32_t i = internalOffsets_[tileOffset];
          if (i<0) throw PBException("internal inconsistency");
          *internalOffset = static_cast<uint32_t>(i);
      }
      return inside;
  }

  void SequelSparseROI::ToStream(std::ostream& s) const
  {
      s << "SequelSparseROI, totalPixels:" << totalPixels_ << " (Sensor ROI:" << SensorROI() << ")";
      s <<  "\n" + PacBio::IPC::RenderJSON(Condense());
  }

  std::string SequelSparseROI::ToString() const
  {
      return "SequelSparseROI, totalPixels:" + std::to_string(totalPixels_) ;
  }
  void SequelSparseROI::CheckROI() const
  {
      if (SensorROI().PhysicalColOffset() % Tile::NumPixels)
          throw PBException("ROI phys pixel col is not divisible by 32: " + std::to_string(SensorROI().PhysicalColOffset()));
  }
  SequelROI* SequelSparseROI::Clone() const
  {
    return new SequelSparseROI(*this);
  }

  bool SequelSparseROI::operator==(const SequelSparseROI& other) const
  {
      return TotalPixels() == other.TotalPixels() &&
             Everything() == other.Everything() &&
             bitmask_ == other.bitmask_;
  }

  Json::Value SequelSparseROI::GetJson() const
  {
      return json_;
  }

  bool SequelSparseROI::operator==(const SequelROI& a) const
  {
      if (a.Type() != ROI_Type_e::Sparse) return false;

      return (*this) == dynamic_cast<const SequelSparseROI&>(a);
  }

  SequelSparseROI SequelSparseROI::Intersect(SequelSparseROI& b) const
  {
      SequelSparseROI intersection(sensorRoi_);
      if (bitmask_.size() != b.bitmask_.size()) throw PBException("ROIs have different sensor ROIs");

      intersection.bitmask_ = bitmask_;

      for (uint32_t i = 0; i < bitmask_.size(); i++)
      {
          intersection.bitmask_[i] = intersection.bitmask_[i] && b.bitmask_[i];
          intersection.totalPixels_ += (intersection.bitmask_[i] ? 32 : 0);
      }
      intersection.json_ = Condense();
      intersection.everything_ = intersection.totalPixels_ == intersection.sensorRoi_.TotalPixels();
      return intersection;
  }
 }
}
