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
/// \brief  Declaration of ROI handler, including rectangular and sparse variants


#pragma once

#include <array>

#include <json/json.h>

#include <pacbio/PBException.h>
#include <pacbio/primary/UnitCell.h>
#include <pacbio/primary/Tile.h>
#include <pacbio/primary/SequelDefinitions.h>
#include <pacbio/utilities/SmartEnum.h>


namespace PacBio
{
 namespace Primary
 {
  class ChipLayout;
  class SequelRectangularROI;

  struct RowPixels
  {
      RowPixels(uint32_t x)
              : pixels_(x)
      { }

      RowPixels operator+(const RowPixels& a) const
      { return RowPixels(pixels_ + a.pixels_); }

      RowPixels& operator+=(const RowPixels& a)
      {  pixels_ = pixels_ + a.pixels_; return *this; }

      uint32_t Value() const
      { return pixels_; }

      bool operator==(RowPixels r) const
      {
          return pixels_ == r.pixels_;
      }
  private:
      uint32_t pixels_;
  };

  struct ColPixels
  {
      ColPixels(uint32_t x)
              : pixels_(x)
      { }

      ColPixels operator+(const ColPixels& a) const
      { return ColPixels(pixels_ + a.pixels_); }

      ColPixels& operator+=(const ColPixels& a)
      {  pixels_ = pixels_ + a.pixels_; return *this; }

      uint32_t Value() const
      { return pixels_; }

      bool operator==(ColPixels r)  const
      {
          return pixels_ == r.pixels_;
      }

  private:
      uint32_t pixels_;
  };

#if __INTEL_COMPILER >= 1500
  inline RowPixels operator
  ""

  _rowPixels(unsigned long long n)
  {
      return RowPixels(n);
  }

  inline RowPixels operator
  ""

  _rowZmws(unsigned long long n)
  {
      return RowPixels(n);
  }

  inline ColPixels operator
  ""

  _colPixels(unsigned long long n)
  {
      return ColPixels(n);
  }

  inline ColPixels operator
  ""

  _colZmws(unsigned long long n)
  {
      return ColPixels(n / 2);
  }

#endif

  struct PixelCoord
  {
      RowPixels row;
      ColPixels col;
      PixelCoord(RowPixels row0, ColPixels col0) : row(row0), col(col0) {}
  };
  inline bool operator==(const PixelCoord& a, const PixelCoord& b)
  {
      return a.row == b.row && a.col == b.col;
  }
  inline std::ostream& operator<<(std::ostream& s, const PixelCoord& p)
  {
      s << p.row.Value() << "," << p.col.Value();
      return s;
  }

  class SequelSensorROI
  {
  private:
      uint32_t physicalRowOffset_;
      uint32_t physicalColOffset_;
      uint32_t physicalRows_;
      uint32_t physicalCols_;
      uint32_t numPixelRowsPerZmw_;
      uint32_t numPixelColsPerZmw_;
      uint32_t columnModulo_;
  public:
      SequelSensorROI(uint32_t rowMin, uint32_t colMin, uint32_t numRows, uint32_t numCols,
        uint32_t numPixelRowsPerZmw, uint32_t numPixelColsPerZmw)
              :
              physicalRowOffset_(rowMin),
              physicalColOffset_(colMin),
              physicalRows_(numRows),
              physicalCols_(numCols),
              numPixelRowsPerZmw_(numPixelRowsPerZmw),
              numPixelColsPerZmw_(numPixelColsPerZmw),
              columnModulo_(Tile::NumPixels)
     { }
      SequelSensorROI(uint32_t rowMin, uint32_t colMin, uint32_t numRows, uint32_t numCols,
                      const PacBio::Primary::ChipLayout*);

      static SequelSensorROI POC()
      {
          return SequelSensorROI(0, 0, 1024, 512,1,2);
      }

      static SequelSensorROI SequelAlpha()
      {
          return SequelSensorROI(0, 0,
                                 SequelLayout::maxPixelRows, SequelLayout::maxPixelCols,
                                 Sequel::numPixelRowsPerZmw, Sequel::numPixelColsPerZmw);
      }
      static SequelSensorROI Spider()
      {
          auto roi = SequelSensorROI(0, 0,
                                 SpiderLayout::maxPixelRows, SpiderLayout::maxPixelCols,
                                 Spider::numPixelRowsPerZmw, Spider::numPixelColsPerZmw);
          roi.SetPixelLaneWidth(Tile::NumPixels);
          return roi;
      }

      /// Use this ROI to indicate an ROI that is not set yet.
      static SequelSensorROI Null()
      {
          auto roi = SequelSensorROI(0, 0, 0,0, 1,1);
          roi.SetPixelLaneWidth(32);
          return roi;
      }

      static SequelSensorROI GetDefault(ChipClass chipClass)
      {
          switch (chipClass) {
          case ChipClass::Sequel:
          case ChipClass::DONT_CARE:
              return SequelAlpha();
          case ChipClass::Spider:
              return Spider();
          default:
              throw PBException("Can't determine SensorROI for unknown chip class");
          }
      }

      uint32_t PhysicalRowOffset() const
      { return physicalRowOffset_; }

      uint32_t PhysicalColOffset() const
      { return physicalColOffset_; }

      uint32_t PhysicalRows() const
      { return physicalRows_; }

      uint32_t PhysicalCols() const
      { return physicalCols_; }

      uint32_t PhysicalRowMax() const
      { return physicalRowOffset_ + physicalRows_; }

      uint32_t PhysicalColMax() const
      { return physicalColOffset_ + physicalCols_; }

      uint32_t NumPixelRowsPerZmw() const { return numPixelRowsPerZmw_; }
      uint32_t NumPixelColsPerZmw() const { return numPixelColsPerZmw_; }
      uint32_t PixelsPerZmw() const { return numPixelRowsPerZmw_ * numPixelColsPerZmw_;}

      void SetPhysicalOffset(uint32_t physicalRowOffset0, uint32_t physicalColOffset0)
      {
          physicalRowOffset_ = physicalRowOffset0;
          physicalColOffset_ = physicalColOffset0;
      }

      void SetPhysicalSize(uint32_t physicalRows0, uint32_t physicalCols0)
      {
          physicalRows_ = physicalRows0;
          physicalCols_ = physicalCols0;
      }

      uint32_t PixelLaneWidth() const
      {
          return columnModulo_;
      }

      void SetPixelLaneWidth(uint32_t x)
      {
          columnModulo_ = x;
      }

      uint32_t ConvertRelativeRowPixelToAbsolute(uint32_t row) const
      {
          return row + PhysicalRowOffset();;
      }

      uint32_t ConvertRelativeColPixelToAbsolute(uint32_t col) const
      {
          return col + PhysicalColOffset();;
      }

      uint32_t TotalPixels() const
      {
          return PhysicalRows() * PhysicalCols();
      }

      /// This is almost the same as the number of ZMWs, but not all sensor pixels cover
      /// sequencing ZMWs.
      uint32_t NumUnitCells() const {
          return TotalPixels() / numPixelRowsPerZmw_ / numPixelColsPerZmw_;
      }

      uint32_t GetTileOffsetOfPixel(RowPixels row, ColPixels col) const
      {
          return (row.Value() * PhysicalCols() + col.Value()) / Tile::NumPixels;
      }

      // advances the pixel at (row,col) by one ZMW
      // \returns true if next pixel is in ROI, false if pixel is off ROI
      bool AdvanceOneZmw(RowPixels& row, ColPixels& col) const
      {
          col += ColPixels(numPixelColsPerZmw_);
          if (col == PhysicalColMax())
          {
              col = PhysicalColOffset();
              row += RowPixels(numPixelRowsPerZmw_);
              if (row.Value() >= PhysicalRowMax())
              {
                  return false;
              }
          }
          return true;
      }

      void ToStream(std::ostream& s) const;

      std::string ToString() const;

      /// throws exceptions if the proposed rectangle is not valid
      void CheckProposedROI(RowPixels minRow, ColPixels minCol, RowPixels numRows, ColPixels numCols) const;
  };

    inline bool operator==(const SequelSensorROI& a, const SequelSensorROI& b)
    {
        return a.PhysicalRowOffset() == b.PhysicalRowOffset()
            && a.PhysicalColOffset() == b.PhysicalColOffset()
            && a.PhysicalRows()      == b.PhysicalRows()
            && a.PhysicalCols()      == b.PhysicalCols() ;
    }
    inline bool operator!=(const SequelSensorROI& a, const SequelSensorROI& b)
    {
        return !(a == b);
    }

  class SequelSparseROI;

  class SequelROI
  {
  public:
      virtual ~SequelROI() {}
      SMART_ENUM(ROI_Type_e, Rectangular, Sparse);

      /// The enumerator class enumerates over Unit Cells. It advances the row_ and col_
      /// members over the pixels.  For Sequel, this means that col_ will advance by 2 pixels
      /// and for Spider, col_ will advance by one 1 pixel.
      class Enumerator
      {
      public:
          Enumerator(const SequelROI& roi, size_t begin, size_t end)
                  :
                  roi_(roi),
                  row_(roi.SensorROI().PhysicalRowOffset()),
                  col_(roi.SensorROI().PhysicalColOffset()),
                  index_(0),
                  end_(end)
          {
              if (!roi_.ContainsPixel(row_, col_)) AdvanceToNext();
              index_ = 0;
              while (index_ < begin && (bool) (*this))
              {
                  AdvanceToNext();
              }
          }

          Enumerator& operator++(int)
          {
              AdvanceToNext();
              return *this;
          }

          Enumerator& operator+=(int x)
          {
              for (int i = 0; i < x; i++) AdvanceToNext();
              return *this;
          }

          operator bool() const
          { return index_ < end_; }

          std::pair<RowPixels, ColPixels> UnitCellPixelOrigin() const
          {
              return std::pair<RowPixels, ColPixels>(row_, col_);
          }

          uint32_t Index() const
          { return index_; }

          RowPixels GetPixelRow() const { return row_;}
          ColPixels GetPixelCol() const { return col_;}
          PixelCoord GetPixelCoord() const { return PixelCoord(row_,col_);}
      private:
          void AdvanceToNext()
          {
              do
              {
                  if (!roi_.SensorROI().AdvanceOneZmw(row_,col_))
                  {
                      index_ = end_;
                      return;
                  }
              }
              while (!roi_.ContainsPixel(row_, col_));
              index_++;
          }

          const SequelROI& roi_;
          RowPixels row_;
          ColPixels col_;
          uint32_t index_;
          uint32_t end_;
      };

  public:
      // interface declaration
      virtual uint32_t TotalPixels() const = 0;

      virtual bool Everything() const = 0;

      /// \param Input: tileOffset is the 0-base offset of the tile to test, based on the upper-left corner
      /// \param Output: *internalOffset is the 0-based offset inside the ROI, if the tileOffset is inside the ROI. 
      //          Note that the internalOffsets may be in any order. 0 is not necessarily the upper-left corner.
      ///         If internalOffset is nullptr when called, then it is not written to.
      /// \returns True if tileOffset is within the ROI, false if it is not. If false, then *internalOffset is not written.
      virtual bool ContainsTileOffset(uint32_t tileOffset, uint32_t* internalOffset = nullptr) const = 0;

      virtual void ToStream(std::ostream& s) const = 0;

      virtual std::string ToString() const = 0;

      virtual void CheckROI() const = 0;

      virtual ROI_Type_e Type() const = 0;

      virtual SequelROI* Clone() const = 0;

      virtual Json::Value GetJson() const = 0;

      virtual bool operator==(const SequelROI& a)  const = 0;

      virtual uint32_t AbsoluteRowPixelMin() const = 0;
      virtual uint32_t AbsoluteColPixelMin() const = 0;

  public:
      // specific methods
      uint32_t CountHoles() const
      {
          return CountZMWs();
      }

      uint32_t CountZMWs() const
      {
          return TotalPixels() / PixelsPerZmw();
      }

      const SequelSensorROI& SensorROI() const
      {
          return sensorRoi_;
      }

      operator SequelSparseROI&()
      {
          if (Type() == ROI_Type_e::Sparse)
          {
              return *(SequelSparseROI*) this;
          }
          throw PBException("SequelROI Type not supported: " + Type().toString());
      }

      operator SequelRectangularROI&()
      {
          if (Type() == ROI_Type_e::Rectangular)
          {
              return *(SequelRectangularROI*) this;
          }
          throw PBException("SequelROI Type not supported" + Type().toString());
      }

      bool ContainsPixel(RowPixels row, ColPixels col) const
      {
          uint32_t tileOffset = SensorROI().GetTileOffsetOfPixel(row, col);
          return ContainsTileOffset(tileOffset);
      }
      bool ContainsPixel(const PixelCoord&& coord) const
      {
          uint32_t tileOffset = SensorROI().GetTileOffsetOfPixel(coord.row, coord.col);
          return ContainsTileOffset(tileOffset);
      }

  public:
      // static methods
      static SequelRectangularROI Null();

      uint32_t PixelsPerZmw() const { return sensorRoi_.PixelsPerZmw(); }

  protected:
      SequelROI(const SequelSensorROI& sensorRoi0)
              : sensorRoi_(sensorRoi0)
      { }

      SequelSensorROI sensorRoi_;
  };


  class SequelRectangularROI :
          public SequelROI
  {
  public:
      virtual ~SequelRectangularROI() {}
      // Interface implemention for SequelROI interface
      uint32_t TotalPixels() const override
      {
          return NumPixelRows() * NumPixelCols();
      }

      bool Everything() const override
      {
          return everything_;
      }

      bool ContainsTileOffset(uint32_t tileOffset, uint32_t* internalOffset=nullptr) const override;

      void ToStream(std::ostream& s) const override;

      std::string ToString() const override;

      void CheckROI() const override;

      ROI_Type_e Type() const override
      {
          return ROI_Type_e::Rectangular;
      }

      SequelROI* Clone() const override;

      Json::Value GetJson() const override;

      bool operator==(const SequelROI& a)  const override ;

  public:
      // specific methods
      SequelRectangularROI(const SequelRectangularROI& a)
              : SequelRectangularROI(a.AbsoluteRowPixelMin(), a.AbsoluteColPixelMin(),
                                     a.NumPixelRows(), a.NumPixelCols(), a.SensorROI())
      {

      }

      SequelRectangularROI(const SequelRectangularROI& a, const SequelSensorROI& sensorROI)
              : SequelRectangularROI(a.AbsoluteRowPixelMin(), a.AbsoluteColPixelMin(),
                                     a.NumPixelRows(), a.NumPixelCols(), sensorROI)
      {

      }
      SequelRectangularROI(const SequelSensorROI& sensorROI)
              : SequelRectangularROI(sensorROI.PhysicalRowOffset(), sensorROI.PhysicalColOffset(),
                                     sensorROI.PhysicalRows(), sensorROI.PhysicalCols(), sensorROI)
      {

      }
      SequelRectangularROI(const Json::Value& jsonSpecification, const SequelSensorROI& sensorROI);

      SequelRectangularROI(RowPixels rowMin0, ColPixels colMin0, RowPixels numRows0, ColPixels numCols0,
                           const SequelSensorROI& sensorRoi0);

      uint32_t NumPixelRows() const
      { return numRows_.Value(); }

      uint32_t NumPixelCols() const
      { return numCols_.Value(); }

      // min value is inclusive
      uint32_t AbsoluteRowPixelMin() const override
      { return rowMin_.Value(); }

      // max value is exclusive (i.e. +1 more than the last row)
      uint32_t AbsoluteRowPixelMax() const
      { return rowMax_.Value(); }

      uint32_t AbsoluteColPixelMin() const override
      { return colMin_.Value(); }

      uint32_t AbsoluteColPixelMax() const
      { return colMax_.Value(); }

      /// Relative to the physicalOffset
      int32_t RelativeRowPixelMin() const
      { return rowMin_.Value() - sensorRoi_.PhysicalRowOffset(); }

      int32_t RelativeRowPixelMax() const
      { return rowMax_.Value() - sensorRoi_.PhysicalRowOffset(); }

      /// Relative to the physicalOffset
      int32_t RelativeColPixelMin() const
      { return colMin_.Value() - sensorRoi_.PhysicalColOffset(); }

      int32_t RelativeColPixelMax() const
      { return colMax_.Value() - sensorRoi_.PhysicalColOffset(); }

  private:
      RowPixels rowMin_;
      ColPixels colMin_;
      RowPixels rowMax_;
      ColPixels colMax_;
      RowPixels numRows_;
      ColPixels numCols_;
      bool everything_;
  };

  class SequelSparseROI :
          public SequelROI
  {
  public:
      struct Rect
      {
          uint32_t minRow;
          uint32_t minCol;
          uint32_t maxRow;
          uint32_t maxCol;
      };

  public:
      // Interface implemention for SequelROI interface
      uint32_t TotalPixels() const override;

      bool Everything() const override;

//      std::pair<RowPixels,ColPixels> UnitCellPixelOriginByIndex(size_t index) const;
      bool ContainsTileOffset(uint32_t tileOffset,uint32_t* internalOffset=nullptr) const override;

      void ToStream(std::ostream& s) const override;

      std::string ToString() const override;

      void CheckROI() const override;

      ROI_Type_e Type() const override
      { return ROI_Type_e::Sparse; }

      SequelROI* Clone() const override;

      Json::Value GetJson() const override;

      bool operator==(const SequelROI& a)  const override;

      void SelectAll();

  public:
      virtual ~SequelSparseROI() {}
      // specific methods
      SequelSparseROI(const SequelSensorROI& sensorROI);

      SequelSparseROI(const SequelSparseROI& a);

      SequelSparseROI(const Json::Value& jsonSpecification, const SequelSensorROI& sensorROI);

      SequelSparseROI(const std::string& jsonSpecificationAsString, const SequelSensorROI& sensorROI);

      bool operator==(const SequelSparseROI& other) const;

      bool Contains(const PixelCoord& coord) const;
      void AddRectangle(RowPixels minRow, ColPixels minCol, RowPixels numRows, ColPixels numCols);
      void AddZMW(const PixelCoord& coord);

      SequelSparseROI Intersect(SequelSparseROI& b) const;

      std::vector<Rect> CondensedRects() const;
      Json::Value Condense() const;
      void PostAddRectangle();

      uint32_t AbsoluteRowPixelMin() const override;
      uint32_t AbsoluteColPixelMin() const override;
  private:
      void ImportJson(const Json::Value& jsonSpecification);
      void Recache() const;

  private:
      std::vector<bool> bitmask_;
      mutable std::vector<int32_t> internalOffsets_;
      uint32_t totalPixels_;
      bool everything_;
      Json::Value json_;
      mutable bool dirty_ = true;
      mutable uint32_t absRowPixelMin_;
      mutable uint32_t absColPixelMin_;
  };

  std::ostream& operator<<(std::ostream& s, const PacBio::Primary::SequelROI& roi);

  std::ostream& operator<<(std::ostream& s, const PacBio::Primary::SequelSensorROI& roi);
 }
}


