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
/// \brief  class for managing unit cells in the chip layout


#ifndef SEQUELACQUISITION_UNITCELL_H
#define SEQUELACQUISITION_UNITCELL_H

#include <stdint.h>

namespace PacBio
{
 namespace Primary
 {
  struct UnitX;
  struct UnitY;

  struct UnitX
  {
      UnitX(int32_t x)
              : value_(x)
      { }

      int32_t Value() const
      { return value_; }

  private:
      int32_t value_;
  };

  struct UnitY
  {
      UnitY(int32_t y)
              : value_(y)
      { }

      int32_t Value() const
      { return value_; }

  private:
      int32_t value_;
  };


  struct UnitCell
  {
      union
      { ;
          struct
          {
              int16_t y; // Yes, y comes first in memory order, before on purpose.
              int16_t x; // This makes y the least-significant halfword, and x the most-significant halfword.
          };
          uint32_t id;
      };

      UnitCell()
              : y(0),
                x(0)
      { }

      explicit UnitCell(uint32_t number) : id(number)
      { }

      explicit UnitCell(int16_t x0, int16_t y0)
              : y(y0),
                x(x0)
      { }

      explicit UnitCell(UnitX x0, UnitY y0)
              : y(static_cast<int16_t>(y0.Value())),
                x(static_cast<int16_t>(x0.Value()))
      { }
      explicit UnitCell(const char* excelLabel);


      std::string ExcelCell() const;

      /// deprecated. Use Number() instead
      uint32_t ID() const
      { return id; }

      uint32_t Number() const
      { return id;}

  };

//  static_assert(UnitCell(1,2).Number() == 0x00010002,"Checking endianess of the compiler");
 }

}

#endif //SEQUELACQUISITION_UNITCELL_H

