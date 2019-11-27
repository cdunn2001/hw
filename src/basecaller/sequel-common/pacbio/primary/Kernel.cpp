// Copyright (c) 2017, Pacific Biosciences of California, Inc.
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
// Description:
/// \brief definition of the Kernel (mathematic sense) object for convolutions


#include <pacbio/primary/Kernel.h>
#include <pacbio/PBException.h>

namespace PacBio
{
 namespace Primary
 {
  Kernel::Kernel() : Kernel(0,0) {}
  Kernel::Kernel(const Json::Value& arrays) : Kernel(arrays.size(), arrays[0].size())
  {
      for(int i=0;i<NumRows();i++)
      {
          for (int j=0;j<NumCols();j++)
          {
              (*this)(i,j) = arrays[i][j].asDouble();
          }
      }
  }


/// Creates a kernel of the indicated rows and columns and fills with zeros.
  Kernel::Kernel(int rows, int cols)
  {
      marray_.resize(boost::extents[rows][cols]);
  }

/// Convoles one kernel with another. the resultant size is 2n-1 x 2n-1, if each
/// input is nxn.
  Kernel Kernel::Convolve(const Kernel& a, const Kernel& b)
  {
      int newRows = a.NumRows() + b.NumRows() - 1;
      int newCols = a.NumCols() + b.NumCols() - 1;
      Kernel a2 = a.Resize(newRows,newCols);
      Kernel b2 = b.Resize(newRows,newCols);
      Kernel out(newRows, newCols);

      for(int i=0;i<newRows;i++)
      {
          for( int j=0;j<newCols;j++)
          {
              double term = 0.0;
              for(int m= 0;m<=a.NumRows();m++)
              {
                  for(int n=0;n<=a.NumCols();n++)
                  {
                      term += a.GetOrDefault(m,n ,0) *  b.GetOrDefault(i - m,j - n,0);
                  }
              }
              out(i,j) = term;
          }
      }
      return out;
  }


/// Convolves the kernel with another.
  Kernel Kernel::operator*(const Kernel& a) const
  {
      return Convolve(*this,a);
  }

  Kernel Kernel::operator+(const Kernel& a) const
  {
      Kernel out(a.NumRows(),a.NumCols());
      for(int i=0;i<a.NumRows();i++)
      {
          for (int j = 0; j < a.NumCols(); j++)
          {
              out(i, j) = (*this)(i, j) + a(i, j);
          }
      }
      return out;
  }
/// Expands or truncates a kernel to a larger size.
/// The center of the kernel is preserved.
/// If expanding, zeros are written to the outer rows and columns.
  Kernel Kernel::Resize(int rows, int cols) const
  {
      Kernel out(rows,cols);

      int rowShift = (rows - NumRows())/2;
      int colShift = (cols - NumCols())/2;
      for(int i=0;i<NumRows();i++)
      {
          for (int j = 0; j < NumCols(); j++)
          {
              out.SafeSet(rowShift+i, colShift+ j, (*this)(i,j));
          }
      }
      return out;
  }

  Kernel Kernel::Scale(double factor) const
  {
      Kernel out(*this);
      for(int i=0;i<NumRows();i++)
      {
          for (int j = 0; j < NumCols(); j++)
          {
              out(i, j) *= factor;
          }
      }
      return out;
  }
  double Kernel::Sum() const
  {
      double sum = 0;

      for (int i = 0; i < NumRows(); i++)
      {
          for (int j = 0; j < NumCols(); j++)
          {
              sum += (*this)(i, j);
          }
      }
      return sum;
  }

  Kernel Kernel::Normalize(double target ) const
  {
      double sum = Sum();
      if (sum ==0 )
      {
          throw PBException("Can't scale kernel because current sum is zero");
      }
      double factor = target/sum;
      return Scale(factor);
  }

///return kernel coefficient at row and col.
  double Kernel::operator()( int row, int col ) const
  {
    return marray_[row][col];
  }
  double& Kernel::operator()( int row, int col )
    {
        return marray_[row][col];
    }
  double Kernel::GetOrDefault( int row, int col , double value) const
    {
        if (row >=0 && row < NumRows() && col >=0 && col < NumCols())
        {
            return marray_[row][col];
        }
        else
        {
            return value;
        }
    }
  void Kernel::SafeSet( int row, int col , double value)
  {
      if (row >=0 && row < NumRows() && col >=0 && col < NumCols())
      {
          marray_[row][col] = value;
      }
  }

  bool Kernel::IsUnity() const
  {
      if ((NumRows() % 2) == 0 || (NumCols() % 2) ==0) return false; // must be odd
      // center must be 1, the rest must be 0
      for(int row =0; row < NumRows(); row++)
      {
          for(int col =0; col < NumCols(); col++)
          {
              bool center = (row == NumRows() / 2 && col == NumCols() / 2);
              if (marray_[row][col] != (center ? 1.0 : 0.0)) return false;
          }
      }
      return true;
  }

  // reference: http://sharepoint/progmgmt/Sequel/ResearchAndDevelopment/Software/Primary/AlgorithmInvestigations/LWQC_Sequel_Crosstalk.docx
  //
  // We want to invert P. We can define P as I + A, where A is "smaller".
  // The inverse of P is Sum_k=0^infinity (-A)^-k
  // The first 3 terms of this infinite sum are
  //   P^-1 = A^0 -A^1 + A^2
  // Rewriting A = P - I
  // P^-1 = I - (P-I) + (P-I)^2
  // P^-1 = 3I -3P + P^2

  Kernel Kernel::InvertTo7x7() const
  {
      if (! (NumRows() == 5 && NumCols() == 5))
      {
          throw PBException("Sorry, InvertTo7x7 must use 5x5 kernel. Got " +
              std::to_string(NumRows()) + "x" + std::to_string(NumCols()));
      }
      const Kernel& p(*this);

      Kernel I(9,9);
      I(4,4) = 1.0;
      Kernel Ix3 = I.Scale(3.0);
      Kernel negPx3 = p.Resize(9,9).Scale(-3.0);
      Kernel Psquared = p * p;

      Kernel out = Ix3 + negPx3 + Psquared;
      Kernel out2 = out.Resize(7,7);
      return out2;
  }

  Kernel Kernel::Flip() const
  {
      Kernel out(NumRows(),NumCols());
      for(int i=0;i<NumRows();i++)
      {
          for(int j=0;j<NumCols();j++)
          {
              out(NumRows()-i-1,NumCols()-j-1) = (*this)(i,j);
          }
      }
      return out;
  }

  std::ostream& operator<<(std::ostream& s, const Kernel& k)
  {
      s << "[";
      for(const auto& row : k.marray_)
      {
          s << "  [";
          for(const auto& v : row)
          {
              s << "\t" << v;
          }
          s << "]" << std::endl;
      }
      s << "]" << std::endl;
      return s;
  }

  void Spectrum::Normalize()
  {
      double sum = values_[0] + values_[1];
      if (sum <= 0.0) throw PBException("negative spectrum can't be normalized");
      values_[0] /= sum;
      values_[1] /= sum;
  }

  Spectrum CorrectSpectrum(const Kernel& kernel, const Spectrum& raw)
  {
      int mid = kernel.NumRows()/2;
      if (mid < 1) throw PBException("Kernel too small");

      double fLeft = kernel(mid, mid-1);
      double fMiddle = kernel(mid, mid);
      double fRight = kernel(mid, mid+1);

      Spectrum sp;
      sp[0] = fMiddle * raw[0] +
              fLeft   * raw[1];
      sp[1] = fRight  * raw[0] +
              fMiddle * raw[1];
      sp.Normalize();
      return sp;
  }


 }
}
