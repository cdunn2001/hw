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
/// \brief declaration of the Kernel (mathematic sense) object for convolutions


#ifndef SEQUELACQUISITION_KERNEL_H
#define SEQUELACQUISITION_KERNEL_H

#include <array>
#include <json/json.h>
#include <boost/multi_array.hpp>
#include <pacbio/PBException.h>
#include "SequelMovie.h"

namespace PacBio
{
 namespace Primary
 {
  class Kernel
  {
  public:
      /// structors. default
      Kernel();

      /// structors. This one takes a 2 D JSON array
      explicit Kernel(const Json::Value& arrays);

      /// Creates a kernel of the indicated rows and columns and fills with zeros.
      Kernel(int rows, int cols);

      static Kernel UnityKernel()
      {
          Kernel unity(1,1);
          unity(0,0) = 1.0;
          return unity;
      }

     /// Create a kernel from the multi_array, foray into duck typing
      template<typename Array>
      explicit Kernel(const Array& ma)
      {
          auto& ma_shape = reinterpret_cast<boost::array<size_t, 2> const&>(* ma.shape());
          marray_.resize(ma_shape);
          marray_ = ma;
      }


      explicit Kernel(const Kernel& a)
      {
          auto& ma_shape = reinterpret_cast<boost::array<size_t, 2> const&>(* a.marray_.shape());
          marray_.resize(ma_shape);
          marray_ = a.marray_;
      }

      Kernel(Kernel&& a)
      {
          auto& ma_shape = reinterpret_cast<boost::array<size_t, 2> const&>(* a.marray_.shape());
          marray_.resize(ma_shape);
          marray_ = a.marray_;
      }

      Kernel& operator=(const Kernel& a)
      {
          auto& ma_shape = reinterpret_cast<boost::array<size_t, 2> const&>(* a.marray_.shape());
          marray_.resize(ma_shape);
          marray_ = a.marray_;
          return *this;
      }

      /// Convoles one kernel with another.
      static Kernel Convolve(const Kernel& a, const Kernel& b);

      /// Convolves the kernel with another.
      Kernel operator*(const Kernel& a) const;

      /// Convolves the kernel with another.
      Kernel operator+(const Kernel& a) const;

      /// expands or truncates a kernel to a larger size.
      /// If expanding, zeros are written to the outer rows and oclumns.
      /// If reducing, the remaining kernel elements are renormalized to 1.
      Kernel Resize(int rows, int cols) const;

      /// scales each coefficient by factor
      Kernel Scale(double factor) const;

      /// sums all coefficients
      double Sum() const;

      /// scales al coefficients so that they add up to sum. If the original sum is
      /// zero, then an exception is thrown.
      Kernel Normalize(double sum = 1.0) const;

      ///return kernel coefficient at row and col, counting from 0 to NumRows-1, etc.
      double operator()(int row, int col) const;

      /// return reference to kernel coefficient at row and col
      double& operator()(int row, int col);

      /// returns the coefficient at row and col, or value if row and col are not within the kernel
      double GetOrDefault( int row, int col , double value) const;

      /// sets an element, only if hte row and col are in range
      void SafeSet( int row, int col , double value);

      /// returns numbers of rows in the kernel
      int NumRows() const { return marray_.shape()[0];}

      // returns number of columns in the kernel
      int NumCols() const { return marray_.shape()[1];}


      std::vector< std::vector<double> > AsVectors() const {
          std::vector< std::vector<double>> x;
          x.resize(marray_.shape()[0]);
          for (unsigned int i = 0; i < marray_.shape()[0]; i++)
          {
              x[i].resize(marray_.shape()[1]);
              for (unsigned int j = 0; j < marray_.shape()[1]; j++)
              {
                  x[i][j] = marray_[i][j];
              }
          }
          return x;
      }

      bool IsUnity() const;

      // apply a 2nd order inversion approximation
      Kernel InvertTo7x7() const;

      /// flips the kernel around both axes to covert a colution kernel to a correlation kernel (and vice versa)
      Kernel Flip() const;

      const boost::multi_array<double,2>& AsMultiArray() const { return marray_; }
  private:
      /// the outer vector is rows
      /// the inner vector is columns
#if 1
      boost::multi_array<double,2> marray_;
#else
      std::vector <std::vector<double> > array_;
#endif
      friend std::ostream& operator<<(std::ostream& s, const Kernel& k);
  };

  struct Spectrum
  {
      Spectrum(double g,double r) : values_{ g, r } {}
      Spectrum(const std::array<float,2>& dyeSpectrum) : values_{ dyeSpectrum[0], dyeSpectrum[1] } {}
      Spectrum(const std::vector<float> & dyeSpectrum)
      {
          if (dyeSpectrum.size() == 1 )
          {
              values_[0] = dyeSpectrum[0];
              values_[1] = dyeSpectrum[0]; // replicate value
          }
          else if (dyeSpectrum.size() == 2 )
          {
              values_[0] = dyeSpectrum[0];
              values_[1] = dyeSpectrum[1];
          }
          else
          {
              throw PBException("dyeSpectrum of size " + std::to_string(dyeSpectrum.size()) + " is not supported");
          }
      }
      Spectrum() : values_{ 0, 0 } {}
      void Normalize();
      double& operator[](int i) { if (i<0 || i>1) throw PBException("out of range"); return values_[i];}
      double operator[](int i) const { if (i<0 || i>1) throw PBException("out of range"); return values_[i];}
      operator std::array<float,2>() const { return std::array<float,2>{{ static_cast<float>(values_[0]),static_cast<float>(values_[1])}}; }
      operator std::vector<float>() const { return std::vector<float>{{ static_cast<float>(values_[0]),static_cast<float>(values_[1])}}; }
  private:
      double values_[2];
  };

  Spectrum CorrectSpectrum(const Kernel& kernel, const Spectrum& raw);

  std::ostream& operator<<(std::ostream& s, const Kernel& k);
 }
}


#endif //SEQUELACQUISITION_KERNEL_H
