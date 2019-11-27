// Copyright (c) 2016, Pacific Biosciences of California, Inc.
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
//  Defines members of struct SmrtSensor.

#include "SmrtSensor.h"

#include <algorithm>

#include <pacbio/PBException.h>
#include <pacbio/logging/Logger.h>
#include <pacbio/text/String.h>

using namespace PacBio::Text;

namespace PacBio {
namespace Primary {

SmrtSensor::SmrtSensor(uint32_t numFilters)
    : numFilters_(0)
    , filterMap_(0)
    , imagePsf          (boost::extents[0][5][5])
    , xtalkCorrection   (boost::extents[7][7])
    , refDwsSnr         (1.0f)
    , refSpectrum(0)
    , photoelectronSensitivity_{1.0}
    , frameRate_(100.0f)
{
    xtalkCorrection[3][3] = 1.0f;
    SetNumFilters(numFilters);
}

void SmrtSensor::SetNumFilters(uint32_t numFilters)
{
    numFilters_ = numFilters;
    filterMap_.resize(numFilters);
    imagePsf.resize(boost::extents[numFilters][5][5]);
    xtalkCorrection.resize(boost::extents[7][7]);
    refSpectrum.resize(numFilters);

    for (uint16_t i = 0; i < NumFilters(); i++)
    {
        filterMap_[i] = i;
        imagePsf[i][2][2] = 1.0f;
    }
    if (numFilters == 2)
    {
        refSpectrum[0] = 0.0f;
        refSpectrum[1] = 1.0f;
    }
    else if (numFilters == 1)
    {
        refSpectrum[0] = 1.0f;
    }
    else
    {
        throw PBException("unsupported numFilters");
    }
}




SmrtSensor& SmrtSensor::FilterMap(const std::vector<uint16_t>& value)
{
    // filterMap is a permutation of [0, 2).
    std::vector<unsigned int> a(NumFilters());
    for (uint32_t i=0;i<NumFilters();i++)
    {
        a[i] = i;
    }
    if (!std::is_permutation(value.cbegin(), value.cend(), a.cbegin()))
    {
        throw PBException("FilterMap must be a legal permutation of 0 to " + std::to_string(NumFilters()-1) +
        " was " + String::AsArray(value.cbegin(), value.cend()));
    }

    filterMap_ = value;
    return *this;
}

#if 0
SmrtSensor& SmrtSensor::ImagePsf(const MultiArray<float, 3>& value)
{
    // First extent is NumFilters. Second and third extents are odd.
    const auto nRows = value.shape()[1];
    const auto nCols = value.shape()[2];
    if (value.shape()[0] != NumFilters || nRows % 2 == 0 || nCols % 2 == 0)
    {
        throw PBException("Badly shaped multiarray for image PSF.");
    }

    for (const auto& psf : value)
    {
        float sum = 0.0f;
        for (const auto& row : psf)
        {
            for (float x : row)
            {
                // Any negative elements should have a small magnitude.
                if (x < -0.005) throw PBException("Magnitude of negative xtalk PSF element exceeds tolerance.");
                sum += x;
            }
        }
        // Sum of elements approximately equal 1.0f.
        if (std::abs(sum - 1.0f) > 0.0001f)
        {
            throw PBException("Badly normalized crosstalk PSF.");
        }
    }

    imagePsf = value;
    return *this;
}

SmrtSensor& SmrtSensor::XtalkCorrection(const MultiArray<float, 2>& value)
{
    // Each extent is odd.
    const auto nRows = value.shape()[0];
    const auto nCols = value.shape()[1];
    if (nRows % 2 == 0 || nCols % 2 == 0)
    {
        throw PBException("Badly shaped multiarray for crosstalk correction filter.");
    }
    float sum = 0.0f;
    for (const auto& row : value)
    {
        for (float x : row) sum += x;
    }
    if (std::abs(sum - 1.0f) > 0.0001f)
    {
        throw PBException("Badly normalized crosstalk correction filter.");
    }

    xtalkCorrection = value;
    return *this;
}
#endif


SmrtSensor& SmrtSensor::RefDwsSnr(float value)
{
    if (value <= 0.0f)
    {
        throw PBException("RefDwsSnr must be positive.");
    }

    refDwsSnr = value;
    return *this;
}


SmrtSensor& SmrtSensor::RefSpectrum(const std::vector<float>& value)
{
    if (value.size() != NumFilters())
    {
        throw PBException("RefSpectrum argument was wrong size");
    }
    // Elements must be non-negative.
    if (std::any_of(value.cbegin(), value.cend(), [](float x){return x < 0.0f;}))
    {
        throw PBException("Elements of RefSpectrum must be non-negative. " +
                                  String::AsArray(value.cbegin(), value.cend()));
    }

    // RefSpectrum must be normalized.
    const auto sum = std::accumulate(value.cbegin(), value.cend(), 0.0f);
    if (std::abs(sum - 1.0f) > 0.0001f)
    {
        throw PBException("RefSpectrum must be normalized. Sum was " + std::to_string(sum) + " " +
                                  String::AsArray(value.cbegin(), value.cend()));
    }

    refSpectrum = value;
    return *this;
}


std::vector<float>
SmrtSensor::ShotNoiseCovar(unsigned int /*filterChannel*/)
{
    // size = NumCvr(SmrtSensor::NumFilters)
    // TODO: Define correctly.
    if (NumFilters() == 1)
    {
        std::vector<float> covar{{1.0f}};
        return covar;
    }
    else if (NumFilters() == 2)
    {
        std::vector<float> covar{{1.0f, 1.0f, 0.0f}};
        return covar;
    }
    throw PBException("not supported");
}


// void SmrtSensor::CalculateShotNoiseCovar(const MultidimensionVector<float>& imagePsf, const MultidimensionVector<float>& xtalkCorrection)
// {
//#if 0
//     function KcSN=ColoredShotNoise(v,xtalkPSF,corrPSF)

//                   %   v: [pkMidG; pkMidR] after xtalk correction (i.e. before xtalk)
//     %   xTalkPSF and corrPSF are square matrices

//                                         %   N = larger matrix size
//                                                               %   define a box with #Rows = N and #Columns = 1 + larger #Rows
//                                                                                                                          %   The extra column is because we're going to place one matrix
//                                                                                                                                                            %   centered on green pixel and one matrix centered on red pixel.
//             N = max(size(xtalkPSF,1),size(corrPSF,1));
//     Nc = N+1;   % green and red as adjacent columns
//     Nr = N;

//     %   After placing matrices in a box, unroll box elements into a column vector
//                                                                            %   Do this two times for xtalkPSF
//             mxG = unroll(xtalkPSF,Nr,Nc,0,0);   % centered on green pixel
//     mxR = unroll(xtalkPSF,Nr,Nc,0,1);   % centered on red pixel
//                                                           %   Do this two times for corrPSF
//             mcG = unroll(corrPSF,Nr,Nc,0,0);
//     mcR = unroll(corrPSF,Nr,Nc,0,1);

//     %   x: unrolled pixel values in box (after xtalk, before correction)
//     x = [mxG mxR]*v;
//     %   shot noise covariance (big diagonal matrix before correction)
//     KxSN = diag(x);

//     %   Each row of mc is the vector of weights to apply to values in x
//                                                                       %   to cross-talk correct the green and red pixels respectively.
//             Mc = [mcG'; mcR'];
//     whos Mc
//     KcSN = Mc*KxSN*Mc';
//     end
//#endif

//#if 0
//function v=unroll(M,Nr,Nc,rOffset,cOffset)

//    Mr = size(M,1);
//    Mc = size(M,2);

//    v = zeros(Nr*Nc,1);
//    rShift = (Nr-Mr)/2 + rOffset;
//    cShift = (Nc-1-Mc)/2 + cOffset;
//    for r = 1:Mr
//        for c = 1:Mc
//            n = (r + rShift-1)*Nc + (c + cShift);
//            v(n) = M(r,c);
//        end
//    end
//end

//#endif
// }

}}  // PacBio::Primary
