#ifndef Sequel_Common_HQRFParamsMCart_H_
#define Sequel_Common_HQRFParamsMCart_H_


// Copyright (c) 2018-2019, Pacific Biosciences of California, Inc.
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
// File Description: A trained model generated programmatically using
// the primary-toolkit python package

/// This file is automatically generated by the CART training scripts in
/// primary-toolkit (classifiers/CART.py). Do not edit this file manually!

#include <array>
#include <tuple>
#include <limits>
#include <pacbio/primary/HQRFMethod.h>

namespace PacBio {
namespace Primary {
namespace ActivityLabeler {

struct HQRFParamsMCart
{
	static const float maxAcceptableHalfsandwichRate;
	static const std::array<float, 2> hswCurve;
	static const std::array<int16_t, 663> childrenLeft;
	static const std::array<int16_t, 663> childrenRight;
	static const std::array<int8_t, 663> feature;
	static const std::array<float, 663> threshold;
	static const std::array<int8_t, 663> value;
    static const std::array<CART_FEATURES, 11> featureOrder;

};

}}} // PacBio::Primary::ActivityLabeler
#endif // Sequel_Common_HQRFParamsMCart_H_
