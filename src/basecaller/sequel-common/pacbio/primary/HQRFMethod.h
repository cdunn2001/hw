// Copyright (c) 2020, Pacific Biosciences of California, Inc.
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
/// \brief  declaration of the HQRF method enumerations

#ifndef SEQUELBASECALLER_HQRFPUBLICMETHOD_H
#define SEQUELBASECALLER_HQRFPUBLICMETHOD_H

#include <pacbio/logging/Logger.h>
#include <pacbio/utilities/SmartEnum.h>
#include <pacbio/PBException.h>
#include <map>

namespace PacBio {
namespace Primary {

/// This enum is made public by to other processes and should be used as the interface to ICS.
SMART_ENUM(HqrfPublicMethod, DEFAULT, K1,M1,N1,M2,M3,M4,N2);

/// This enum should be private to C++ code only and the names are not public facing.
enum class HQRFMethod
{
    SEQUEL_CRF_HMM,
    SPIDER_CRF_HMM,
    ZOFFSET_CRF_HMM,
    TRAINED_CART_HMM,
    TRAINED_CART_CART,
    BAZ_HMM,
    ZOFFSET_CART_HMM
};

//const std::map<HqrfPublicMethod, ChipClass> chipClassCompatible = {
//        { HqrfPublicMethod::K1, ChipClass::Sequel },
//        { HqrfPublicMethod::M1, ChipClass::Spider },
//        { HqrfPublicMethod::N1, ChipClass::Spider },
//        { HqrfPublicMethod::M2, ChipClass::Spider },
//        { HqrfPublicMethod::M3, ChipClass::Spider },
//        { HqrfPublicMethod::M4, ChipClass::Spider },
//        { HqrfPublicMethod::N2, ChipClass::Spider },
//};

const std::map<HQRFMethod, bool> realtimeCompatible = {
        {HQRFMethod::SEQUEL_CRF_HMM, false},
        {HQRFMethod::SPIDER_CRF_HMM, false},
        {HQRFMethod::ZOFFSET_CRF_HMM, false},
        {HQRFMethod::TRAINED_CART_HMM, true},
        {HQRFMethod::TRAINED_CART_CART, true},
        {HQRFMethod::BAZ_HMM, false},
        {HQRFMethod::ZOFFSET_CART_HMM, true},
};

enum class CART_FEATURES {
    PULSERATE = 0,
    SANDWICHRATE,
    LOCALHSWRATENORM,
    VITERBISCORE,
    MEANPULSEWIDTH,
    LABELSTUTTERRATE,
    BLOCKLOWSNR,
    MAXPKMAXNORM,
    AUTOCORRELATION,
    BPZVARNORM,
    PKZVARNORM,
    NUM_FEATURES
};

inline HQRFMethod GetPrivateHQRFMethod(HqrfPublicMethod method)
{
    if (method == HqrfPublicMethod::DEFAULT)
    {
        //switch(chipClass)
        //{
        //case ChipClass::Sequel:
        //case ChipClass::DONT_CARE:
        //    method = HqrfPublicMethod::K1;
        //    break;
        //case ChipClass::Spider:
        //    method = HqrfPublicMethod::M1;
        //    break;
        //default:
        //    throw PBException("Can't determine HQRFMethod from chip class: " + chipClass.toString());
        //}
        method = HqrfPublicMethod::M1;
    }

    //if (chipClassCompatible.at(method) != chipClass)
    //    PBLOG_WARN << "hqrf.method : " + method.toString() + " incompatible with : " + chipClass.toString();

   switch (method)
    {
    case HqrfPublicMethod::K1:            return HQRFMethod::SEQUEL_CRF_HMM;
    case HqrfPublicMethod::M1:            return HQRFMethod::SPIDER_CRF_HMM;
    case HqrfPublicMethod::N1:            return HQRFMethod::ZOFFSET_CRF_HMM;
    case HqrfPublicMethod::M2:            return HQRFMethod::TRAINED_CART_HMM;
    case HqrfPublicMethod::M3:            return HQRFMethod::TRAINED_CART_CART;
    case HqrfPublicMethod::M4:            return HQRFMethod::BAZ_HMM;
    case HqrfPublicMethod::N2:            return HQRFMethod::ZOFFSET_CART_HMM;
    default:
        throw PBException("Unknown hqrf.method : " + method.toString());
    }
}

}}

#endif //SEQUELBASECALLER_HQRFPUBLICMETHOD_H
