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
//  Description:
/// \brief   Event object definition, transmitted from ICS to pa-acq and through the pipeline
///
#include <boost/regex.hpp>

#include <json/json.h>

#include <pacbio/primary/Event.h>
#include <pacbio/logging/Logger.h>
#include <pacbio/utilities/ISO8601.h>

namespace PacBio
{
 namespace Primary
 {

    Event::Event(const std::string& id0,
                 const std::string& name0) :
            id(id0),
            name(name0)
    {
        boost::regex r("^[A-Za-z]([A-Za-z0-9_]+\\.){2,99}[A-Za-z0-9_]*[A-Za-z0-9]$");
        if (!boost::regex_match(id, r))
            throw PBException("Invalid Event id: " + id);

        whenChanged = PacBio::Utilities::ISO8601::TimeString();
    }

    Event::Event(const Json::Value& value) :
            id (value["id"].asString()),
            name ( value["name"].asString())
    {
        whenChanged = value["whenChanged"].asString() ;
    }

    Json::Value Event::ToJSON() const
    {
        Json::Value value;
        value["id"] = id;
        value["name"] = name;
        value["whenChanged"] = whenChanged;
        return value;
    }

    std::string Event::RenderJSON() const
    {
        std::stringstream ss;
        ss << ToJSON();
        return ss.str();
    }

 }
}
