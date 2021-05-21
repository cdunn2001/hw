// Copyright (c) 2014-2016, Pacific Biosciences of California, Inc.
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted (subject to the limitations in the
// disclaimer below) provided that the following conditions are met:
//
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
//  * Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
//  * Neither the name of Pacific Biosciences nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE
// GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY PACIFIC
// BIOSCIENCES AND ITS CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL PACIFIC BIOSCIENCES OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
// USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
// OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGE.

// Programmer: Armin Töpfer

#include <memory>
#include <string>

#include <boost/regex.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>

#include <pbbam/BamRecord.h>
#include <pbbam/DataSet.h>

#include <pacbio/logging/Logger.h>
#include <pacbio/utilities/ISO8601.h>
#include <pacbio/text/PBXml.h>
#include <pacbio/text/String.h>
#include <pacbio/PBException.h>

#include <bazio/FileHeader.h>

#include <postprimary/bam/RuntimeMetaData.h>

#include "UserParameters.h"

namespace PacBio {
namespace Primary {
namespace Postprimary {

using namespace PacBio::Text;
using namespace PacBio::Utilities;
using namespace PacBio::BAM;
using namespace PacBio::Primary;

class MetadataParser
{
public:
    static std::shared_ptr<RuntimeMetaData> ParseRMD(const std::string& runtimeMetaDataFilePath)
    {
        // Prepare result
        auto rmd = std::make_shared<RuntimeMetaData>();

        // If user defined the meta data
        if (!runtimeMetaDataFilePath.empty())
        {
            PBLOG_INFO << "Parsing " + runtimeMetaDataFilePath;
            // Parse XML file very efficient and save in a string
            std::ifstream in(runtimeMetaDataFilePath.c_str(),
                             std::ios::in | std::ios::binary);
            std::string xmlS;
            if (!in)
            {
                throw PBException("Can't open metadata file " + runtimeMetaDataFilePath);
            }
            else
            {
                in.seekg(0, std::ios::end);
                xmlS.resize(in.tellg());
                in.seekg(0, std::ios::beg);
                in.read(&xmlS[0], xmlS.size());
                in.close();

                // Parse XML content
                PBXml xml(xmlS);

                try
                {
                    auto subreadset = xml.Down("pbdm:PacBioDataModel")
                            .Down("pbdm:ExperimentContainer")
                            .Down("pbdm:Runs")
                            .Down("pbdm:Run")
                            .Down("pbdm:Outputs")
                            .Down("pbdm:SubreadSets")
                            .Down("pbds:SubreadSet");

                    rmd->schemaVersion              = xml.Down("pbdm:PacBioDataModel")
                                                         .currentNode.attribute("Version").value();
                    rmd->dataSetCollection          = subreadset.Down("pbds:DataSetMetadata")
                                                                .Down("pbmeta:Collections").Print();
                    rmd->externalResource           = subreadset.RawChildXML("ExternalResources");
                    rmd->subreadSet.uniqueId        = subreadset.currentNode.attribute("UniqueId").value();
                    rmd->subreadSet.timeStampedName = subreadset.currentNode.attribute("TimeStampedName").value();
                    rmd->subreadSet.createdAt       = subreadset.currentNode.attribute("CreatedAt").value();
                    rmd->subreadSet.name            = subreadset.currentNode.attribute("Name").value();
                    rmd->subreadSet.tags            = subreadset.currentNode.attribute("Tags").value();

                    auto run = xml.Down("pbdm:PacBioDataModel")
                            .Down("pbdm:ExperimentContainer")
                            .Down("pbdm:Runs")
                            .Down("pbdm:Run");

                    auto collection = subreadset.Down("pbds:DataSetMetadata")
                            .Down("pbmeta:Collections");
                    if (collection.currentNode != NULL)
                    {
                        auto collMeta = collection.Down("pbmeta:CollectionMetadata");

                        // Get runID
                        rmd->runId = collMeta.currentNode.attribute("Context").value();

                        if (collMeta.currentNode != NULL)
                        {
                            // Get PartNumber of the BindingKit
                            if (collMeta.Down("pbmeta:BindingKit").currentNode != NULL)
                            {
                                rmd->bindingKit = collMeta.Down("pbmeta:BindingKit")
                                        .currentNode
                                        .attribute("PartNumber")
                                        .value();
                            }
                            // Get PartNumber of SequencingKitPlate
                            if (collMeta.Down("pbmeta:SequencingKitPlate").currentNode != NULL)
                            {
                                rmd->sequencingKit = collMeta.Down("pbmeta:SequencingKitPlate")
                                        .currentNode
                                        .attribute("PartNumber")
                                        .value();
                            }
                            // Get adapter sequences of TemplatePrepKit
                            if (collMeta.Down("pbmeta:TemplatePrepKit").currentNode != NULL)
                            {
                                if (collMeta.Down("pbmeta:TemplatePrepKit").Down("pbbase:LeftAdaptorSequence").currentNode != NULL)
                                {
                                    rmd->leftAdapter = collMeta.Down("pbmeta:TemplatePrepKit")
                                            .Down("pbbase:LeftAdaptorSequence")
                                            .currentNode
                                            .child_value();
                                }
                                if (collMeta.Down("pbmeta:TemplatePrepKit").Down("pbbase:RightAdaptorSequence").currentNode != NULL)
                                {
                                    rmd->rightAdapter = collMeta.Down("pbmeta:TemplatePrepKit")
                                            .Down("pbbase:RightAdaptorSequence")
                                            .currentNode
                                            .child_value();
                                }
                            }
                            // Get control sequence and adapters of ControlKit
                            if (collMeta.Down("pbmeta:ControlKit").currentNode != NULL)
                            {
                                if (collMeta.Down("pbmeta:ControlKit").Down("pbbase:CustomSequence").currentNode != NULL)
                                {
                                    std::string s = collMeta.Down("pbmeta:ControlKit")
                                            .Down("pbbase:CustomSequence")
                                            .currentNode
                                            .child_value();

                                    // Parse the sequence for the left adapter, right adapter, and actual control sequence.
                                    std::vector<std::string> sp = PacBio::Text::String::Split(s, '\\');
                                    // Get rid of escaped newline.
                                    for (uint32_t i = 1; i < sp.size(); i++) sp[i] = std::string(sp[i].begin()+1, sp[i].end());
                                    if (sp[0] == ">left_adapter") rmd->leftAdapterControl = sp[1];
                                    if (sp[2] == ">right_adapter") rmd->rightAdapterControl = sp[3];
                                    if (sp[4] == ">custom_sequence") rmd->control = sp[5];
                                }
                            }

                        }
                        rmd->movieName  = rmd->runId;
                    }
                }
                catch (PBXmlException& e)
                {
                    std::cerr << e.what() << std::endl;
                }
            }
        }
        else // Fake information
        {
            PBLOG_INFO << "No metadata available";
            rmd->bindingKit    = "100372700";
            rmd->sequencingKit = "100356200";
            rmd->basecallerVersion = "2.3";
            //rmd->chemistry = "P6-C4";
            rmd->platform = Platform::RSII;
            rmd->runId = "m150426_185711_42175_c100779252550000001823156808111521_s1_p0";
        }

        return rmd;
    }

    static std::shared_ptr<RuntimeMetaData> ParseRMD(const std::shared_ptr<UserParameters>& user)
    {
        return ParseRMD(user->runtimeMetaDataFilePath);
    }

    static std::shared_ptr<RuntimeMetaData> ParseRMD(
        const FileHeader& fileHeader,
        const std::shared_ptr<UserParameters>& user)
    {
        auto rmd = ParseRMD(user);

        // Now that we've parsed the MetadataXML, finish the RMD with data from the fileHeader
        rmd->basecallerVersion = fileHeader.BaseCallerVersion();
        if (user->runtimeMetaDataFilePath.empty())
            rmd->movieName  = fileHeader.MovieName();

        rmd->platform = Platform::SEQUELII;

        return rmd;
    }

private:
    static std::string Encode(const std::string& data) {
        std::string buffer;
        buffer.reserve(data.size());
        for(size_t pos = 0; pos != data.size(); ++pos) {
            switch(data[pos]) {
                case '&':  buffer.append("&amp;");       break;
                case '\"': buffer.append("&quot;");      break;
                case '\'': buffer.append("&apos;");      break;
                case '<':  buffer.append("&lt;");        break;
                case '>':  buffer.append("&gt;");        break;
                default:   buffer.append(&data[pos], 1); break;
            }
        }
        return buffer;
    }
};

}}}

