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
//  Description:
/// \brief  C++ wrapper around JSON object passed through BAM comments.
///

#ifndef PA_POST_PRIMARY_BAM_BAM_COMMENT_SIDEBAND_H
#define PA_POST_PRIMARY_BAM_BAM_COMMENT_SIDEBAND_H

#include <json/json.h>

namespace PacBio {
namespace Primary {
namespace Postprimary {

/// baz2bam streams its output as a binary BAM file through to stdout.
/// As there is only one stream, side band data that ccs needs is packaged into the "comment"
/// fields of the BAM header. If you use samtools to view the BAM, you'll see it markeds as @CO
/// at the start of the file.
/// Inside this comment is a JSON object, with a field "Name":"baz2bam_css_sideband".
/// The rest of the JSON object follows the setters for this class.
/// Documentation for this class is found in
///  TAK-816  version "1.2"
///  TAK-778  version "1.1"
///  TAG-4606 version 1.0

class BamCommentSideband
{
public:
    /// This struct is used to pass named parameters to AddExternalResource.
    /// \param file_name - the absolute path to the external resource.  The final
    ///                    dataset XML may convert this to a relative path.
    /// \param meta_type - The type defined in PacBioDataModel.xsd from the xsd-datamodels repo
    struct ExternalResource
    {
        std::string file_name;
        std::string meta_type;
    };

    /// default ctor
    BamCommentSideband()
    {
        comment_["name"] = "baz2bam_ccs_sideband";
        comment_["version"] = "1.2";
        comment_["external_resources"] = Json::arrayValue;
    }

    /// This is the number of ZMWs that baz2bam processed and will be streaming to ccs.
    /// This number is only used to predict progress for ccs.
    void SetNumZmws(int64_t numZmws)
    {
        comment_["zmws"] = numZmws;
    }

    /// This sets the UUID of the subreadset
    void SetDatasetUUID(const std::string& uuid)
    {
        comment_["dataset_uuid"] = uuid;
    }

    /// \param xml - The string representation of the <CollectionMetadata> node
    /// that is constructed by the run design and passed through pa-ws by ICS.
    /// This string shall start with "<CollectionMetadata " and end with
    /// "</CollectionMetadata>".
    /// Internally, the string is converted to base64 encoding and then placed
    /// inside the JSON object. This is to avoid escaping the XML to fit inside
    /// JSON.
    void SetCollectionMetadataXML(const std::string& xml)
    {
        comment_["collection_metadata_xml"] = EncodeBase64(xml);
    }

    /// Adds another external resource that will be represented in the final
    /// consensusreadset.xml.
    void AddExternalResource (const ExternalResource& resource)
    {
        Json::Value jsonExternalResource;
        jsonExternalResource["file_name"] = resource.file_name;
        jsonExternalResource["meta_type"] = resource.meta_type;
        comment_["external_resources"].append(jsonExternalResource);
    }

    /// Set the name of the subread set
    void SetSubreadsetName(const std::string& name)
    {
        comment_["subreadset_name"] = name;
    }

    /// The JSON field is read-only.
    const Json::Value& GetJsonComment() const
    {
        return comment_;
    }

    /// The following two static functions should satisfy round-trip
    /// encoding and decoding.

    /// Encodes the plainText using base64 MIME-style encoding. This
    /// encoding converts 24 bits of binary data at a time (3 bytes) into 4 ASCII
    /// characters.  The end of the encoding is properly padded.
    /// \param plainText - text to encode into base64
    /// \returns encoded text
    static std::string EncodeBase64(const std::string& plainText);

    /// Decodes the encodedText using base64 MIME-style decoding. This
    /// decoding converts 4 ASCII characters at a time to 24 bits of binary data (3 bytes)
    /// characters.  The end of the encoding padded with null characters.
    /// \param encodedText - text to decode using MIME base64
    /// \returns plain text
    static std::string DecodeBase64(const std::string& encodedText);

private:
    Json::Value comment_;
};



}}} // namespace
#endif //PA_POST_PRIMARY_BAM_BAM_COMMENT_SIDEBAND_H
