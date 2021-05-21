//
// Created by mlakata on 6/18/20.
//

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <postprimary/bam/BamCommentSideband.h>


#include <json/json.h>

using namespace PacBio::Primary::Postprimary;

// sample files
extern const char* encodedMetadataXml;
extern const char* plainTextMetadataXml;

TEST(BamCommentSideband, EncodeBase64)
{
    using namespace std::string_literals;

    EXPECT_EQ("YWJjZGVmZ2gxMjM=",BamCommentSideband::EncodeBase64("abcdefgh123"));
    EXPECT_EQ("AAECAwQ=",BamCommentSideband::EncodeBase64("\000\001\002\003\004"s));
    EXPECT_EQ(encodedMetadataXml, BamCommentSideband::EncodeBase64(plainTextMetadataXml));
}


TEST(BamCommentSideband, DecodeBase64)
{
    using namespace std::string_literals;

    EXPECT_EQ("abcdefgh123",BamCommentSideband::DecodeBase64("YWJjZGVmZ2gxMjM="));
    EXPECT_EQ("\000\001\002\003\004"s,BamCommentSideband::DecodeBase64("AAECAwQ="));
    EXPECT_EQ(plainTextMetadataXml, BamCommentSideband::DecodeBase64(encodedMetadataXml));
}


TEST(BamCommentSideband, BasicUse)
{
    BamCommentSideband bcs;

    BamCommentSideband::ExternalResource resource;
    resource.file_name = "/foo/bar.bam";
    resource.meta_type = "experimental";
    bcs.AddExternalResource(resource);
    bcs.SetNumZmws(123);
    bcs.SetCollectionMetadataXML("abcdefgh123");
    bcs.SetDatasetUUID("0d045cc0-9929-4975-ac64-84922b1bd17a");
    /// Set the name of the subread set
    bcs.SetSubreadsetName("TAK1337 well sample-Cell1");


    const Json::Value j = bcs.GetJsonComment();

    EXPECT_EQ("1.2", j["version"].asString());
    ASSERT_EQ(1, j["external_resources"].size());
    const Json::Value& extRes = j["external_resources"][0];

    EXPECT_EQ("experimental", extRes["meta_type"].asString()) << bcs.GetJsonComment();
    EXPECT_EQ("/foo/bar.bam", extRes["file_name"].asString()) << bcs.GetJsonComment();

    EXPECT_EQ("0d045cc0-9929-4975-ac64-84922b1bd17a", j["dataset_uuid"].asString());
    EXPECT_EQ("YWJjZGVmZ2gxMjM=", j["collection_metadata_xml"].asString());
    EXPECT_EQ("TAK1337 well sample-Cell1", j["subreadset_name"].asString());
}

const char* encodedMetadataXml =
        "PENvbGxlY3Rpb25NZXRhZGF0YSBDcmVhdGVkQXQ9IjIwMTgtMDktMDdUMT"
        "U6MzY6NTkuOTE5WiIgTW9kaWZpZWRBdD0iMDAwMS0wMS0wMVQwMDowMDowMC"
        "IgVW5pcXVlSWQ9IjBkMDQ1Y2MwLTk5MjktNDk3NS1hYzY0LTg0OTIyYjFiZD"
        "E3YSIgTWV0YVR5cGU9IkNvbGxlY3Rpb25NZXRhZGF0YSIgVGltZVN0YW1wZW"
        "ROYW1lPSI1NDIzOC1Db2xsZWN0aW9uTWV0YWRhdGEtMjAxOC01NC0wN1QxNj"
        "o1NDoxMi4zNDZaIiBTdGF0dXM9IlJlYWR5IiBJbnN0cnVtZW50SWQ9IjU0Mj"
        "M4IiBJbnN0cnVtZW50TmFtZT0iNTQyMzgiIENvbnRleHQ9InRpbnkiPgogIC"
        "AgPEluc3RDdHJsVmVyPjYuMC4wLjQ1NjE2PC9JbnN0Q3RybFZlcj4KICAgID"
        "xTaWdQcm9jVmVyPjYuMC4wLjQ1MzAwPC9TaWdQcm9jVmVyPgogICAgPFJ1bk"
        "RldGFpbHM+CiAgICAgICAgPFRpbWVTdGFtcGVkTmFtZT5yNTQyMzhfMjAxOD"
        "A5MDdfMTY1NDEyPC9UaW1lU3RhbXBlZE5hbWU+CiAgICAgICAgPE5hbWU+Mj"
        "AxOC0wOS0wN19IRzJfcjU0MjM4PC9OYW1lPgogICAgICAgIDxDcmVhdGVkQn"
        "k+cHBlbHVzbzwvQ3JlYXRlZEJ5PgogICAgICAgIDxXaGVuQ3JlYXRlZD4yMD"
        "E4LTA5LTA3VDE1OjM2OjU5LjkxOVo8L1doZW5DcmVhdGVkPgogICAgICAgID"
        "xTdGFydGVkQnk+dW5rbm93bjwvU3RhcnRlZEJ5PgogICAgICAgIDxXaGVuU3"
        "RhcnRlZD4wMDAxLTAxLTAxVDAwOjAwOjAwPC9XaGVuU3RhcnRlZD4KICAgID"
        "wvUnVuRGV0YWlscz4KICAgIDxXZWxsU2FtcGxlIE5hbWU9IkhHMl9TQnYyX0"
        "VMRjE1a2JBXzRwTSIgQ3JlYXRlZEF0PSIyMDE4LTA5LTA3VDE1OjM2OjU5Lj"
        "kxOVoiIE1vZGlmaWVkQXQ9IjAwMDEtMDEtMDFUMDA6MDA6MDAiPgogICAgIC"
        "AgIDxXZWxsTmFtZT5CMDE8L1dlbGxOYW1lPgogICAgICAgIDxDb25jZW50cm"
        "F0aW9uPjA8L0NvbmNlbnRyYXRpb24+CiAgICAgICAgPE9uUGxhdGVMb2FkaW"
        "5nQ29uY2VudHJhdGlvbj4wPC9PblBsYXRlTG9hZGluZ0NvbmNlbnRyYXRpb2"
        "4+CiAgICAgICAgPEluc2VydFNpemU+MTUwMDA8L0luc2VydFNpemU+CiAgIC"
        "AgICAgPFNhbXBsZVJldXNlRW5hYmxlZD5mYWxzZTwvU2FtcGxlUmV1c2VFbm"
        "FibGVkPgogICAgICAgIDxTdGFnZUhvdHN0YXJ0RW5hYmxlZD5mYWxzZTwvU3"
        "RhZ2VIb3RzdGFydEVuYWJsZWQ+CiAgICAgICAgPFNpemVTZWxlY3Rpb25Fbm"
        "FibGVkPmZhbHNlPC9TaXplU2VsZWN0aW9uRW5hYmxlZD4KICAgICAgICA8VX"
        "NlQ291bnQ+MDwvVXNlQ291bnQ+CiAgICAgICAgPEJpb1NhbXBsZXMgeG1sbn"
        "M9Imh0dHA6Ly9wYWNpZmljYmlvc2NpZW5jZXMuY29tL1BhY0Jpb1NhbXBsZU"
        "luZm8ueHNkIj4KICAgICAgICAgICAgPEJpb1NhbXBsZSBOYW1lPSJIRzJfU0"
        "J2Ml9FTEYxNWtiQV80cE0iIC8+CiAgICAgICAgPC9CaW9TYW1wbGVzPgogIC"
        "AgPC9XZWxsU2FtcGxlPgogICAgPEF1dG9tYXRpb24gTmFtZT0iV29ya2Zsb3"
        "dfRGlmZnVzaW9uLnB5Ij4KICAgICAgICA8QXV0b21hdGlvblBhcmFtZXRlcn"
        "MgeG1sbnM9Imh0dHA6Ly9wYWNpZmljYmlvc2NpZW5jZXMuY29tL1BhY0Jpb0"
        "Jhc2VEYXRhTW9kZWwueHNkIj4KICAgICAgICAgICAgPEF1dG9tYXRpb25QYX"
        "JhbWV0ZXIgTmFtZT0iTW92aWVMZW5ndGgiIENyZWF0ZWRBdD0iMjAxOC0wOS"
        "0wN1QxNTozNjo1OS45MTlaIiBNb2RpZmllZEF0PSIwMDAxLTAxLTAxVDAwOj"
        "AwOjAwIiBWYWx1ZURhdGFUeXBlPSJEb3VibGUiIFNpbXBsZVZhbHVlPSIxND"
        "QwIiAvPgogICAgICAgICAgICA8QXV0b21hdGlvblBhcmFtZXRlciBOYW1lPS"
        "JFeHRlbnNpb25UaW1lIiBDcmVhdGVkQXQ9IjIwMTgtMDktMDdUMTU6MzY6NT"
        "kuOTE5WiIgTW9kaWZpZWRBdD0iMDAwMS0wMS0wMVQwMDowMDowMCIgVmFsdW"
        "VEYXRhVHlwZT0iRG91YmxlIiBTaW1wbGVWYWx1ZT0iNzIwIiAvPgogICAgIC"
        "AgICAgICA8QXV0b21hdGlvblBhcmFtZXRlciBOYW1lPSJFeHRlbmRGaXJzdC"
        "IgQ3JlYXRlZEF0PSIyMDE4LTA5LTA3VDE1OjM2OjU5LjkxOVoiIE1vZGlmaW"
        "VkQXQ9IjAwMDEtMDEtMDFUMDA6MDA6MDAiIFZhbHVlRGF0YVR5cGU9IkJvb2"
        "xlYW4iIFNpbXBsZVZhbHVlPSJUcnVlIiAvPgogICAgICAgICAgICA8QXV0b2"
        "1hdGlvblBhcmFtZXRlciBOYW1lPSJSZXVzZUNlbGwiIENyZWF0ZWRBdD0iMj"
        "AxOC0wOS0wN1QxNTozNjo1OS45MTlaIiBNb2RpZmllZEF0PSIwMDAxLTAxLT"
        "AxVDAwOjAwOjAwIiBWYWx1ZURhdGFUeXBlPSJCb29sZWFuIiBTaW1wbGVWYW"
        "x1ZT0iRmFsc2UiIC8+CiAgICAgICAgICAgIDxBdXRvbWF0aW9uUGFyYW1ldG"
        "VyIE5hbWU9IkltbW9iaWxpemF0aW9uVGltZSIgVmFsdWVEYXRhVHlwZT0iRG"
        "91YmxlIiBTaW1wbGVWYWx1ZT0iMTIwLjAiIC8+CiAgICAgICAgICAgIDxBdX"
        "RvbWF0aW9uUGFyYW1ldGVyIE5hbWU9IkNlbGxORkNJbmRleCIgVmFsdWVEYX"
        "RhVHlwZT0iSW50MzIiIFNpbXBsZVZhbHVlPSIxIiAvPgogICAgICAgICAgIC"
        "A8QXV0b21hdGlvblBhcmFtZXRlciBOYW1lPSJFeHRyYUlNV2FzaGVzIiBWYW"
        "x1ZURhdGFUeXBlPSJJbnQzMiIgU2ltcGxlVmFsdWU9IjIiIC8+CiAgICAgIC"
        "AgICAgIDxBdXRvbWF0aW9uUGFyYW1ldGVyIE5hbWU9IkV4cG9zdXJlIiBWYW"
        "x1ZURhdGFUeXBlPSJEb3VibGUiIFNpbXBsZVZhbHVlPSIwLjAxIiAvPgogIC"
        "AgICAgICAgICA8QXV0b21hdGlvblBhcmFtZXRlciBOYW1lPSJQQ0RpblBsYX"
        "RlIiBWYWx1ZURhdGFUeXBlPSJCb29sZWFuIiBTaW1wbGVWYWx1ZT0iVHJ1ZS"
        "IgLz4KICAgICAgICAgICAgPEF1dG9tYXRpb25QYXJhbWV0ZXIgTmFtZT0iUH"
        "JlRXh0ZW5zaW9uV29ya2Zsb3ciIFZhbHVlRGF0YVR5cGU9IkJvb2xlYW4iIF"
        "NpbXBsZVZhbHVlPSJUcnVlIiAvPgogICAgICAgICAgICA8QXV0b21hdGlvbl"
        "BhcmFtZXRlciBOYW1lPSJDb2xsZWN0aW9uTnVtYmVyIiBWYWx1ZURhdGFUeX"
        "BlPSJJbnQzMiIgU2ltcGxlVmFsdWU9IjEiIC8+CiAgICAgICAgICAgIDxBdX"
        "RvbWF0aW9uUGFyYW1ldGVyIE5hbWU9IlVzZVN0YWdlSG90U3RhcnQiIFZhbH"
        "VlRGF0YVR5cGU9IkJvb2xlYW4iIFNpbXBsZVZhbHVlPSJGYWxzZSIgLz4KIC"
        "AgICAgICAgICAgPEF1dG9tYXRpb25QYXJhbWV0ZXIgTmFtZT0iSW5zZXJ0U2"
        "l6ZSIgVmFsdWVEYXRhVHlwZT0iSW50MzIiIFNpbXBsZVZhbHVlPSIxNTAwMC"
        "IgLz4KICAgICAgICAgICAgPEF1dG9tYXRpb25QYXJhbWV0ZXIgTmFtZT0iSG"
        "FzTjJTd2l0Y2giIFZhbHVlRGF0YVR5cGU9IkJvb2xlYW4iIFNpbXBsZVZhbH"
        "VlPSJUcnVlIiAvPgogICAgICAgICAgICA8QXV0b21hdGlvblBhcmFtZXRlci"
        "BOYW1lPSJUaXBTZWFyY2hNYXhEdXJhdGlvbiIgVmFsdWVEYXRhVHlwZT0iSW"
        "50MzIiIFNpbXBsZVZhbHVlPSI1NzYiIC8+CiAgICAgICAgICAgIDxBdXRvbW"
        "F0aW9uUGFyYW1ldGVyIE5hbWU9IlNOUkN1dCIgVmFsdWVEYXRhVHlwZT0iRG"
        "91YmxlIiBTaW1wbGVWYWx1ZT0iMy43NSIgLz4KICAgICAgICA8L0F1dG9tYX"
        "Rpb25QYXJhbWV0ZXJzPgogICAgPC9BdXRvbWF0aW9uPgogICAgPENvbGxlY3"
        "Rpb25OdW1iZXI+MTwvQ29sbGVjdGlvbk51bWJlcj4KICAgIDxDZWxsSW5kZX"
        "g+MTwvQ2VsbEluZGV4PgogICAgPFNldE51bWJlcj4wPC9TZXROdW1iZXI+Ci"
        "AgICA8Q2VsbFBhYyBOYW1lPSJTTVJUwq4gQ2VsbCAxTSB2MyBMUiAoNC9QYW"
        "NrKSIgRGVzY3JpcHRpb249IkluZGl2aWR1YWwgNCBQYWNrIGNvbnRhaW5pbm"
        "cgNCBTTVJUwq5DZWxscyBlYWNoIGNvbnRhaW5pbmcgMSBtaWxsaW9uIFpNV3"
        "MiIFZlcnNpb249IjMuMCIgUGFydE51bWJlcj0iMTAxLTUzMS0wMDEiIExvdE"
        "51bWJlcj0iMzI0MjM3IiBCYXJjb2RlPSJCQTI0MzQ0MSIgRXhwaXJhdGlvbk"
        "RhdGU9IjIwMTktMDQtMTYiIE1vdmllVGltZUdyYWRlPSJMUiI+CiAgICAgIC"
        "AgPENoaXBMYXlvdXQgeG1sbnM9Imh0dHA6Ly9wYWNpZmljYmlvc2NpZW5jZX"
        "MuY29tL1BhY0Jpb0Jhc2VEYXRhTW9kZWwueHNkIj5TZXF1RUxfNC4wX1JUTz"
        "M8L0NoaXBMYXlvdXQ+CiAgICA8L0NlbGxQYWM+CiAgICA8VGVtcGxhdGVQcm"
        "VwS2l0IE5hbWU9IlNNUlRiZWxswq4gVGVtcGxhdGUgUHJlcCBLaXQgMS4wIi"
        "BEZXNjcmlwdGlvbj0iVGhlIFNNUlRiZWxswq4gVGVtcGxhdGUgUHJlcCBLaX"
        "QgY29udGFpbnMgcmVhZ2VudCBzdXBwbGllcyB0byBwZXJmb3JtIFNNUlRiZW"
        "xsIGxpYnJhcnkgcHJlcGFyYXRpb25zIG9mIHByaW1lci1hbm5lYWxlZCBTTV"
        "JUYmVsbCBsaWJyYXJpZXMgZm9yIGluc2VydCBzaXplcyByYW5naW5nIGZyb2"
        "0gNTAwIGJwIHRvIG92ZXIgMjAga2IuIiBUYWdzPSJUZW1wbGF0ZSBQcmVwIE"
        "tpdCwgVFBLIiBWZXJzaW9uPSIxLjAiIFBhcnROdW1iZXI9IjEwMC0yNTktMT"
        "AwIiBMb3ROdW1iZXI9IkRNMTIzNCIgQmFyY29kZT0iRE0xMjM0MTAwMjU5MT"
        "AwMTIzMTIwIiBFeHBpcmF0aW9uRGF0ZT0iMjAyMC0xMi0zMSIgTWluSW5zZX"
        "J0U2l6ZT0iNTAwIiBNYXhJbnNlcnRTaXplPSIyMDAwMCI+CiAgICAgICAgPE"
        "xlZnRBZGFwdG9yU2VxdWVuY2UgeG1sbnM9Imh0dHA6Ly9wYWNpZmljYmlvc2"
        "NpZW5jZXMuY29tL1BhY0Jpb0Jhc2VEYXRhTW9kZWwueHNkIj5BVENUQ1RDVE"
        "NBQUNBQUNBQUNBQUNHR0FHR0FHR0FHR0FBQUFHQUdBR0FHQVQ8L0xlZnRBZG"
        "FwdG9yU2VxdWVuY2U+CiAgICAgICAgPExlZnRQcmltZXJTZXF1ZW5jZSB4bW"
        "xucz0iaHR0cDovL3BhY2lmaWNiaW9zY2llbmNlcy5jb20vUGFjQmlvQmFzZU"
        "RhdGFNb2RlbC54c2QiPmFhY2dnYWdnYWdnYWdnYTwvTGVmdFByaW1lclNlcX"
        "VlbmNlPgogICAgICAgIDxSaWdodEFkYXB0b3JTZXF1ZW5jZSB4bWxucz0iaH"
        "R0cDovL3BhY2lmaWNiaW9zY2llbmNlcy5jb20vUGFjQmlvQmFzZURhdGFNb2"
        "RlbC54c2QiPkFUQ1RDVENUQ0FBQ0FBQ0FBQ0FBQ0dHQUdHQUdHQUdHQUFBQU"
        "dBR0FHQUdBVDwvUmlnaHRBZGFwdG9yU2VxdWVuY2U+CiAgICAgICAgPFJpZ2"
        "h0UHJpbWVyU2VxdWVuY2UgeG1sbnM9Imh0dHA6Ly9wYWNpZmljYmlvc2NpZW"
        "5jZXMuY29tL1BhY0Jpb0Jhc2VEYXRhTW9kZWwueHNkIj5hYWNnZ2FnZ2FnZ2"
        "FnZ2E8L1JpZ2h0UHJpbWVyU2VxdWVuY2U+CiAgICA8L1RlbXBsYXRlUHJlcE"
        "tpdD4KICAgIDxCaW5kaW5nS2l0IE5hbWU9IlNlcXVlbMKuIEJpbmRpbmcgS2"
        "l0IDMuMCIgRGVzY3JpcHRpb249IlRoZSBTZXF1ZWwgQmluZGluZyBLaXQgMy"
        "4wIGNvbnRhaW5zIHJlYWdlbnQgc3VwcGxpZXMgdG8gYmluZCBwcmVwYXJlZC"
        "BETkEgdGVtcGxhdGUgbGlicmFyaWVzIHRvIHRoZSBTZXF1ZWwgUG9seW1lcm"
        "FzZSAzLjAgaW4gcHJlcGFyYXRpb24gZm9yIHNlcXVlbmNpbmcgb24gdGhlIF"
        "NlcXVlbCBTeXN0ZW0uIFRoZSByZXN1bHQgaXMgYSBETkEgcG9seW1lcmFzZS"
        "90ZW1wbGF0ZSBjb21wbGV4LiBTZXF1ZWwgQmluZGluZyBLaXQgMy4wIHNob3"
        "VsZCBiZSB1c2VkIG9ubHkgd2l0aCBTZXF1ZWwgU2VxdWVuY2luZyBLaXQgMy"
        "4wLiBSZWFnZW50IHF1YW50aXRpZXMgc3VwcG9ydCAyNCBiaW5kaW5nIHJlYW"
        "N0aW9ucy4iIFRhZ3M9IkJpbmRpbmcgS2l0LCBCREsiIFZlcnNpb249IjMuMC"
        "IgUGFydE51bWJlcj0iMTAxLTUwMC00MDAiIExvdE51bWJlcj0iRE0xMjM0Ii"
        "BCYXJjb2RlPSJETTEyMzQxMDE1MDA0MDAxMjMxMjAiIEV4cGlyYXRpb25EYX"
        "RlPSIyMDIwLTEyLTMxIiBDaGlwVHlwZT0iMW1DaGlwIiAvPgogICAgPFNlcX"
        "VlbmNpbmdLaXRQbGF0ZSBOYW1lPSJTZXF1ZWzCriBTZXF1ZW5jaW5nIFBsYX"
        "RlIDMuMCAoNCByeG4pIiBEZXNjcmlwdGlvbj0iVGhlIEROQSBTZXF1ZW5jaW"
        "5nIEtpdCBjb250YWlucyBhIHNlcXVlbmNpbmcgcmVhZ2VudCBwbGF0ZSB3aX"
        "RoIGNoZW1pc3RyeSBmb3Igc2luZ2xlIG1vbGVjdWxlIHJlYWwtdGltZSBzZX"
        "F1ZW5jaW5nIG9uIHRoZSBQYWNCaW8gU2VxdWVswq4uIFJlYWdlbnQgcXVhbn"
        "RpdGllcyBzdXBwb3J0IDQgc2VxdWVuY2luZyByZWFjdGlvbnMgdG8gYmUgdX"
        "NlZCBpbiBjb25qdW5jdGlvbiB3aXRoIFNNUlTCriBDZWxsIDRQYWMocykuIC"
        "AoNCBDZWxscyBtYXgvRWFjaCBSb3cgc3VwcGxpZXMgcmVhZ2VudHMgZm9yID"
        "EgU2VxdWVsIFNNUlQgQ2VsbCkiIFRhZ3M9IlNlcXVlbmNpbmcgS2l0LCBTUU"
        "siIFZlcnNpb249IjMuMCIgUGFydE51bWJlcj0iMTAxLTQyNy04MDAiIExvdE"
        "51bWJlcj0iMDEzMTYxIiBCYXJjb2RlPSIwMTMxNjExMDE0Mjc4MDAwNDI2MT"
        "kiIEV4cGlyYXRpb25EYXRlPSIyMDE5LTA0LTI2IiBDaGlwVHlwZT0iMW1DaG"
        "lwIiBNYXhDb2xsZWN0aW9ucz0iNCIgTnVtT3NlVHViZXM9IjAiPgogICAgIC"
        "AgIDxSZWFnZW50VHViZXMgTmFtZT0iU2VxdWVswq4gU01SVMKuQ2VsbCBPaW"
        "wiIFBhcnROdW1iZXI9IjEwMC02MTktNjAwIiBMb3ROdW1iZXI9IjAxMjcxMi"
        "IgQmFyY29kZT0iMDEyNzEyMTAwNjE5NjAwMDMzMTIyIiBFeHBpcmF0aW9uRG"
        "F0ZT0iMjAyMi0wMy0zMSIgCiAgICAgICAgICAgIHhtbG5zPSJodHRwOi8vcG"
        "FjaWZpY2Jpb3NjaWVuY2VzLmNvbS9QYWNCaW9SZWFnZW50S2l0LnhzZCIgLz"
        "4KICAgIDwvU2VxdWVuY2luZ0tpdFBsYXRlPgogICAgPFByaW1hcnk+CiAgIC"
        "AgICAgPEF1dG9tYXRpb25OYW1lPlNlcXVlbEFscGhhPC9BdXRvbWF0aW9uTm"
        "FtZT4KICAgICAgICA8Q29uZmlnRmlsZU5hbWU+U3FsUG9DX1N1YkNyZl8yQz"
        "JBLXQyLnhtbDwvQ29uZmlnRmlsZU5hbWU+CiAgICAgICAgPFNlcXVlbmNpbm"
        "dDb25kaXRpb24+RGVmYXVsdFByaW1hcnlTZXF1ZW5jaW5nQ29uZGl0aW9uPC"
        "9TZXF1ZW5jaW5nQ29uZGl0aW9uPgogICAgICAgIDxPdXRwdXRPcHRpb25zPg"
        "ogICAgICAgICAgICA8UmVzdWx0c0ZvbGRlcj4zMTQvUFBlbHVzby9IRzJfMT"
        "VrYi9yNTQyMzhfMjAxODA5MDdfMTY1NDEyLzJfQjAxLzwvUmVzdWx0c0ZvbG"
        "Rlcj4KICAgICAgICAgICAgPENvbGxlY3Rpb25QYXRoVXJpPi9wYmkvY29sbG"
        "VjdGlvbnMvMzE0L1BQZWx1c28vSEcyXzE1a2IvcjU0MjM4XzIwMTgwOTA3Xz"
        "E2NTQxMi8yX0IwMS88L0NvbGxlY3Rpb25QYXRoVXJpPgogICAgICAgICAgIC"
        "A8Q29weUZpbGVzPgogICAgICAgICAgICAgICAgPENvbGxlY3Rpb25GaWxlQ2"
        "9weT5GYXN0YTwvQ29sbGVjdGlvbkZpbGVDb3B5PgogICAgICAgICAgICAgIC"
        "AgPENvbGxlY3Rpb25GaWxlQ29weT5CYW08L0NvbGxlY3Rpb25GaWxlQ29weT"
        "4KICAgICAgICAgICAgPC9Db3B5RmlsZXM+CiAgICAgICAgICAgIDxSZWFkb3"
        "V0PkJhc2VzX1dpdGhvdXRfUVZzPC9SZWFkb3V0PgogICAgICAgICAgICA8TW"
        "V0cmljc1ZlcmJvc2l0eT5NaW5pbWFsPC9NZXRyaWNzVmVyYm9zaXR5PgogIC"
        "AgICAgICAgICA8VHJhbnNmZXJSZXNvdXJjZT4KICAgICAgICAgICAgICAgID"
        "xJZD5yc3luYy1wYmktY29sbGVjdGlvbnM8L0lkPgogICAgICAgICAgICAgIC"
        "AgPFRyYW5zZmVyU2NoZW1lPlJTWU5DPC9UcmFuc2ZlclNjaGVtZT4KICAgIC"
        "AgICAgICAgICAgIDxOYW1lPlBCSSBDb2xsZWN0aW9ucyBSc3luYzwvTmFtZT"
        "4KICAgICAgICAgICAgICAgIDxEZXNjcmlwdGlvbj5Mb2NhdGlvbiBmb3Igd3"
        "JpdGluZyBUcmFuc2ZlciBzZXJ2aWNlcyB0byB3cml0ZSB0by4gRm9yIHRlc3"
        "RpbmcsIEludGVybmFsIHRvb2xzIChQQSBTSU0gYW5kIElDUykgdGVzdHMgbX"
        "VzdCBleHBsaWNpdGx5IHNldCB0aGUgcmVsYXRpdmUgcGF0aCBwcmVmaXggdG"
        "8gJ3hmZXItdGVzdCc8L0Rlc2NyaXB0aW9uPgogICAgICAgICAgICAgICAgPE"
        "Rlc3RQYXRoPi9wYmkvY29sbGVjdGlvbnM8L0Rlc3RQYXRoPgogICAgICAgIC"
        "AgICA8L1RyYW5zZmVyUmVzb3VyY2U+CiAgICAgICAgPC9PdXRwdXRPcHRpb2"
        "5zPgogICAgPC9QcmltYXJ5PgogICAgPFNlY29uZGFyeT4KICAgICAgICA8QX"
        "V0b21hdGlvbk5hbWU+RGVmYXVsdFNlY29uZGFyeUF1dG9tYXRpb25OYW1lPC"
        "9BdXRvbWF0aW9uTmFtZT4KICAgICAgICA8QXV0b21hdGlvblBhcmFtZXRlcn"
        "M+CiAgICAgICAgICAgIDxBdXRvbWF0aW9uUGFyYW1ldGVyIE5hbWU9IlJlZm"
        "VyZW5jZSIgQ3JlYXRlZEF0PSIwMDAxLTAxLTAxVDAwOjAwOjAwIiBNb2RpZm"
        "llZEF0PSIwMDAxLTAxLTAxVDAwOjAwOjAwIiBWYWx1ZURhdGFUeXBlPSJTdH"
        "JpbmciIFNpbXBsZVZhbHVlPSJEZWZhdWx0U2Vjb25kYXJ5QW5hbHlzaXNSZW"
        "ZlcmVuY2VOYW1lIiAvPgogICAgICAgIDwvQXV0b21hdGlvblBhcmFtZXRlcn"
        "M+CiAgICAgICAgPENlbGxDb3VudEluSm9iPjA8L0NlbGxDb3VudEluSm9iPg"
        "ogICAgPC9TZWNvbmRhcnk+CiAgICA8VXNlckRlZmluZWRGaWVsZHM+CiAgIC"
        "AgICAgPERhdGFFbnRpdGllcyBOYW1lPSIgTElNU19JTVBPUlQgIiBWYWx1ZU"
        "RhdGFUeXBlPSJTdHJpbmciIFNpbXBsZVZhbHVlPSJEZWZhdWx0VXNlckRlZm"
        "luZWRGaWVsZExJTVMiIAogICAgICAgICAgICB4bWxucz0iaHR0cDovL3BhY2"
        "lmaWNiaW9zY2llbmNlcy5jb20vUGFjQmlvQmFzZURhdGFNb2RlbC54c2QiIC"
        "8+CiAgICA8L1VzZXJEZWZpbmVkRmllbGRzPgogICAgPENvbXBvbmVudFZlcn"
        "Npb25zPgogICAgICAgIDxWZXJzaW9uSW5mbyBOYW1lPSJpY3MiIFZlcnNpb2"
        "49IjYuMC4wLjQ1NjE2IiAvPgogICAgICAgIDxWZXJzaW9uSW5mbyBOYW1lPS"
        "JpdWkiIFZlcnNpb249IjYuMC4wLjQ1NjE2IiAvPgogICAgICAgIDxWZXJzaW"
        "9uSW5mbyBOYW1lPSJjaGVtaXN0cnkiIFZlcnNpb249IjYuMC4wLjQ1MTExIi"
        "AvPgogICAgICAgIDxWZXJzaW9uSW5mbyBOYW1lPSJwYSIgVmVyc2lvbj0iNi"
        "4wLjAuNDUzMDAiIC8+CiAgICAgICAgPFZlcnNpb25JbmZvIE5hbWU9InBhd3"
        "MiIFZlcnNpb249IjYuMC4wLjQ1MzAwIiAvPgogICAgICAgIDxWZXJzaW9uSW"
        "5mbyBOYW1lPSJwcGEiIFZlcnNpb249IjYuMC4wLjQ1MzAwIiAvPgogICAgIC"
        "AgIDxWZXJzaW9uSW5mbyBOYW1lPSJyZWFsdGltZSIgVmVyc2lvbj0iNi4wLj"
        "AuNDUzMDAiIC8+CiAgICAgICAgPFZlcnNpb25JbmZvIE5hbWU9InRyYW5zZm"
        "VyIiBWZXJzaW9uPSI2LjAuMC40NTMwMCIgLz4KICAgICAgICA8VmVyc2lvbk"
        "luZm8gTmFtZT0ic21ydGxpbmstYW5hbHlzaXNzZXJ2aWNlcy1ndWkiIFZlcn"
        "Npb249IjYuMC4wLjQ1NjE4IiAvPgogICAgICAgIDxWZXJzaW9uSW5mbyBOYW"
        "1lPSJzbXJ0aW1pc2MiIFZlcnNpb249IjYuMC4wLjQ1NjIxIiAvPgogICAgIC"
        "AgIDxWZXJzaW9uSW5mbyBOYW1lPSJzbXJ0bGluayIgVmVyc2lvbj0iNi4wLj"
        "AuNDU2MjEiIC8+CiAgICAgICAgPFZlcnNpb25JbmZvIE5hbWU9InNtcnR0b2"
        "9scyIgVmVyc2lvbj0iNi4wLjAuNDU1ODAiIC8+CiAgICAgICAgPFZlcnNpb2"
        "5JbmZvIE5hbWU9InNtcnRpbnViIiBWZXJzaW9uPSI2LjAuMC40NTU4MCIgLz"
        "4KICAgICAgICA8VmVyc2lvbkluZm8gTmFtZT0ic21ydHZpZXciIFZlcnNpb2"
        "49IjYuMC4wLjQ1NTgwIiAvPgogICAgPC9Db21wb25lbnRWZXJzaW9ucz4KPC"
        "9Db2xsZWN0aW9uTWV0YWRhdGE+";

const char* plainTextMetadataXml = R"GTESTXML(<CollectionMetadata CreatedAt="2018-09-07T15:36:59.919Z" ModifiedAt="0001-01-01T00:00:00" UniqueId="0d045cc0-9929-4975-ac64-84922b1bd17a" MetaType="CollectionMetadata" TimeStampedName="54238-CollectionMetadata-2018-54-07T16:54:12.346Z" Status="Ready" InstrumentId="54238" InstrumentName="54238" Context="tiny">
    <InstCtrlVer>6.0.0.45616</InstCtrlVer>
    <SigProcVer>6.0.0.45300</SigProcVer>
    <RunDetails>
        <TimeStampedName>r54238_20180907_165412</TimeStampedName>
        <Name>2018-09-07_HG2_r54238</Name>
        <CreatedBy>ppeluso</CreatedBy>
        <WhenCreated>2018-09-07T15:36:59.919Z</WhenCreated>
        <StartedBy>unknown</StartedBy>
        <WhenStarted>0001-01-01T00:00:00</WhenStarted>
    </RunDetails>
    <WellSample Name="HG2_SBv2_ELF15kbA_4pM" CreatedAt="2018-09-07T15:36:59.919Z" ModifiedAt="0001-01-01T00:00:00">
        <WellName>B01</WellName>
        <Concentration>0</Concentration>
        <OnPlateLoadingConcentration>0</OnPlateLoadingConcentration>
        <InsertSize>15000</InsertSize>
        <SampleReuseEnabled>false</SampleReuseEnabled>
        <StageHotstartEnabled>false</StageHotstartEnabled>
        <SizeSelectionEnabled>false</SizeSelectionEnabled>
        <UseCount>0</UseCount>
        <BioSamples xmlns="http://pacificbiosciences.com/PacBioSampleInfo.xsd">
            <BioSample Name="HG2_SBv2_ELF15kbA_4pM" />
        </BioSamples>
    </WellSample>
    <Automation Name="Workflow_Diffusion.py">
        <AutomationParameters xmlns="http://pacificbiosciences.com/PacBioBaseDataModel.xsd">
            <AutomationParameter Name="MovieLength" CreatedAt="2018-09-07T15:36:59.919Z" ModifiedAt="0001-01-01T00:00:00" ValueDataType="Double" SimpleValue="1440" />
            <AutomationParameter Name="ExtensionTime" CreatedAt="2018-09-07T15:36:59.919Z" ModifiedAt="0001-01-01T00:00:00" ValueDataType="Double" SimpleValue="720" />
            <AutomationParameter Name="ExtendFirst" CreatedAt="2018-09-07T15:36:59.919Z" ModifiedAt="0001-01-01T00:00:00" ValueDataType="Boolean" SimpleValue="True" />
            <AutomationParameter Name="ReuseCell" CreatedAt="2018-09-07T15:36:59.919Z" ModifiedAt="0001-01-01T00:00:00" ValueDataType="Boolean" SimpleValue="False" />
            <AutomationParameter Name="ImmobilizationTime" ValueDataType="Double" SimpleValue="120.0" />
            <AutomationParameter Name="CellNFCIndex" ValueDataType="Int32" SimpleValue="1" />
            <AutomationParameter Name="ExtraIMWashes" ValueDataType="Int32" SimpleValue="2" />
            <AutomationParameter Name="Exposure" ValueDataType="Double" SimpleValue="0.01" />
            <AutomationParameter Name="PCDinPlate" ValueDataType="Boolean" SimpleValue="True" />
            <AutomationParameter Name="PreExtensionWorkflow" ValueDataType="Boolean" SimpleValue="True" />
            <AutomationParameter Name="CollectionNumber" ValueDataType="Int32" SimpleValue="1" />
            <AutomationParameter Name="UseStageHotStart" ValueDataType="Boolean" SimpleValue="False" />
            <AutomationParameter Name="InsertSize" ValueDataType="Int32" SimpleValue="15000" />
            <AutomationParameter Name="HasN2Switch" ValueDataType="Boolean" SimpleValue="True" />
            <AutomationParameter Name="TipSearchMaxDuration" ValueDataType="Int32" SimpleValue="576" />
            <AutomationParameter Name="SNRCut" ValueDataType="Double" SimpleValue="3.75" />
        </AutomationParameters>
    </Automation>
    <CollectionNumber>1</CollectionNumber>
    <CellIndex>1</CellIndex>
    <SetNumber>0</SetNumber>
    <CellPac Name="SMRT® Cell 1M v3 LR (4/Pack)" Description="Individual 4 Pack containing 4 SMRT®Cells each containing 1 million ZMWs" Version="3.0" PartNumber="101-531-001" LotNumber="324237" Barcode="BA243441" ExpirationDate="2019-04-16" MovieTimeGrade="LR">
        <ChipLayout xmlns="http://pacificbiosciences.com/PacBioBaseDataModel.xsd">SequEL_4.0_RTO3</ChipLayout>
    </CellPac>
    <TemplatePrepKit Name="SMRTbell® Template Prep Kit 1.0" Description="The SMRTbell® Template Prep Kit contains reagent supplies to perform SMRTbell library preparations of primer-annealed SMRTbell libraries for insert sizes ranging from 500 bp to over 20 kb." Tags="Template Prep Kit, TPK" Version="1.0" PartNumber="100-259-100" LotNumber="DM1234" Barcode="DM1234100259100123120" ExpirationDate="2020-12-31" MinInsertSize="500" MaxInsertSize="20000">
        <LeftAdaptorSequence xmlns="http://pacificbiosciences.com/PacBioBaseDataModel.xsd">ATCTCTCTCAACAACAACAACGGAGGAGGAGGAAAAGAGAGAGAT</LeftAdaptorSequence>
        <LeftPrimerSequence xmlns="http://pacificbiosciences.com/PacBioBaseDataModel.xsd">aacggaggaggagga</LeftPrimerSequence>
        <RightAdaptorSequence xmlns="http://pacificbiosciences.com/PacBioBaseDataModel.xsd">ATCTCTCTCAACAACAACAACGGAGGAGGAGGAAAAGAGAGAGAT</RightAdaptorSequence>
        <RightPrimerSequence xmlns="http://pacificbiosciences.com/PacBioBaseDataModel.xsd">aacggaggaggagga</RightPrimerSequence>
    </TemplatePrepKit>
    <BindingKit Name="Sequel® Binding Kit 3.0" Description="The Sequel Binding Kit 3.0 contains reagent supplies to bind prepared DNA template libraries to the Sequel Polymerase 3.0 in preparation for sequencing on the Sequel System. The result is a DNA polymerase/template complex. Sequel Binding Kit 3.0 should be used only with Sequel Sequencing Kit 3.0. Reagent quantities support 24 binding reactions." Tags="Binding Kit, BDK" Version="3.0" PartNumber="101-500-400" LotNumber="DM1234" Barcode="DM1234101500400123120" ExpirationDate="2020-12-31" ChipType="1mChip" />
    <SequencingKitPlate Name="Sequel® Sequencing Plate 3.0 (4 rxn)" Description="The DNA Sequencing Kit contains a sequencing reagent plate with chemistry for single molecule real-time sequencing on the PacBio Sequel®. Reagent quantities support 4 sequencing reactions to be used in conjunction with SMRT® Cell 4Pac(s).  (4 Cells max/Each Row supplies reagents for 1 Sequel SMRT Cell)" Tags="Sequencing Kit, SQK" Version="3.0" PartNumber="101-427-800" LotNumber="013161" Barcode="013161101427800042619" ExpirationDate="2019-04-26" ChipType="1mChip" MaxCollections="4" NumOseTubes="0">
        <ReagentTubes Name="Sequel® SMRT®Cell Oil" PartNumber="100-619-600" LotNumber="012712" Barcode="012712100619600033122" ExpirationDate="2022-03-31" 
            xmlns="http://pacificbiosciences.com/PacBioReagentKit.xsd" />
    </SequencingKitPlate>
    <Primary>
        <AutomationName>SequelAlpha</AutomationName>
        <ConfigFileName>SqlPoC_SubCrf_2C2A-t2.xml</ConfigFileName>
        <SequencingCondition>DefaultPrimarySequencingCondition</SequencingCondition>
        <OutputOptions>
            <ResultsFolder>314/PPeluso/HG2_15kb/r54238_20180907_165412/2_B01/</ResultsFolder>
            <CollectionPathUri>/pbi/collections/314/PPeluso/HG2_15kb/r54238_20180907_165412/2_B01/</CollectionPathUri>
            <CopyFiles>
                <CollectionFileCopy>Fasta</CollectionFileCopy>
                <CollectionFileCopy>Bam</CollectionFileCopy>
            </CopyFiles>
            <Readout>Bases_Without_QVs</Readout>
            <MetricsVerbosity>Minimal</MetricsVerbosity>
            <TransferResource>
                <Id>rsync-pbi-collections</Id>
                <TransferScheme>RSYNC</TransferScheme>
                <Name>PBI Collections Rsync</Name>
                <Description>Location for writing Transfer services to write to. For testing, Internal tools (PA SIM and ICS) tests must explicitly set the relative path prefix to 'xfer-test'</Description>
                <DestPath>/pbi/collections</DestPath>
            </TransferResource>
        </OutputOptions>
    </Primary>
    <Secondary>
        <AutomationName>DefaultSecondaryAutomationName</AutomationName>
        <AutomationParameters>
            <AutomationParameter Name="Reference" CreatedAt="0001-01-01T00:00:00" ModifiedAt="0001-01-01T00:00:00" ValueDataType="String" SimpleValue="DefaultSecondaryAnalysisReferenceName" />
        </AutomationParameters>
        <CellCountInJob>0</CellCountInJob>
    </Secondary>
    <UserDefinedFields>
        <DataEntities Name=" LIMS_IMPORT " ValueDataType="String" SimpleValue="DefaultUserDefinedFieldLIMS" 
            xmlns="http://pacificbiosciences.com/PacBioBaseDataModel.xsd" />
    </UserDefinedFields>
    <ComponentVersions>
        <VersionInfo Name="ics" Version="6.0.0.45616" />
        <VersionInfo Name="iui" Version="6.0.0.45616" />
        <VersionInfo Name="chemistry" Version="6.0.0.45111" />
        <VersionInfo Name="pa" Version="6.0.0.45300" />
        <VersionInfo Name="paws" Version="6.0.0.45300" />
        <VersionInfo Name="ppa" Version="6.0.0.45300" />
        <VersionInfo Name="realtime" Version="6.0.0.45300" />
        <VersionInfo Name="transfer" Version="6.0.0.45300" />
        <VersionInfo Name="smrtlink-analysisservices-gui" Version="6.0.0.45618" />
        <VersionInfo Name="smrtimisc" Version="6.0.0.45621" />
        <VersionInfo Name="smrtlink" Version="6.0.0.45621" />
        <VersionInfo Name="smrttools" Version="6.0.0.45580" />
        <VersionInfo Name="smrtinub" Version="6.0.0.45580" />
        <VersionInfo Name="smrtview" Version="6.0.0.45580" />
    </ComponentVersions>
</CollectionMetadata>)GTESTXML";
