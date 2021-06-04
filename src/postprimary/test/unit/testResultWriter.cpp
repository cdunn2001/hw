//
// Created by mlakata on 5/1/17.
//

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <pbbam/RunMetadata.h>
#include <pacbio/dev/gtest-extras.h>
#include <pacbio/dev/TemporaryDirectory.h>
#include <pacbio/System.h>
#include <pacbio/ipc/JSON.h>
#include <pacbio/text/PBXml.h>
#include <pacbio/text/String.h>

#include <bazio/FileHeaderBuilder.h>
#include <bazio/MetricData.h>

#include <postprimary/bam/ResultWriter.h>
#include <postprimary/bam/RuntimeMetaData.h>
#include <postprimary/application/MetadataParser.h>

#include <pbbam/BamReader.h>

#include "ReadSimulator.h"
#include "test_data_config.h"

using namespace PacBio::Primary;
using namespace PacBio::Primary::Postprimary;

const std::string SUBREADSET      = std::string(PacBio::PaPpaTestConfig::cmakeCurrentListDir) + "/data/subreadset.xml";

/// a wrapper class to expose protected members for the purpose of unit testing.
class ResultWriterEx : public ResultWriter
{
public:
    using ResultWriter::ResultWriter;
};

// nb. I will move this object to pa-common in another story, but I need it now.
/// A class to redirect an open file descriptor (such as stdout or stderr) to a named file, for the lifetime of the
/// object (RAII style).  When the object goes out of scope, the original file descriptor is restored.
/// Example usage:
///   {
///      TemporaryFileRedirection redirector(STDOUT_?, "/tmp/myfile");
///      std::cout << "here is some text" << std::endl;
///   }
///   std::string theText = slurp("/tmp/myfile");

class TemporaryFileRedirection
{
public:
    /// \param desc : suggested values: STDOUT_FILENO or STDERR_FILENO (which equate to 1 or 2)
    /// \param filename : the file to write to.  "/dev/null" is also useful for discarding text.
    TemporaryFileRedirection(int desc, const std::string& filename)
    : desc_(desc)
    , backupFd_(-1)
    , filename_(filename)
    {
        fflush(NULL); // flush all streams, including stdout.
        backupFd_ = dup(desc);
        if (backupFd_ == -1) throw PBExceptionErrno("dup failed with " + std::to_string(desc));
        int newFd = open(filename_.c_str(), O_WRONLY|O_CREAT, S_IRUSR|S_IWUSR );
        if (newFd == -1) throw PBExceptionErrno("open failed with " + filename_);
        if (dup2(newFd, desc) == -1) throw PBExceptionErrno("dup2 failed with newFd:"+ std::to_string(newFd));
        if (close(newFd) == -1) throw PBExceptionErrno("close failed");
    }
    ~TemporaryFileRedirection()
    {
        try
        {
            Restore();
        }
        catch(const std::exception& ex)
        {
            // write to both output streams, since it is unclear which stream is visible to the user at this point.
            std::cerr << "Fatal exception restoring output stream " << backupFd_ << " " << ex.what();
            std::cout << "Fatal exception restoring output stream " << backupFd_ << " " << ex.what();
        }
        catch(...)
        {
            std::cerr << "Fatal exception restoring output stream " << backupFd_;
            std::cout << "Fatal exception restoring output stream " << backupFd_;
        }
    }
    void Restore()
    {
        if (backupFd_ != -1)
        {
            fflush(NULL); // flush all streams, including stdout
            if (dup2(backupFd_, desc_) == -1)
                throw PBExceptionErrno("dup2 failed with backupFd_:" + std::to_string(backupFd_));
            if (close(backupFd_) == -1)
                throw PBExceptionErrno("close failed on backupFd_:" + std::to_string(backupFd_));
            backupFd_ = -1; // mark as already backed-up
        }
    }

    std::string Filename() const { return filename_; }

private:
    int desc_;
    int backupFd_;
    std::string filename_;
};


TEST(ResultWriter,Basics)
{
    PacBio::Logging::LogSeverityContext context(PacBio::Logging::LogLevel::WARN);

    PacBio::Dev::TemporaryDirectory tmpdir;
    tmpdir.Keep(true);

    UserParameters user;
    PpaAlgoConfig ppaAlgoConfig;
    user.savePbi = false;
    std::shared_ptr<RuntimeMetaData> rmd(new RuntimeMetaData);
    FileHeaderBuilder fhb("FakeMovie", 80.0, 80.0*60*60*3,
            Readout::BASES, MetricsVerbosity::MINIMAL,
            generateExperimentMetadata(),
            "{}", {0},{},1024,4096,16384);
    std::string header = fhb.CreateJSON();
    FileHeader fileHeader(header.c_str(), header.size());
    const std::vector<PacBio::BAM::ProgramInfo> apps;

    user.outputPrefix = tmpdir.DirName() + "/foo_";

    rmd->subreadSet.uniqueId = "9c428b0b-8d5d-439c-a0c4-16b6db7e3007";
    rmd->subreadSet.timeStampedName = "SubreadSetCollection_170416_01082744";
    rmd->subreadSet.createdAt = "2017-04-16T01:08:27Z";
    rmd->subreadSet.name = "2017-04-14_54043_19k_4CWD_CornerZTweak_SilwetInSample2";
    rmd->subreadSet.tags = "subreadset";
    rmd->schemaVersion = "4.0.1";

    rmd->dataSetCollection = "<Collections />\n";


    PacBio::BAM::CollectionMetadata cmd;

    {
        bool computeStats = true;
        ResultWriter rw(&user, &cmd, rmd, &ppaAlgoConfig, fileHeader, apps, computeStats, fileHeader.MaxNumZMWs());

        std::vector<ComplexResult> crv;

        std::vector<ResultPacket> rpv;
        ResultPacket rp;
        rp.label = RegionLabelType::INSERT;
        rp.length = 101;
        rpv.push_back(std::move(rp));

        const auto& events = SimulateEventData();
        const auto& metrics = SimulateMetrics();
        ReadConfig config{};
        ZmwMetrics statsArg = RunMetrics(events, metrics, config.GenerateHQRegion(), config);

        ComplexResult cr(std::move(rpv), std::move(statsArg));
        crv.push_back(std::move(cr));

        rw.AddResultsToBuffer(0,std::move(crv));
    }

    {
        std::string datasetfile = user.outputPrefix + ".subreadset.xml";
        PBLOG_INFO << "will open " << datasetfile;
        // std::cerr << PacBio::Text::String::Slurp(datasetfile) << std::endl;

        {
            PacBio::BAM::DataSet ds(datasetfile);

            EXPECT_EQ("1",ds.Metadata().NumRecords());
            EXPECT_EQ("101",ds.Metadata().TotalLength());
            EXPECT_EQ("2017-04-14_54043_19k_4CWD_CornerZTweak_SilwetInSample2",ds.Name());
            EXPECT_EQ("",ds.ResourceId());
            EXPECT_EQ("subreadset",ds.Tags());
            EXPECT_EQ("9c428b0b-8d5d-439c-a0c4-16b6db7e3007",ds.UniqueId());
            EXPECT_EQ("4.0.1",ds.Version());
            EXPECT_EQ("2017-04-16T01:08:27Z",ds.CreatedAt());
            EXPECT_EQ("PacBio.DataSet.SubreadSet",ds.MetaType());

        }
        {
            std::ifstream is(datasetfile);
            PacBio::Text::PBXml ds(is);
            const std::string out = ds.Down("pbds:SubreadSet").Down("pbds:DataSetMetadata").Down("pbmeta:Collections").Print();
            EXPECT_TRUE(out.find("pbmeta:CollectionMetadata") != std::string::npos);
            EXPECT_EQ("<Collections />\n",rmd->dataSetCollection);
        }

        //auto s = PacBio::System::Run("xmllint --noout --schema PacBioDatasets.xsd " + datasetfile);
        //TEST_COUT << s;
    }


    {
        std::string datasetfile = user.outputPrefix + ".subreadset.xml";
        PBLOG_INFO << "will open " << datasetfile;
        // std::cerr << PacBio::Text::String::Slurp(datasetfile) << std::endl;

        {
            PacBio::BAM::DataSet ds(datasetfile);

            EXPECT_EQ("1",ds.Metadata().NumRecords());
            EXPECT_EQ("101",ds.Metadata().TotalLength());
            EXPECT_EQ("2017-04-14_54043_19k_4CWD_CornerZTweak_SilwetInSample2",ds.Name());
            EXPECT_EQ("",ds.ResourceId());
            EXPECT_EQ("subreadset",ds.Tags());
            EXPECT_EQ("9c428b0b-8d5d-439c-a0c4-16b6db7e3007",ds.UniqueId());
            EXPECT_EQ("4.0.1",ds.Version());
            EXPECT_EQ("2017-04-16T01:08:27Z",ds.CreatedAt());
            EXPECT_EQ("PacBio.DataSet.SubreadSet",ds.MetaType());
        }

        bool computeStats = false; // bam2bam usage
        user.subreadsetFilePath = SUBREADSET;
        PacBio::BAM::DataSet ds(datasetfile);
        ResultWriter rw(&user, &cmd, rmd, &ppaAlgoConfig, fileHeader, apps, computeStats, fileHeader.MaxNumZMWs());
        EXPECT_NO_THROW(rw.SetCollectionMetadataDataSet(&ds););
    }

    if (HasFailure()) tmpdir.Keep();
}

/// Tests 3 types of XML that contain WellSample and BioSample attributes, as they are
/// written to the BAM files with LB and SM tags.
TEST(ResultWriter,LB_SM_tags)
{
    PacBio::Logging::LogSeverityContext context(PacBio::Logging::LogLevel::WARN);
    PacBio::Dev::TemporaryDirectory tmpdir;
    tmpdir.Keep();

    UserParameters user;
    PpaAlgoConfig ppaAlgoConfig;

    FileHeaderBuilder fhb("LB_SM_movie",
                          80.0, // frame rate hz
                          80 * 60 * 60 * 3, // moie length frames
                          Readout::BASES,
                          MetricsVerbosity::MINIMAL,
                          generateExperimentMetadata(), //experimentMetadata,
                          "{}", //basecallerConfig,
                          {0}, // zmqNumbers
                          {} , //zmwUnitFeatures,
                          1024, //hfMetricFrames
                          4096, //mfMetricFrames,
                          16384); //sliceLengthFrames

    std::shared_ptr<RuntimeMetaData> rmd(new RuntimeMetaData);
    rmd->subreadSet.uniqueId = "9c428b0b-8d5d-439c-a0c4-16b6db7e3007";
    rmd->subreadSet.timeStampedName = "SubreadSetCollection_170416_01082744";
    rmd->subreadSet.createdAt = "2017-04-16T01:08:27Z";
    rmd->subreadSet.name = "2017-04-14_54043_19k_4CWD_CornerZTweak_SilwetInSample2";
    rmd->subreadSet.tags = "subreadset";
    rmd->schemaVersion = "4.0.1";
    std::string jsonHeader = fhb.CreateJSON();
    FileHeader fileHeader(jsonHeader.c_str(), jsonHeader.size());
    const std::vector<PacBio::BAM::ProgramInfo> apps;

    bool computeStats = true;

    PacBio::BAM::CollectionMetadata cmd;

    // create 3 datasets, each with different XML Wellsamples
    {
        // No WellSample node
        rmd->dataSetCollection = R"(
<Collections>
</Collections>
        )";
        user.outputPrefix = tmpdir.DirName() + "/LB_SM_tags_1_";
        ResultWriter rw1(&user, &cmd, rmd, &ppaAlgoConfig, fileHeader, apps, computeStats, fileHeader.MaxNumZMWs());
    }
    {
        std::string datasetfile = user.outputPrefix + ".subreadset.xml";
        PBLOG_INFO << "will open " << datasetfile ;
        PacBio::BAM::DataSet ds(datasetfile);
        for (const auto& bam : ds.BamFiles())
        {
            PBLOG_INFO << "bamfile: " << bam.Filename();
            const PacBio::BAM::BamHeader& header(bam.Header());
            for (const auto& rg : header.ReadGroups())
            {
                PBLOG_INFO << " looking at RG:" << rg.Id() << " " <<
                    rg.ToSam() << " platform:" << rg.Platform();
                if (rg.Platform() == "PACBIO")
                {
                    EXPECT_EQ("", rg.Library());
                    EXPECT_EQ("", rg.Sample());
                }
            }

            bool externalResourceCommentSeen = false;
            for(auto& comment : header.Comments())
            {
                if (PacBio::Text::String::Contains(comment,"LB_SM_tags_1_.sts.xml"))
                {
                    externalResourceCommentSeen = true;
                }
                PBLOG_INFO << " Comment: " << comment << std::endl;
            }
            EXPECT_TRUE(externalResourceCommentSeen);
        }
    }

    // The biosample names are purposely degraded with newlines, tab and ESC characters to make
    // sure that the conversion from XML to BAM handles them, as newlines and tabs are not
    // supports in BAM tags.  pugi_xml only supports escaped control characters. Literal white
    // space characters (i.e. tab and newline) are treated as spaces (&#20) on input, so they
    // are effectively ignored when parsed.
    // The second biosample will be ignored.
    {
        // Fully populated WellSample Node
        rmd->dataSetCollection = R"(
<Collections>
                <CollectionMetadata>
                  <WellSample Name="12345_SAT">
                    <BioSamples>
                      <BioSample Name="lambda&#10;11&#09;22&#x1B;33" />
                      <BioSample Name="lambda 2" />
                    </BioSamples>
                  </WellSample>
                </CollectionMetadata>
</Collections>
        )";
        user.outputPrefix = tmpdir.DirName() + "/LB_SM_tags_2_";
        ResultWriter rw2(&user, &cmd, rmd, &ppaAlgoConfig, fileHeader, apps, computeStats, fileHeader.MaxNumZMWs());
    }
    {
        std::string datasetfile = user.outputPrefix + ".subreadset.xml";
        PBLOG_INFO << "will open " << datasetfile;
        PacBio::BAM::DataSet ds(datasetfile);
        for (const auto& bam : ds.BamFiles())
        {
            PBLOG_INFO << "bamfile: " << bam.Filename();
            const PacBio::BAM::BamHeader& header(bam.Header());
            for (const auto& rg : header.ReadGroups())
            {
                PBLOG_INFO << " looking at RG:" << rg.Id() << " " <<
                        rg.ToSam() << " platform:" << rg.Platform();
                if (rg.Platform() == "PACBIO")
                {
                    EXPECT_EQ("12345_SAT", rg.Library());
                    // newline and tab is replaced by _
                    EXPECT_EQ("lambda_11_22_33", rg.Sample());
                }
            }
        }
    }


    {
        // Wellsample Node, without BioSamples nodes. The WellSample name has an escaped TAB character.
        rmd->dataSetCollection = R"(
<Collections>
                <CollectionMetadata>
                  <WellSample Name="12345&#9;SAT">
                  </WellSample>
                </CollectionMetadata>
</Collections>
        )";
        user.outputPrefix = tmpdir.DirName() + "/LB_SM_tags_3_";
        ResultWriter rw3(&user, &cmd, rmd, &ppaAlgoConfig, fileHeader, apps, computeStats, fileHeader.MaxNumZMWs());
    }

    {
        std::string datasetfile = user.outputPrefix + ".subreadset.xml";
        PBLOG_INFO << "will open " << datasetfile;
        PacBio::BAM::DataSet ds(datasetfile);
        for (const auto& bam : ds.BamFiles())
        {
            PBLOG_INFO << "bamfile: " << bam.Filename();
            const PacBio::BAM::BamHeader& header(bam.Header());
            for (const auto& rg : header.ReadGroups())
            {
                PBLOG_INFO << " looking at RG:" << rg.Id() << " " <<
                        rg.ToSam() << " platform:" << rg.Platform();
                if (rg.Platform() == "PACBIO")
                {
                    EXPECT_EQ("12345_SAT", rg.Library());
                    EXPECT_EQ("", rg.Sample());
                }
            }
        }
    }
}



TEST(ResultWriter,StreamingToStdout)
{
    // This tests that a proper BAM file is written to stdout, and a proper subreadset.xml file is created with
    // a sts.xml as an external resource file.
    // The subreadset.xml will mark the stdout external resource as "-" which is the convention for BamWriter.

    PacBio::Logging::LogSeverityContext context(PacBio::Logging::LogLevel::WARN);

    PacBio::Dev::TemporaryDirectory tmpdir;

    UserParameters user;
    PpaAlgoConfig ppaAlgoConfig;
    user.savePbi = false;
    FileHeaderBuilder fhb("FakeMovie",
                          80.0,
                          80.0 * 60 * 60 * 3,
                          Readout::BASES,
                          MetricsVerbosity::MINIMAL,
                          generateExperimentMetadata(),
                          "{}",
                          {0}, // just 1 ZMW in this virtual analysis
                          {},
                          1024,
                          4096,
                          16384,
                          FileHeaderBuilder::Flags()
                            .NewBazFormat(false));
    std::string header = fhb.CreateJSON();
    FileHeader fileHeader(header.c_str(), header.size());
    const std::vector<PacBio::BAM::ProgramInfo> apps;

    user.outputPrefix = tmpdir.DirName() + "/foo_";

    user.runtimeMetaDataFilePath = std::string(PacBio::PaPpaTestConfig::cmakeCurrentListDir) +
                                   "/data/ics_metadata.xml";
    std::shared_ptr<RuntimeMetaData> rmd = MetadataParser::ParseRMD(user.runtimeMetaDataFilePath);



    PacBio::BAM::CollectionMetadata cmd;

    {
        bool computeStats = true;

        UserParameters user1 = user;
        user1.outputPrefix = tmpdir.DirName() + "/bar.streamed";
        user1.savePbi = false;
        user1.inlinePbi = false;
        user1.streamBam = true; // check out the streaming option.
        user1.saveScraps = false;
        user1.nobam = false;

        std::string bamFileName = tmpdir.DirName() + "/test.bam";
        // TEST_COUT << "bamFileName: " << bamFileName << std::endl;
        {
            // this RAII object will capture all of stdout to bamFileName, and will release stdout
            // when it goes out of scope.
            TemporaryFileRedirection redirector(STDOUT_FILENO, bamFileName);
            ResultWriterEx rw(&user1, &cmd, rmd, &ppaAlgoConfig, fileHeader, apps, computeStats, fileHeader.MaxNumZMWs());
        }
        // TEST_COUT << " wrote BAM to " << bamFileName << std::endl;


        PacBio::BAM::BamReader bam(bamFileName);
        EXPECT_TRUE(bam.Header().HasProgram("bazwriter")) << bam.Header().ToSam();
        EXPECT_TRUE(bam.Header().HasProgram("bazFormat")) << bam.Header().ToSam();
        auto ids = bam.Header().ReadGroupIds();

//        bool gotPlatform = false;
//        for(const auto& rg : bam.Header().ReadGroups())
//        {
//            //TEST_COUT << "rg:" << rg.Id() <<" " << rg.Programs() << std::endl;
//
//            if (rg.Platform() == "PACBIO" && rg.PlatformModel() == PacBio::BAM::PlatformModelType::SEQUEL)
//            {
//                gotPlatform = true;
//            }
//        }
//        EXPECT_TRUE(gotPlatform) << bam.Header().ToSam();

        for(const auto& co : bam.Header().Comments())
        {
            // there is only supposed to be one comment in the BAM header
            //TEST_COUT << "co:" << co << std::endl;
            EXPECT_THAT(co,::testing::HasSubstr("bar.streamed.sts.xml")) << bam.Header().ToSam();
            if (co[0] =='{')
            {
                // found JSON header
                Json::Value sideband = PacBio::IPC::ParseJSON(co);
                ASSERT_EQ(sideband["name"].asString() ,"baz2bam_ccs_sideband") << sideband;
                EXPECT_EQ("faf4adf1-73f4-4ea8-be07-89a92318ff2c", sideband["dataset_uuid"].asString()) << sideband;
                EXPECT_GT(sideband["collection_metadata_xml"].asString().size() , 100)  << sideband;
                EXPECT_EQ(1, sideband["zmws"].asInt64());
                EXPECT_EQ("Sample Name-Cell1",sideband["subreadset_name"].asString());
                for(const auto& x : sideband["external_resources"])
                {
                    EXPECT_EQ("PacBio.SubreadFile.ChipStatsFile", x["meta_type"].asString());
                    EXPECT_THAT(x["file_name"].asString(), ::testing::EndsWith("/bar.streamed.sts.xml"));
                }
            }
        }

        // This is an example of what the SAM header might look like.
        //    @HD     VN:1.5  SO:unknown      pb:3.0.7
        //    @RG     ID:c611311a     PL:PACBIO       DS:READTYPE=SUBREAD;Ipd:CodecV1=ip;PulseWidth:CodecV1=pw;BASECALLERVERSION=5.0.0;FRAMERATEHZ=80.000000  PM:SEQUEL
        //    @PG     ID:bazFormat    PN:bazformat    VN:1.6.0
        //    @PG     ID:bazwriter    PN:bazwriter    VN:9.1.0
        //    @CO     {"external_resources":[{"file_name":"/tmp/f4ba_ceae_c185_93b6/bar.streamed.sts.xml","meta_type":"PacBio.SubreadFile.ChipStatsFile"}],"name":"baz2bam_css_sideband","version":1}

        const std::string datasetPath = user1.outputPrefix + ".subreadset.xml";

        PacBio::BAM::DataSet ds(datasetPath);

        bool haveStreamingBam = false;
        bool haveStsXml = false;
        try
        {
            for (const auto& file : ds.AllFiles())
            {
                // TEST_COUT << " " << datasetPath << ":" << file << std::endl;

                if (file == tmpdir.DirName() + "/-") haveStreamingBam = true;
                if (file == tmpdir.DirName() + "/bar.streamed.sts.xml") haveStsXml = true;
            }
        }
        catch(const std::exception& ex)
        {
            TEST_COUT << "EXCEPTION: " << ex.what() << std::endl;
            PBLOG_ERROR << ex.what();
        }
        EXPECT_TRUE(haveStreamingBam);
        EXPECT_TRUE(haveStsXml);
    }
    if (HasFailure())
    {
        TEST_COUT << "Temporary files kept in " << tmpdir.DirName() << " because of test failure." << std::endl;
        tmpdir.Keep();
    }
}
