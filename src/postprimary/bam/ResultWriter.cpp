// Copyright (c) 2017-2020, Pacific Biosciences of California, Inc.
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
/// \brief  Class that controls writing of BAM files
///

#include <boost/filesystem/operations.hpp>

#include <pacbio/text/PBXml.h>
#include <pacbio/text/pugixml_pb.hpp>
#include <pacbio/ipc/JSON.h>

#include <bazio/PacketFieldMap.h>

#include <postprimary/stats/ChipStatsWriter.h>

#include "ResultWriter.h"
#include "BamProducer.h"
#include "RuntimeMetaData.h"
#include "BamCommentSideband.h"

namespace {

/// C++ equivalent of clib basename. Strips off directories from file path name.
std::string JustBasename(const std::string& fullPath)
{
    boost::filesystem::path path=fullPath;
    return path.filename().string();
}


std::string Encode(const std::string& data)
{
    std::string buffer;
    buffer.reserve(data.size());
    for (size_t pos = 0; pos != data.size(); ++pos)
    {
        switch (data[pos])
        {
            case '&':
                buffer.append("&amp;");
                break;
            case '\"':
                buffer.append("&quot;");
                break;
            case '\'':
                buffer.append("&apos;");
                break;
            case '<':
                buffer.append("&lt;");
                break;
            case '>':
                buffer.append("&gt;");
                break;
            default:
                buffer.append(&data[pos], 1);
                break;
        }
    }
    return buffer;
}

/// Converts a strings into a string that is valid for BAM tags.
/// BAM tags only support ASCII 0x20 to 0x7F.
/// This is used to validate "user" supplied input.
std::string SanitizeBAMTag(const std::string& inTag)
{
    std::string outTag;
    PBLOG_DEBUG << "Sanitize called with:\"" << inTag << "\"";
    for (auto x : inTag)
    {
        char outChar;
        if ((int) x >= 0x20 && (int) x <= 0x7F)
        {
            // legal BAM string character
            outChar = x;
        }
        else
        {
            outChar = '_';
        }
        PBLOG_TRACE << " " << x << " 0x" << std::hex << (int) x << std::dec << " -> "
                    << outChar << " 0x" << std::hex << (int) outChar << std::dec;
        outTag += outChar;
    }
    return outTag;
}

void AddXMLNamespaces(PacBio::BAM::DataSet& ds)
{
    ds.Attribute("xmlns"             , "http://pacificbiosciences.com/PacBioDatasets.xsd")
      .Attribute("xmlns:xsi"         , "http://www.w3.org/2001/XMLSchema-instance")
      .Attribute("xmlns:pbbase"      , "http://pacificbiosciences.com/PacBioBaseDataModel.xsd")
      .Attribute("xmlns:pbds"        , "http://pacificbiosciences.com/PacBioDatasets.xsd")
      .Attribute("xmlns:pbdm"        , "http://pacificbiosciences.com/PacBioDataModel.xsd")
      .Attribute("xmlns:pbmeta"      , "http://pacificbiosciences.com/PacBioCollectionMetadata.xsd")
      .Attribute("xmlns:pbpn"        , "http://pacificbiosciences.com/PacBioPartNumbers.xsd")
      .Attribute("xmlns:pbrk"        , "http://pacificbiosciences.com/PacBioReagentKit.xsd")
      .Attribute("xmlns:pbsample"    , "http://pacificbiosciences.com/PacBioSampleInfo.xsd")
      .Attribute("xsi:schemaLocation", "http://pacificbiosciences.com/PacBioSecondaryDataModel.xsd");
}

}

namespace PacBio {
namespace Primary {
namespace Postprimary {

using namespace PacBio;
using namespace PacBio::BAM;
using namespace PacBio::Primary;


ResultWriter::ResultWriter(const UserParameters* user,
                           const PacBio::BAM::CollectionMetadata* cmd,
                           std::shared_ptr<RuntimeMetaData>& rmd,
                           const PpaAlgoConfig* ppaAlgoConfig,
                           const std::string& movieName,
                           float movieTimeInHrs,
                           const std::string& bazVersion,
                           const std::string& bazWriterVersion,
                           const std::string& basecallerVersion,
                           double frameRateHz,
                           bool internal,
                           const std::vector<ProgramInfo>& apps,
                           bool computeStats,
                           uint32_t maxNumZwms)
    : abortNow_(false),
    user_(user),
    cmd_(cmd),
    rmd_(rmd),
    ppaAlgoConfig_(ppaAlgoConfig),
    movieName_(movieName),
    movieTimeInHrs_(movieTimeInHrs),
    bazVersion_(bazVersion),
    bazWriterVersion_(bazWriterVersion),
    basecallerVersion_(basecallerVersion),
    frameRateHz_(frameRateHz),
    internal_(internal),
    computeStats_(computeStats),
    fastaStream_(nullptr),
    fastqStream_(nullptr),
    chipStats_(maxNumZwms),
    subreadSet_(BAM::DataSet::TypeEnum::SUBREAD),
    maxNumZmws_(maxNumZwms)
{
    if (user == nullptr)
    {
        throw PBException("UserParameters parameters was not allocated (was null)");
    }
    if (!rmd)
    {
        throw PBException("RuntimeMetaData parameters was not allocated (was null)");
    }
    if (!ppaAlgoConfig)
    {
        throw PBException("PpaAlgoConfig was not allocated (was null)");
    }

    CreateDataSet();

    if (computeStats_)
    {
        stsXmlFilename_ = user_->outputPrefix + ".sts.xml";
    }


    if (!user_->nobam)
    {
        CreateBams(CreateProgramInfos(apps));
        PopulateScrapTags();
    }

    // Name fasta/q files
    fastqFileName_ = user_->outputPrefix + ".fastq.gz";
    fastaFileName_ = user_->outputPrefix + ".fasta.gz";

    fastqFileTmpName_ = fastqFileName_ + ".tmp";
    fastaFileTmpName_ = fastaFileName_ + ".tmp";

    if (user_->outputFasta) CreateFastaFiles();
    if (user_->outputFastq) CreateFastqFiles();

    // Controls the termination of the writeThread
    writeThreadContinue_ = true;
    // Tells the writeThread to finish everything in queue, now.
    finishSaving_ = false;
    // Spawn new the writeThread
    writeThread_ = std::thread([this]() { WriteToDiskLoop(); });
    cleanThread_ = std::thread([this]() { CleanLoop(); });
}

/// Closes output streams.
ResultWriter::~ResultWriter()
{
    CloseAndValidate();
}

Validation ResultWriter::CloseAndValidate()
{
    if (validationStatus_ != Validation::OPEN && !abortNow_)
        return validationStatus_;

    if (!writeThread_.joinable() || !cleanThread_.joinable())
    {
        return Validation::CLOSED_TRUNCATED;
    }

    // Forces the writeThread to finish up
    finishSaving_ = true;
    // Wait until the buffer is empty
    while (!resultMap_.Empty() && !abortNow_)
        std::this_thread::sleep_for(std::chrono::seconds(1));
    // Terminates writeThread_
    writeThreadContinue_ = false;
    writeThread_.join();
    cleanThread_.join();

    if (user_->lookForLoop || user_->trimToLoop)
    {
        PBLOG_INFO << numHasAdapters_
                   << " ZMWs found to have adapters, "
                   << numHasStem_
                   << " of which contained stem sequences ("
                   << 100.0 * static_cast<float>(numHasStem_)/numHasAdapters_
                   << "%)";
        PBLOG_INFO << numHasAdaptersGEQ3_
                   << " ZMWs found to have >= 3 adapters, "
                   << numHasStemGEQ3_
                   << " of which contained stem sequences ("
                   << 100.0 * static_cast<float>(numHasStemGEQ3_)/numHasAdaptersGEQ3_
                   << "%)";
    }


    auto closeAndRenameFastx = [this](gzFile& stream,
                                      const std::string& tmpName,
                                      const std::string& finalName) {
        gzclose(stream);
        if (!abortNow_)
        {
            PBLOG_INFO << "Renaming file " << tmpName << " -> " << finalName;
            std::rename(tmpName.c_str(), finalName.c_str());
        }
    };

    // Close all gz streams
    if (fastaStream_)
        closeAndRenameFastx(fastaStream_, fastaFileTmpName_, fastaFileName_);
    if (fastqStream_)
        closeAndRenameFastx(fastqStream_, fastqFileTmpName_, fastqFileName_);

    if (!abortNow_)
    {
        // Regular behavior
        if (computeStats_)
        {
            ChipStatsWriter csw(stsXmlFilename_, !ppaAlgoConfig_->controlFilter.disableControlFiltering);
            csw.WriteChip(chipStats_, movieName_, movieTimeInHrs_, rmd_->schemaVersion);
            WriteDataset();
        }
        else // bam2bam
        {
            std::ofstream datasetOut(user_->outputPrefix + ".subreadset.xml");
            PacBio::BAM::DataSetMetadata dataSetMetadata(std::to_string(numRecords_),
                                                         std::to_string(totalLength_));

            subreadSet_.Name(Encode(ds_->Name()))
                    .Tags("subreadset")
                    .Version(ds_->Version())
                    .CreatedAt(ds_->CreatedAt())
                    .TimeStampedName(ds_->TimeStampedName())
                    .UniqueId(ds_->UniqueId());

            AddXMLNamespaces(subreadSet_);
            subreadSet_.Metadata(dataSetMetadata).Metadata().CollectionMetadata(ds_->Metadata().CollectionMetadata());
            subreadSet_.SaveToStream(datasetOut, DataSetPathMode::ALLOW_RELATIVE);
        }
    }

    bool valid = true;
    if (bamPrimaryStream_)
        valid &= bamPrimaryStream_->Close(abortNow_);
    if (bamScrapsStream_)
        valid &= bamScrapsStream_->Close(abortNow_);

    if (valid)
        validationStatus_ = Validation::CLOSED_VALIDATED;
    else
        validationStatus_ = Validation::CLOSED_TRUNCATED;


    return validationStatus_;
}


/// Writes content of buffer to disk.
void ResultWriter::WriteToDiskLoop()
{

    // We need to be able to terminate this thread
    while (writeThreadContinue_)
    {
        if (finishSaving_ && resultMap_.Empty()) break;

        if (!resultMap_.HasKey(batchIdCounter_))
        {
            PBLOG_INFO << "[WriteToDiskLoop]: Waiting for batchIdCounter = " << batchIdCounter_;
            while (!resultMap_.HasKey(batchIdCounter_))
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                if (finishSaving_ && resultMap_.Empty()) break;
            }
            continue;
        }
        // Transfer ownership
        auto& resultBufferTmp = resultMap_.Ref(batchIdCounter_);

        // Save subreads to BAM
        if (!user_->nobam)
        {
            // auto startTime = std::chrono::high_resolution_clock::now();
            for (auto& cplx : resultBufferTmp)
            {
                // We want to skip if we don't know the RegionLabelType
                bool skip = false;
                for (const auto& result : cplx.resultPackets)
                {
                    switch (result.label)
                    {
                    case RegionLabelType::INSERT:
                    case RegionLabelType::POLYMERASEREAD:
                    case RegionLabelType::ADAPTER:
                    case RegionLabelType::BARCODE:
                    case RegionLabelType::LQREGION:
                    case RegionLabelType::FILTERED:
                        break;
                    default:
                        skip = true;
                        break;
                    }
                }
                if (skip)
                {
                    PBLOG_ERROR << "Skipping ZMW number " << cplx.resultPackets[0].zmwNum
                                << " due to unknown RegionLabelType";
                    continue;
                }

                // We do it this way because imputed adapters may not go through
                // stem detection, there-by changing the ZMW classification
                bool hasStem = false;
                bool hasAdapters = false;
                bool isFiltered = false;
                int numAdapters = 0;
                // Save records
                for (auto& result : cplx.resultPackets)
                {
                    if (abortNow_) goto abort;

                    switch (result.label)
                    {
                    case RegionLabelType::INSERT:
                    case RegionLabelType::POLYMERASEREAD:
                        if (!result.control)
                        {
                            ++numRecords_;
                            totalLength_ += result.length;
                            WriteSHPRecord(&result);
                        }
                        else
                        {
                            WriteScrapRecord(scrapsFiltered_, &result);
                        }
                        if (result.context.hasStemBefore ||
                                result.context.hasStemAfter)
                        {
                            hasStem = true;
                        }
                        break;
                    case RegionLabelType::ADAPTER:
                        WriteScrapRecord(scrapsAdapter_, &result);
                        hasAdapters = true;
                        ++numAdapters;
                        break;
                    case RegionLabelType::BARCODE:
                        WriteScrapRecord(scrapsBarcode_, &result);
                        break;
                    case RegionLabelType::LQREGION:
                        WriteScrapRecord(scrapsLQRegion_, &result);
                        break;
                    case RegionLabelType::FILTERED:
                        WriteScrapRecord(scrapsFiltered_, &result);
                        isFiltered = true;
                        break;
                    default:
                        throw PBException(std::string("[ResultWriter] Unknown type ") + static_cast<char>(result.label));
                    }
                }
                if (hasStem && hasAdapters && !isFiltered)
                {
                    ++numHasStem_;
                    if (numAdapters >= 3)
                        ++numHasStemGEQ3_;
                }
                if (hasAdapters && !isFiltered)
                {
                   ++numHasAdapters_;
                    if (numAdapters >= 3)
                        ++numHasAdaptersGEQ3_;
                }
                if (computeStats_)
                    chipStats_.AddData(cplx.zmwMetrics);

            }
        }

        // Save fasta.gz
        if (user_->outputFasta)
        {
            for (const auto& cplx : resultBufferTmp)
            {
                for (const auto& result : cplx.resultPackets)
                {
                    if (abortNow_) goto abort;
                    if (!result.control &&
                        (result.label == RegionLabelType::INSERT
                         || result.label == RegionLabelType::POLYMERASEREAD))
                        WriteFasta(result, &fastaStream_);
                }
            }
        }

        // Save fastq.gz
        if (user_->outputFastq)
        {
            for (const auto& cplx : resultBufferTmp)
            {
                for (const auto& result : cplx.resultPackets)
                {
                    if (abortNow_) goto abort;
                    if (!result.control &&
                        (result.label == RegionLabelType::INSERT
                         || result.label == RegionLabelType::POLYMERASEREAD))
                        WriteFastq(result, &fastqStream_);
                }
            }
        }

        RemoveBatchBytes(batchIdCounter_);

        // Inc to get next batch on the next round
        ++batchIdCounter_;
    }
    abort:;
}

void ResultWriter::WriteFasta(const ResultPacket& result, gzFile* outputStream)
{
    int gzerr;
    if (gzwrite(*outputStream, ">", 1) <= 0
        || gzwrite(*outputStream, result.name.c_str(), result.name.size()) <= 0
        || gzwrite(*outputStream, "\n", 1) <= 0
        || gzwrite(*outputStream, result.bases.c_str(), result.length) <= 0
        || gzwrite(*outputStream, "\n", 1) <= 0)
    {
        PBLOG_ERROR << "[ResultWriter] Error saving a fasta.gz: " << gzerror(*outputStream, &gzerr);
        throw PBException("fasta.gz output error");
    }
}

void ResultWriter::WriteFastq(const ResultPacket& result, gzFile* outputStream)
{
    int gzerr;
    if (gzwrite(*outputStream, preFastq_.c_str(), 1) <= 0
        || gzwrite(*outputStream, result.name.c_str(), result.name.size()) <= 0
        || gzwrite(*outputStream, "\n", 1) <= 0
        || gzwrite(*outputStream, result.bases.c_str(), result.length) <= 0
        || gzwrite(*outputStream, "\n+\n", 3) <= 0
        || gzwrite(*outputStream, result.overallQv.c_str(), result.length) <= 0
        || gzwrite(*outputStream, "\n", 1) <= 0)
    {
        PBLOG_ERROR << "[ResultWriter] Error saving a fastq.gz: " << gzerror(*outputStream, &gzerr);
        throw PBException("fastq.gz error");
    }
}

void ResultWriter::WriteRecord(const ResultPacket& result,
                        std::unique_ptr<BamProducer>& bamStream)
{
    try
    {
        bamStream->Write(result.bamRecord);
    }
    catch (std::exception&)
    {
        PBLOG_ERROR << "[ResultWriter] Error saving to BAM " << bamStream->BamPath();
        throw;
    }
}


/// Cleans after the WriteToDiskLoop
void ResultWriter::CleanLoop()
{
    uint32_t current = 0;
    // We need to be able to terminate this thread
    while (writeThreadContinue_)
    {
        if (finishSaving_ && resultMap_.Empty()) break;
        if (current == batchIdCounter_)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        while (current < batchIdCounter_)
        {
            resultMap_.Pop(current);
            ++current;
        }
    }
}

void ResultWriter::CreateDataSet()
{
    dataset_ = DataSet(DataSet::SUBREAD);  // this should automatically set the MetaType as "PacBio.DataSet.SubreadSet"
    if (rmd_->dataSetCollection.empty())
    {
        dataset_.Name("DataSet_PPA")
                .Tags("PPA")
                .Version("2.3.0")
                ;
    }
    else
    {
        // carry over values from original RMD
        dataset_.Name           (Encode(rmd_->subreadSet.name))
                .Tags           ("subreadset")
                .Version        (rmd_->schemaVersion)
                .CreatedAt      (rmd_->subreadSet.createdAt)
                .TimeStampedName(rmd_->subreadSet.timeStampedName)
                .UniqueId       (rmd_->subreadSet.uniqueId)
                ;
    }

    AddXMLNamespaces(dataset_);
}


void ResultWriter::WriteDataset()
{
    std::string subreadsetPath = user_->outputPrefix + ".subreadset.xml";
    std::string subreadsetPathTmp = subreadsetPath + ".tmp";
    try
    {
        PacBio::BAM::DataSetMetadata dataSetMetadata(std::to_string(numRecords_),   // casting to string makes me itch.
                                                     std::to_string(totalLength_)); // fixme. PBBAM may update the API to be type safe.
        dataset_.Metadata(dataSetMetadata);

        std::ofstream datasetStream(subreadsetPathTmp);

        if (!rmd_->dataSetCollection.empty())
        {
            if (cmd_ != nullptr)
                dataset_.Metadata().CollectionMetadata(*cmd_);
            else
                throw PBException("Unexpected null collection metadata");
            dataset_.SaveToStream(datasetStream, DataSetPathMode::ALLOW_RELATIVE);
        }
        else
        {
            // Save directly from PBBAM Dataset to external file.
            dataset_.SaveToStream(datasetStream, DataSetPathMode::ALLOW_RELATIVE);
        }

    }
    catch (const std::exception& )
    {
        PBLOG_ERROR << "[ResultWriter] Failed writing to " << subreadsetPathTmp;
        throw;
    }

    if (std::rename(subreadsetPathTmp.c_str(), subreadsetPath.c_str()))
    {
        PBLOG_ERROR << "[ResultWriter] Could not rename " << subreadsetPathTmp << " to " << subreadsetPath;
    }
}

gzFile ResultWriter::CreateGZFile(const char* const cString) const
{
    const gzFile fileHandle = gzopen(cString, "wb1");
    if (!fileHandle)
    {
        PBLOG_ERROR << "[ResultWriter] Error opening output gzip file " << cString;
        throw PBException("Error opening output gzip file");
    }
    PBLOG_INFO << "Opening file " << cString;
    return fileHandle;
}

void ResultWriter::CreateFastqFiles()
{
    fastqStream_ = CreateGZFile(fastqFileTmpName_.c_str());
    AddResourceFastx("fastq", fastqFileName_);
}

void ResultWriter::CreateFastaFiles()
{
    fastaStream_ = CreateGZFile(fastaFileTmpName_.c_str());
    AddResourceFastx("fasta", fastaFileName_);
}

std::string ResultWriter::CreateBam(const std::vector<ProgramInfo>& programInfos,
                              const std::string& localRgType,
                              const std::string& localBamInfix,
                              const std::vector<BamCommentSideband::ExternalResource>& additionalExternalResources,
                              ReadGroupInfo* localRG,
                              std::unique_ptr<BamProducer>* localBam)
{
    BamHeader localHeader;
    localHeader.Version(bamVersion_)
            .SortOrder(sortOrder_)
            .Programs(programInfos);

    BamCommentSideband sideband;
    {
        if (user_ && user_->runtimeMetaDataFilePath != "")
        {
            PBLOG_INFO << "Writing contents of " << user_->runtimeMetaDataFilePath << " <CollectionMetadata>"
                                                                                      " to BAM header comment";
            std::ifstream ifs(user_->runtimeMetaDataFilePath);
            PacBio::Text::PBXml rmd(ifs);
            auto subreadSetElement = rmd.Down("PacBioDataModel")
                    .Down("ExperimentContainer")
                    .Down("Runs")
                    .Down("Run")
                    .Down("Outputs")
                    .Down("SubreadSets")
                    .Down("SubreadSet");
            // just grab the Name attribute
            sideband.SetSubreadsetName(subreadSetElement.currentNode.attribute("Name").as_string(""));

            // grab the entire XML string
            PacBio::Text::PBXml dataSetMetadata = subreadSetElement
                    .Down("DataSetMetadata");
            std::string xml = dataSetMetadata.RawChildXML("Collections");
            PBLOG_DEBUG << "collection metadata" << xml;
            sideband.SetCollectionMetadataXML(xml);

            // just grab the UUID
            std::string consensusReadSetRefUuid = dataSetMetadata
                    .Down("Collections")
                    .Down("CollectionMetadata")
                    .Down("ConsensusReadSetRef").currentNode.attribute("UniqueId").as_string("");
            if (consensusReadSetRefUuid != "")
            {
                sideband.SetDatasetUUID(consensusReadSetRefUuid);
            }
        }
    }
    if (maxNumZmws_ != 0) sideband.SetNumZmws(maxNumZmws_);
    for(const auto& resource: additionalExternalResources)
    {
        sideband.AddExternalResource(resource);
    }
    {
        Json::Value commentCopy = sideband.GetJsonComment();
        if (! PacBio::Logging::PBLogger::SeverityIsEnabled(PacBio::Logging::LogLevel::DEBUG))
        {
            commentCopy["collection_metadata_xml"] = "<suppressed>";
        }
        PBLOG_INFO << "Created BAM Header comment:" << commentCopy;
    }

    std::string comment = PacBio::IPC::RenderJSON(sideband.GetJsonComment());
    localHeader.AddComment(comment);

    *localRG = CreateReadGroupInfo(localRgType);
    localHeader.AddReadGroup(*localRG);

    localBam->reset(
            new BamProducer(user_->outputPrefix, localBamInfix, localHeader,
                            user_->savePbi, user_->inlinePbi,
                            user_->bamthreads, user_->threads,
                            user_->streamBam));

    std::string bamFile = (*localBam)->BamPath();

    return bamFile;
}

/// Creates a primary and possible secondary bam file. For example,
/// the primary BAM file might be the subreads.bam file and the secondary file might be the
/// scraps.bam file.  It also has the sideeffect of creating the sts.xml file and an adapters file too.
void ResultWriter::CreateBams(const std::vector<ProgramInfo>& programInfos)
{
    BamConfig cc;
    if (user_->polymeraseread)
        cc = polyConfig;
    else if (user_->hqonly)
        cc = hqConfig;
    else
        cc = subreadConfig;

    /// external resource files that are generated by baz2bam are sent through to CCS by means of writing
    /// information to a BamCOmmentSideband object that is transmitted through the BAM header.
    /// CCS can add these files as external resources of the final consensusreads.xml dataset.
    std::vector<BamCommentSideband::ExternalResource> additionalExternalResources;

    if (stsXmlFilename_ != "") {
        BamCommentSideband::ExternalResource ex;
        ex.file_name = stsXmlFilename_;
        ex.meta_type = chipStatsMetaType;
        additionalExternalResources.push_back(ex);
    }

    auto primaryBamFile = CreateBam(
            programInfos, // input
            cc.primaryRgType, // input
            cc.primaryBamInfix, // input
            additionalExternalResources, // inputs
            &primaryRG_, // output
            &bamPrimaryStream_); // outputs

    auto primaryResource = CreateResourceBam(primaryBamFile,  cc.primaryBamInfix, cc.primaryMetaType);

    if (user_->saveScraps && primaryBamFile != "-")
    {
        const std::vector<BamCommentSideband::ExternalResource> scrapsExternalResources;
        auto secondaryBamFile = CreateBam(programInfos,
                "SCRAP",
                "scraps",
                scrapsExternalResources,
                &scrapsRG_,
                &bamScrapsStream_);
        auto secondaryResource = CreateResourceBam(secondaryBamFile, "scraps", cc.secondaryMetaType);
        primaryResource.ExternalResources().Add(secondaryResource);
    }

    CreateAndAddResourceAdapterFile(&primaryResource);
    subreadSet_.ExternalResources().Add(primaryResource);

    CreateAndAddResourceStsXml(&primaryResource);
    dataset_.ExternalResources().Add(primaryResource);
}

/// Creates a resource object for a BAM file that can be added to a PBBAM Dataset object. If appropriate, it will also
/// create a child index resource object for the PBI index file.
/// \param bamFileName - a full path to the local file system or "-" meaning the BAM file is streamed to stdout.
/// \returns the object that will be added to the dataset object.
ExternalResource ResultWriter::CreateResourceBam(const std::string& bamFileName,
        const std::string& convention,
        const std::string& metatype)
{
    std::stringstream desc;
    desc << "Points to the "
            << convention
            << " bam file.";
    ExternalResource resource(metatype, JustBasename(bamFileName));

    resource.Name(convention + " bam")
            .Description(desc.str()).Version(rmd_->schemaVersion);

    if(bamFileName != "" && bamFileName != "-" && user_->savePbi)
    {
        const std::string pbiFilename = JustBasename(bamFileName) + ".pbi";
        FileIndex pbi("PacBio.Index.PacBioIndex", pbiFilename);
        pbi.Version(rmd_->schemaVersion);
        resource.FileIndices().Add(pbi);
    }
    return resource;
}

void ResultWriter::CreateAndAddResourceStsXml(ExternalResource* parent)
{
    if (computeStats_)
    {
        ExternalResource resource(chipStatsMetaType, JustBasename(stsXmlFilename_));
        resource.Name("Chipstats XML")
                .Description("Points to the summary sts.xml file.").Version(rmd_->schemaVersion);
        parent->ExternalResources().Add(resource);
    }
}

std::string ResultWriter::RelativizePath(const std::string& originalFilePath)
{
    const auto pos = originalFilePath.rfind("/");
    if (pos != std::string::npos
        && pos != originalFilePath.size())
    {
        return originalFilePath.substr(pos + 1);
    }
    return originalFilePath;
}

void ResultWriter::CreateAndAddResourceAdapterFile(ExternalResource* parent)
{
    if (!user_->adaptersFilePath.empty())
    {
        const auto relAdapterPath = RelativizePath(user_->adaptersFilePath);
        ExternalResource resource(adapterFileMetaType, relAdapterPath);
        resource.Name("Adapters FASTA")
                .Description("Points to the adapters.fasta file.")
                .Version(rmd_->schemaVersion);
        parent->ExternalResources().Add(resource);
    }
}

void ResultWriter::AddResourceFastx(const std::string& convention, const std::string& filename)
{
    std::stringstream desc;
    desc << "Points to the "
            << convention
            << " gz file.";
    ExternalResource resource("SubreadFile.SubreadFastxFile", JustBasename(filename));
    resource.Name(convention + " gz")
            .Description(desc.str()).Version(rmd_->schemaVersion);

    dataset_.ExternalResources().Add(resource);
};

std::vector<ProgramInfo> ResultWriter::CreateProgramInfos(const std::vector<ProgramInfo>& apps)
{
    std::vector<ProgramInfo> pgs;

    // BAZ version
    if (bazVersion_.compare("0.0.0") != 0)
    {
        ProgramInfo bazPg;
        bazPg.Name("bazformat")
                .Version(bazVersion_)
                .Id("bazFormat");
        pgs.emplace_back(std::move(bazPg));
    }

    // BazWriter version
    if (!bazWriterVersion_.empty())
    {
        ProgramInfo bazWriterPg;
        bazWriterPg.Name("bazwriter")
                .Version(bazWriterVersion_)
                .Id("bazwriter");
        pgs.emplace_back(std::move(bazWriterPg));
    }

    // Apps
    for (const auto& a : apps)
        pgs.emplace_back(std::move(a));

    return pgs;
}

ReadGroupInfo ResultWriter::CreateReadGroupInfo(const std::string& readType)
{
    static bool warnOnce = [](){PBLOG_WARN << "Hardcoding platform to SequelII for ReadGroupInfo in BAM file"; return true;}();
    (void)warnOnce;

    PlatformModelType type = PlatformModelType::SEQUELII;
    ReadGroupInfo group(rmd_->movieName, readType, type);

    group.BasecallerVersion(basecallerVersion_)
            .FrameRateHz(std::to_string(frameRateHz_));

    if (!rmd_->sequencingKit.empty())
        group.SequencingKit(SanitizeBAMTag(rmd_->sequencingKit));
    if (!rmd_->bindingKit.empty())
        group.BindingKit(SanitizeBAMTag(rmd_->bindingKit));

    if (!rmd_->dataSetCollection.empty())
    {
        pugi_pb::xml_document collections;
        if (!collections.load(rmd_->dataSetCollection.c_str()))
        {
            throw PBException("failure to parse datasetcollection");
        }

        auto wellSampleNode = collections.select_single_node(".//Collections/CollectionMetadata/WellSample/@Name");
        if (wellSampleNode)
        {
            std::string library = wellSampleNode.attribute().as_string("");
            library = SanitizeBAMTag(library);
            group.Library(library); // aka LB tag
        }

        auto bioSampleNode = collections.select_single_node(".//Collections/CollectionMetadata/WellSample/BioSamples/BioSample/@Name");
        if (bioSampleNode)
        {
            std::string sample = bioSampleNode.attribute().as_string("");
            sample = SanitizeBAMTag(sample);
            group.Sample(sample); // aka SM tag
        }
    }

    // Production tags
    group.BaseFeatureTag(BaseFeature::START_FRAME, "sf");
    group.IpdCodec(FrameCodec::RAW);
    group.PulseWidthCodec(FrameCodec::RAW);

    // Internal tags
    if(internal_)
    {
        group.BaseFeatureTag(BaseFeature::PULSE_CALL, "pc");
        group.BaseFeatureTag(BaseFeature::PRE_PULSE_FRAMES, "pd");
        group.BaseFeatureTag(BaseFeature::PULSE_CALL_WIDTH, "px");
        group.BaseFeatureTag(BaseFeature::PKMEAN, "pa");
        group.BaseFeatureTag(BaseFeature::PKMID, "pm");
        group.BaseFeatureTag(BaseFeature::PULSE_EXCLUSION, "pe");
    }

    return group;
}

void ResultWriter::PopulateScrapTags()
{
    scrapsAdapter_ = Tag(static_cast<char>(RegionLabelType::ADAPTER));
    scrapsBarcode_ = Tag(static_cast<char>(RegionLabelType::BARCODE));
    scrapsLQRegion_ = Tag(static_cast<char>(RegionLabelType::LQREGION));
    scrapsFiltered_ = Tag(static_cast<char>(RegionLabelType::FILTERED));
    scrapsAdapter_.Modifier(TagModifier::ASCII_CHAR);
    scrapsBarcode_.Modifier(TagModifier::ASCII_CHAR);
    scrapsLQRegion_.Modifier(TagModifier::ASCII_CHAR);
    scrapsFiltered_.Modifier(TagModifier::ASCII_CHAR);

    scrapsZmwNormal_ = Tag(static_cast<char>(ZmwType::NORMAL));
    scrapsZmwSentinel_ = Tag(static_cast<char>(ZmwType::SENTINEL));
    scrapsZmwControl_ = Tag(static_cast<char>(ZmwType::CONTROL));
    scrapsZmwMalformed_ = Tag(static_cast<char>(ZmwType::MALFORMED));
    scrapsZmwNormal_.Modifier(TagModifier::ASCII_CHAR);
    scrapsZmwSentinel_.Modifier(TagModifier::ASCII_CHAR);
    scrapsZmwControl_.Modifier(TagModifier::ASCII_CHAR);
    scrapsZmwMalformed_.Modifier(TagModifier::ASCII_CHAR);
}


}}}
