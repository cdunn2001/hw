
#include <memory>
#include <string>

#include <boost/numeric/conversion/cast.hpp>
#include <boost/filesystem.hpp>
#include <iostream>
#include <gtest/gtest.h>

#include <pacbio-primary-test-config.h>		// Defines sequelBaseDir.

#include "HDFMultiArrayIO.h"
#include "testTraceFilePath.h"

namespace PacBio {
namespace Primary {
namespace Basecaller {

using std::string;
using std::unique_ptr;

using boost::numeric_cast;
using boost::filesystem::path;


TEST (TestHDFMultiArrayIO, ConstructorException)
{
	bool exceptionCaught = false;
    const string fileName = "this.cannot.possibly.exist.no_way_jose";

    // Silence the the output that HDF5 generates by default when it throws an
    // exception.
    H5::Exception::dontPrint();

	try
	{
		HDFMultiArrayIO::CreateReader(fileName);
	}
	catch (H5::Exception& ex)
	{
		exceptionCaught = true;
		string detail = ex.getDetailMsg();

		// The exception must include the name of the filename.
		// See Bug 22116.
		EXPECT_NE (string::npos, detail.find(fileName))
			<< "Did not find the file name in the exception detail.";
	}

	EXPECT_TRUE (exceptionCaught);
}


struct TestHDFMultiArrayIORead : public ::testing::Test
{
    path				mHdfInputFilePath;
    unique_ptr<const HDFMultiArrayIO> mHdfInputFile;
	const int			mNumDims;
	std::vector<int>	mShape;
	std::vector<int>	mAttrShape;

	TestHDFMultiArrayIORead()
        : mNumDims		(3)
		, mShape		(mNumDims)
        , mAttrShape	(1u)
	{
#ifdef PB_MIC_COPROCESSOR
		path sequelDir = "/home/pbi";
        mHdfInputFilePath = sequelDir ;
#else
		path sequelDir = PacBio::Primary::CommonTest::sequelBaseDir;
        mHdfInputFilePath = sequelDir / "common" / "pacbio" / "primary" / "test";
#endif
        mHdfInputFilePath /= "UnitTest.h5";
        if (!boost::filesystem::exists(mHdfInputFilePath))
        {
            auto fp = TraceFilePath("/dept/primary/unitTestInput/basecaller/common/test/UnitTest.h5");
            mHdfInputFilePath = boost::filesystem::path(fp);
        }

		mShape[0] = 2;
		mShape[1] = 3;
		mShape[2] = 5;

		mAttrShape[0] = 2;
	}

	~TestHDFMultiArrayIORead ()
	{
	}

	void SetUp(void)
	{
        ASSERT_TRUE (is_regular_file(mHdfInputFilePath))
            << mHdfInputFilePath << " is not a regular file.";
        ASSERT_NO_THROW (mHdfInputFile = HDFMultiArrayIO::CreateReader (mHdfInputFilePath.string()));
	}
};

TEST_F (TestHDFMultiArrayIORead, DatasetDimensionality)
{
    ASSERT_TRUE (static_cast<bool>(mHdfInputFile));
    int numDims = mHdfInputFile->Dimensionality ("/g1/foo1");
	EXPECT_EQ (mNumDims, numDims);
}


TEST_F (TestHDFMultiArrayIORead, AttributeDimensionality)
{
    ASSERT_TRUE (static_cast<bool>(mHdfInputFile));
    int numDims = mHdfInputFile->Dimensionality ("/g1/foo1", "bar1");
	EXPECT_EQ (1, numDims);
}


TEST_F (TestHDFMultiArrayIORead, DatasetShape)
{
    ASSERT_TRUE (static_cast<bool>(mHdfInputFile));

    std::vector<int> shape = mHdfInputFile->Shape ("/g1/foo1");
	ASSERT_EQ (mNumDims, numeric_cast<int> (shape.size()));

	for (size_t i = 0; i < mShape.size(); ++i)
	{
		EXPECT_EQ (mShape[i], shape[i]);
	}
}


TEST_F (TestHDFMultiArrayIORead, AttributeShape)
{
    ASSERT_TRUE (static_cast<bool>(mHdfInputFile));

    std::vector<int> shape = mHdfInputFile->Shape ("/g1/foo1", "bar1");
	ASSERT_EQ (1u, shape.size());

	for (size_t i = 0; i < mAttrShape.size(); ++i)
	{
		EXPECT_EQ (2, shape[i]);
	}
}


TEST_F (TestHDFMultiArrayIORead, DatasetRead)
{
    ASSERT_TRUE (static_cast<bool>(mHdfInputFile));

    boost::multi_array<int, 3> actual (mHdfInputFile->Read <int, 3> ("/g1/foo1"));

	ASSERT_EQ (2u, actual.size());
	ASSERT_EQ (3u, actual[0].size());
	ASSERT_EQ (5u, actual[0][0].size());

	EXPECT_EQ (42, actual[0][0][0]);
	EXPECT_EQ (4, actual[0][1][2]);
	EXPECT_EQ (2, actual[1][2][3]);
	EXPECT_EQ (5, actual[1][0][4]);
}


TEST_F (TestHDFMultiArrayIORead, AttributeRead)
{
    ASSERT_TRUE (static_cast<bool>(mHdfInputFile));

    boost::multi_array <int, 1> actual (mHdfInputFile->Read <int, 1> ("/g1/foo1", "bar1"));

	ASSERT_EQ (2u, actual.size());

	EXPECT_EQ (0, actual[0]);
	EXPECT_EQ (9, actual[1]);
}



struct TestHDFMultiArrayIOWrite : public ::testing::Test
{
    path mHdfOutputFilePath;
    unique_ptr<HDFMultiArrayIO> mHdfOutputFile;

    TestHDFMultiArrayIOWrite()
    {
        mHdfOutputFilePath = boost::filesystem::unique_path("%%%%%%%%.h5");
    }

    ~TestHDFMultiArrayIOWrite ()
    {}

    void SetUp(void)
    {
        ASSERT_NO_THROW (mHdfOutputFile = HDFMultiArrayIO::CreateWriter (mHdfOutputFilePath.string()));
        ASSERT_TRUE(static_cast<bool>(mHdfOutputFile));
    }

    void TearDown()
    {
        mHdfOutputFile.reset();
        boost::filesystem::remove(mHdfOutputFilePath);
    }
};




TEST_F (TestHDFMultiArrayIOWrite, AttributeWrite)
{
    mHdfOutputFile->WriteScalarAttribute(3, "FooBar", "/");
    ASSERT_EQ(3, mHdfOutputFile->ReadScalarAttribute<int>("FooBar", "/"));
}

TEST_F (TestHDFMultiArrayIOWrite, CreateGroup)
{
    mHdfOutputFile->CreateGroup("/PacBio");
    mHdfOutputFile->CreateGroup("/PacBio");  // should just do nothing here...
    mHdfOutputFile->CreateGroup("/PacBio/Sequel");

    mHdfOutputFile->WriteScalarAttribute(3, "FooBar", "/PacBio/Sequel");
    ASSERT_EQ(3, mHdfOutputFile->ReadScalarAttribute<int>("FooBar", "/PacBio/Sequel"));
}


TEST_F (TestHDFMultiArrayIOWrite, Write1DDataset)
{
    using FArray1D = boost::multi_array<float, 1>;

    FArray1D data(boost::extents[3]);
    data[0] = 1.0;
    data[1] = 2.0;
    data[2] = 3.0;

    mHdfOutputFile->CreateGroup("/PacBio");
    mHdfOutputFile->Write("/PacBio/Numbers", data);

    auto actual = mHdfOutputFile->Read<float, 1>("/PacBio/Numbers");
    for (int i = 0; i < 3; i++)
    {
        ASSERT_EQ(data[i], actual[i]);
    }
}


TEST_F (TestHDFMultiArrayIOWrite, Write2DDataset)
{
    using FArray2D = boost::multi_array<float, 2>;

    FArray2D data(boost::extents[3][2]);
    int v = 0;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 2; j++)
            data[i][j] = static_cast<float>(v++);

    mHdfOutputFile->CreateGroup("/PacBio");
    mHdfOutputFile->Write("/PacBio/Numbers", data);

    auto actual = mHdfOutputFile->Read<float, 2>("/PacBio/Numbers");
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 2; j++)
            ASSERT_EQ(data[i][j], actual[i][j]);
}



TEST_F (TestHDFMultiArrayIOWrite, Write1DDatasetFromVector)
{
    using FVector = std::vector<float>;
    using IVector = std::vector<int>;
    using StrVector = std::vector<string>;

    FVector someFloats { 0.0, 1.0, 2.0 };
    IVector someInts   { 3, 4, 5, 6 };
    StrVector someStrings {"Four", "score", "and", "seven"};

    mHdfOutputFile->CreateGroup("/Numbers");
    mHdfOutputFile->Write("/Numbers/Floats", someFloats);
    mHdfOutputFile->Write("/Numbers/Ints",   someInts);
    mHdfOutputFile->Write("/Strings", someStrings);

    auto floatsRead = mHdfOutputFile->Read<float, 1>("/Numbers/Floats");
    ASSERT_EQ(someFloats.size(), floatsRead.size());
    for (unsigned int i = 0; i < someFloats.size(); i++)
        ASSERT_EQ(someFloats[i], floatsRead[i]);

    auto intsRead = mHdfOutputFile->Read<float, 1>("/Numbers/Ints");
    ASSERT_EQ(someInts.size(), intsRead.size());
    for (unsigned int i = 0; i < someInts.size(); i++)
        ASSERT_EQ(someInts[i], intsRead[i]);

    const auto stringsRead = mHdfOutputFile->ReadStringDataset("/Strings");
    ASSERT_EQ (someStrings.size(), stringsRead.size());
    for (unsigned int i = 0; i < someStrings.size(); ++i)
        ASSERT_EQ (someStrings[i], stringsRead[i]);
}


}}} // PacBio::Primary::Basecaller
