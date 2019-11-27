#include <gtest/gtest.h>

#include <pacbio/primary/Codec.h>

using namespace PacBio::Primary;

TEST(Codec, FrameToCode)
{
    Codec c;
    uint16_t frame = 0;

    // Test lookup implementation against the spec
    for (uint16_t idx = 0; idx < 64; ++idx, ++frame)
    {
        ASSERT_EQ(c.FrameToCode(idx), idx);
    }
    for (uint16_t idx = 0; idx < 128; ++idx, ++frame)
    {
        auto rounding = (idx % 2 == 1 ? 1 : 0);
        ASSERT_EQ(c.FrameToCode(frame), 64 + idx/2 + rounding);
    }
    for (uint16_t idx = 0; idx < 256; ++idx, ++frame)
    {
        auto rounding = (idx % 4 < 2 ? 0 : 1);
        ASSERT_EQ(c.FrameToCode(frame), 128 + idx/4 + rounding);
    }
    for (uint16_t idx = 0; idx < 505; ++idx, ++frame)
    {
        auto rounding = (idx % 8 < 4 ? 0 : 1);
        ASSERT_EQ(c.FrameToCode(frame), 192 + idx/8 + rounding);
    }

    // Now make sure the new compute based implementation matches exactly
    for (uint16_t i = 0; i < 1024; ++i)
    {
        ASSERT_EQ(c.FrameToCode(i), Codec::Experimental::FrameToCode(i));
    }
}

TEST(Codec, CodeToFrame)
{
    Codec c;

    // Test lookup implementation against the spec
    for (uint16_t code = 0; code < 64; ++code)
    {
        ASSERT_EQ(c.CodeToFrame(static_cast<uint8_t>(code)), code);
    }
    for (uint16_t code = 64; code < 128; ++code)
    {
        ASSERT_EQ(c.CodeToFrame(static_cast<uint8_t>(code)), (code - 64)*2+64);
    }
    for (uint16_t code = 128; code < 192; ++code)
    {
        ASSERT_EQ(c.CodeToFrame(static_cast<uint8_t>(code)), (code - 128)*4+192);
    }
    for (uint16_t code = 192; code < 256; ++code)
    {
        ASSERT_EQ(c.CodeToFrame(static_cast<uint8_t>(code)), (code - 192)*8+448);
    }

    // Now make sure the new compute based implementation matches exactly
    for (uint16_t i = 0; i <= 255; ++i)
    {
        ASSERT_EQ(c.CodeToFrame(static_cast<uint8_t>(i)), 
                  Codec::Experimental::CodeToFrame(static_cast<uint8_t>(i)));
    }
}

TEST(Codec, DownsampleFrame)
{
    Codec c;

    // Just make sure the two implementations match.  The original function has
    // a output cap at 255 that I'm not sure is intentional, so I don't want
    // to explicitly enshrine it in a test.
    for (uint16_t i = 0; i < 1024; ++i)
    {
        ASSERT_EQ(c.DownsampleFrame(i), Codec::Experimental::DownsampleFrame(i));
    }
}
