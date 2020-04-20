//
// Created by mlakata on 3/20/20.
//
// this benchmarking test is stolen from Benchy code from primaryanalysis/Sequel/acquisition/src.
//

#ifndef PA_MONGO_UNPACKER_H
#define PA_MONGO_UNPACKER_H

#include <vector>
#include <pacbio/PBException.h>
#include <stdint.h>
#include <immintrin.h>
#include <pacbio/logging/Logger.h>
#include <pacbio/utilities/Sysinfo.h>
#include <pacbio/utilities/Time.h>
#include <pacbio/HugePage.h>

namespace PacBio {
namespace Primary {

struct Tile
{
    static const uint32_t NumPixels = 32;
};

class Unpacker
{
public:

/// Scatters 12 bit pixels to Tiles, using a pedantic simple C++ method.
/// \param rasterPixels - 12 bit data, packed together in memory, little endian order
/// \param numPixels - number of rasterPixels
/// \param destPacket - memory to write to
/// \returns number of bytes written
    size_t Expand12BitAndScatter(
            const void *rasterPixels,
            const uint64_t numPixels,
            void *destPacket)
    {
        const uint8_t *rasterU8 = reinterpret_cast<const uint8_t *>(rasterPixels);
        int16_t* destPacketInt16= static_cast<int16_t*>(destPacket);
//        int tilesUsed = 0;
        for (uint64_t i = 0; i < numPixels; i++)
        {
            int16_t value;
            if ((i & 1) == 0)
            {
// even offset
                uint64_t offset = i * 12 / 8;
// 0->0, 2->3, 4->6, ...
                value = static_cast<int16_t>(static_cast<int16_t>((uint16_t) rasterU8[offset] << 4 |
                                                                  (((uint16_t) rasterU8[offset + 1] & 0x0F) << 12))
                        >> 4);
            } else
            {
// odd offset
// 1->1, 3->4, 5->7, ...
                uint64_t offset = i * 12 / 8;
                value = static_cast<int16_t>(static_cast<int16_t>((((uint16_t) rasterU8[offset] & 0xF0)) |
                                                                  ((uint16_t) rasterU8[offset + 1] << 8)) >> 4);

            }
            *destPacketInt16 = value;
#if 1
            destPacketInt16++; // this is not right
#else
            if ((i + 1) % Tile::NumPixels == 0)
            {
                tilesUsed++;
                if (i + 1 < numPixels) // not done yet, need to go to the next tile
                {
                    destTileIter++;
                    if (destTileIter == endTileIter)
                    {
                        PBLOG_ERROR << "Tried to unpack " << numPixels << " used " << tilesUsed << " tiles, needed "
                                    << (numPixels * 1.0 / Tile::NumPixels);
                        throw PBException("Not enough tiles to unpack the frame data, hit end of tiles");
                    }
                    if (*destTileIter == nullptr)
                    {
                        PBLOG_ERROR << "Tried to unpack " << numPixels << " used " << tilesUsed << " tiles, needed "
                                    << (numPixels * 1.0 / Tile::NumPixels);
                        throw PBException("Not enough tiles to unpack the frame data, hit nullptr");
                    }
                }
            }
#endif
        }

        size_t numBytes = reinterpret_cast<uint8_t*>(destPacketInt16) - reinterpret_cast<uint8_t*>(destPacket);
        return numBytes;
    }


// This code has bit rotted. Leaving it here for historical reference in case we want to do
// some more benchmarking.

///// Scatters 16 bit pixels to Tiles. This is not optimized and
///// will probably never be used if the sensor always outputs 12 bit data.
//    long Scatter(
//            const int16_t *rasterPixels,
//            const uint64_t numPixels,
//            void *destPacket
//    )
//    {
//        for (uint64_t i = 0; i < numPixels; i++)
//        {
//            (*destTileIter)->SetPixel(i % Tile::NumPixels, destFrameOffset, rasterPixels[i]);
//            if ((i + 1) % Tile::NumPixels == 0)
//            {
//                destTileIter++;
//            }
//        }
//        return (numPixels + Tile::NumPixels - 1) / Tile::NumPixels;
//    }

// 128 input bits have 10 12-bit sample words, with 8 bits left over.
// 128 output bits is limited to 8 output 16-bit words.
// So we can transfer 8 words at a time, and need to move the input pointer
// by 8*12=48 bits = 6 bytes.
// This implementation only uses SSE instructions. It is a little slower than
// the AVX512 implementation, which seems to have a short cut using masked
// shifts.
// We use the `struct` only to allow this function to be
// instantiated in a template, so that it is inlined.
    struct Expander_SSE
    {
        /// \param input - SIMD register that contains 8 12-bit signed pixel values
        /// \returns - SIMD register that contains 8 16-bit signed pixel values
        inline static __m128i Expand12BitTo16SingleRegister(const __m128i &input)
        {
            //std::cout << "input   :" <<  input << std::endl;

            const __m128i control0 = _mm_set_epi16(
                    0x8080, 0x0a09,
                    0x8080, 0x0706,
                    0x8080, 0x0403,
                    0x8080, 0x0100
            );

            //    std::cout << "control0:" << control0 << std::endl;

            const __m128i mask0 = _mm_set_epi16(
                    0x0000, 0xffff,
                    0x0000, 0xffff,
                    0x0000, 0xffff,
                    0x0000, 0xffff);
            //    std::cout << "mask0   :" << std::hex << std::setw(8) <<  mask0 << std::dec << std::endl;

            __m128i int0 = _mm_shuffle_epi8(input, control0);
            //    std::cout << "int0 0  :" << std::hex << std::setw(8) <<  int0 << std::dec << std::endl;

            // this sign extends the 12 bit value to 16 bits, by shifting the sign bit into the MSB (bit 15),
            // then using the arithmetic right-shift to sign-extend it down to bit 11
            int0 = _mm_slli_epi16(int0, 4); // shift left 4
            int0 = _mm_srai_epi16(int0, 4); // shift right 4, with sign extension

            int0 = _mm_and_si128(int0, mask0);
            //std::cout << "int0 1  :" << std::hex << std::setw(8) <<  int0 << std::dec << std::endl;

            const __m128i control1 = _mm_set_epi16(
                    0x0b0a, 0x8080,
                    0x0807, 0x8080,
                    0x0504, 0x8080,
                    0x0201, 0x8080
            );
            //    std::cout << "control1:" << control0 << std::endl;

            __m128i int1 = _mm_shuffle_epi8(input, control1);
            //std::cout << "int1 0  :" << int1 << std::endl;

            int1 = _mm_srai_epi16(int1, 4); // shift right 4, with sign extension
            //std::cout << "int1 1  :" << int1 << std::endl;

            const __m128i mask1 = _mm_set_epi16(
                    0xFFFF, 0x0,
                    0xFFFF, 0x0,
                    0xFFFF, 0x0,
                    0xFFFF, 0x0);

            int1 = _mm_and_si128(int1, mask1);
            //std::cout << "int1 2  :" << int1 << std::endl;

            __m128i final = _mm_or_si128(int0, int1);
            //std::cout << "final   :" << final << " ***" << std::endl;

            return final;
        }
    };

// 128 input bits has 10 12-bit sample words, with 8 bits left over.
// 128 output bits is limited to 8 output 16-bit words.
// So we can transfer 8 words at a time, and need to move the input pointer
// by 8*12=48 bits = 6 bytes.
// This implementation uses AVX512 instructions. It is a little faster than
// the SSE implementation, by using _mm_mask_srai_epi16 instruction
// which does the 16 bit masking and final OR'ing in 1 step, instead of 3.
    struct Expander_AVX512
    {
        /// \param input - SIMD register that contains 8 12-bit signed pixel values
        /// \returns - SIMD register that contains 8 16-bit signed pixel values
        inline static __m128i Expand12BitTo16SingleRegister(const __m128i &input)
        {
#if defined(__AVX512BW__) && defined(__AVX512VL__)
            //std::cout << "input   :" <<  input << std::endl;

            constexpr __m128i control0 = _mm_set_epi16(
                    0x8080, 0x0a09,
                    0x8080, 0x0706,
                    0x8080, 0x0403,
                    0x8080, 0x0100
            );

    //    std::cout << "control0:" << control0 << std::endl;

            __m128i int0 = _mm_shuffle_epi8(input, control0);
    //    std::cout << "int0 0  :" << std::hex << std::setw(8) <<  int0 << std::dec << std::endl;

            // this sign extends the 12 bit value to 16 bits, by shifting the sign bit into the MSB (bit 15),
            // then using the arithmetic right-shift to sign-extend it down to bit 11
            int0 = _mm_slli_epi16(int0, 4); // shift left 4
            int0 = _mm_srai_epi16(int0, 4); // shift right 4

            //std::cout << "int0 1  :" << std::hex << std::setw(8) <<  int0 << std::dec << std::endl;

            constexpr __m128i control1 = _mm_set_epi16(
                    0x0b0a, 0x8080,
                    0x0807, 0x8080,
                    0x0504, 0x8080,
                    0x0201, 0x8080
            );
            //    std::cout << "control1:" << control0 << std::endl;

            __m128i int1 = _mm_shuffle_epi8(input, control1);
            //std::cout << "int1 0  :" << int1 << std::endl;

            constexpr __mmask8 k = 0xAA; // binary 10101010

            // this magic command does 2 things at the same time
            // 1) performs the arithmetic right shift on the odd 16-bit
            //    words. This preserves the sign bits
            // 2) replaces the even 16-bit words with the contents of int0
            // because the command is 16-bit oriented, the logical ANDs
            // with 0xFFFF is not necessary.
            // So this removed 2 AND operations and 1 OR operation compared
            // to the SSE algorithm.
            int1 = _mm_mask_srai_epi16(int0, k, int1, 4); // shift right 4
    //    std::cout << "int1 1  :" << int1 << std::endl;
            return int1;
#else
            (void) input;
            throw PBException("can't get here!");
#endif
        }
    };

/// \param rasterPixels - 12 bit data, packed together in memory, little endian order
/// \param numPixels - number of rasterPixels
/// \param destFrameOffset - which frame offset to write to
/// \param destTileIter - an iterator of Tile pointers. The caller must make sure there are enough tiles.
/// \returns number of tiles used
    template<class T>
    long Expand12BitAndScatterSIMD_template(
            const void *rasterPixels0,
            const uint64_t numPixels,
            void* destPacket
    )
    {
#if 1
        return 0;
#else
        const uint8_t *rasterPixels = static_cast<const uint8_t *>(rasterPixels0);
        uint64_t count = numPixels * 3 / 2;

        uint64_t ipixel = 0; // pixel offset within the current tile

        constexpr int iterationPixels = 8;
        constexpr int iterationBytes = (iterationPixels * 3 / 2);
        const std::vector<Tile *>::iterator destTileIterStart = destTileIter;

        while (count > 0)
        {
            // unaligned load, to retrieve enough data for at least 8 samples (96 bits).
            // The upper 32 bits ([127:96]) of the load are discarded.
            __m128i value = _mm_loadu_si128(reinterpret_cast<const __m128i *>(rasterPixels));
            // Expand the lowest 8*12 = 96 bits into 128 bits.
            // This corresponds to 8 samples.
            __m128i tileVal = T::Expand12BitTo16SingleRegister(value);
            void *tilePtr = (*destTileIter)->Data16() + destFrameOffset * Tile::NumPixels + ipixel;
            // streaming write of 8 samples
            _mm_stream_si128(reinterpret_cast<__m128i *>(tilePtr), tileVal);

            // move the pointers to the next 8 samples
            rasterPixels += iterationBytes;
            count -= iterationBytes;
            ipixel += iterationPixels;
            if (ipixel >= Tile::NumPixels)
            {
                // go to next tile
                ipixel = 0;
                destTileIter++;
            }
        }
        return numPixels > 0 ? (destTileIter - destTileIterStart + (ipixel ? 1 : 0)) : 0;
#endif
    }

/// \param rasterPixels - 12 bit data, packed together in memory, little endian order
/// \param numPixels - number of rasterPixels
/// \param destPacket - memory to write to
/// \returns number of tiles used
    size_t Expand12BitAndScatterSIMD_SSE_Unrolled(const void *rasterPixels0,
                                                uint64_t numPixels,
                                                void* destPacket)
    {
#if 0
        return 0;
#else
        // const std::vector<Tile *>::iterator destTileIterStart = destTileIter;
//        const int numPixelStart = numPixels;
        const uint8_t *rasterPixels = static_cast<const uint8_t *>(rasterPixels0);

        uint64_t ipixel = 0;
        int16_t *tilePtr = static_cast<int16_t*>(destPacket); //(*destTileIter)->Data16() + iframe * Tile::NumPixels;
        constexpr int diffPixels = 8;
        constexpr int diffBytes = (diffPixels * 3 / 2);
        constexpr int unrollFactor = 4;
        while (numPixels >= unrollFactor)
        {
            PBLOG_TRACE << " writing to " << (void *) tilePtr << " ipixel:" << ipixel << " numPixels:" << numPixels
                        << std::endl;

            // This loop is unrolled by 4X, and if we don't have a multiple of 4X, then the next
            // loop cleans up. The loads have to be unaligned because it loads 96 bits at a time.
            // The stores should always be aligned to 128 bits, because Tiles are guaranteed to
            // be 64 byte aligned.
            __m128i value0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(rasterPixels + 0 * diffBytes));
            __m128i value1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(rasterPixels + 1 * diffBytes));
            __m128i value2 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(rasterPixels + 2 * diffBytes));
            __m128i value3 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(rasterPixels + 3 * diffBytes));
            __m128i tileVal0 = Expander_SSE::Expand12BitTo16SingleRegister(value0);
            __m128i tileVal1 = Expander_SSE::Expand12BitTo16SingleRegister(value1);
            __m128i tileVal2 = Expander_SSE::Expand12BitTo16SingleRegister(value2);
            __m128i tileVal3 = Expander_SSE::Expand12BitTo16SingleRegister(value3);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(tilePtr + ipixel + diffPixels * 0), tileVal0);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(tilePtr + ipixel + diffPixels * 1), tileVal1);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(tilePtr + ipixel + diffPixels * 2), tileVal2);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(tilePtr + ipixel + diffPixels * 3), tileVal3);

            rasterPixels += diffBytes * unrollFactor;
            numPixels -= diffPixels * unrollFactor;
            ipixel += diffPixels * unrollFactor;
            if (ipixel >= Tile::NumPixels)
            {
                ipixel = 0;
#if 1
                // new simple way
                tilePtr += Tile::NumPixels;
#else
                // tile way
                destTileIter++;
#if 0
                if (count >0 && destTileIter == end) throw PBException("ran out of tiles for unpacking");
#endif
                tilePtr = (*destTileIter)->Data16() + iframe * Tile::NumPixels;
#endif
                PBLOG_TRACE << "Going to next tile, tilePtr:" << (void *) tilePtr;
            }
        }
        // clean up loop, if count is not a multiple of 4
        while (numPixels > 0)
        {
            __m128i value = _mm_loadu_si128(reinterpret_cast<const __m128i *>(rasterPixels));
            __m128i tileVal = Expander_SSE::Expand12BitTo16SingleRegister(value);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(tilePtr + ipixel), tileVal);

            rasterPixels += diffBytes;
            numPixels -= diffPixels;
            ipixel += diffPixels;
            if (ipixel >= Tile::NumPixels)
            {
                ipixel = 0;
#if 1
                // new simple way
                tilePtr += Tile::NumPixels;
#else
                destTileIter++;
#if 0
                if (count >0 && destTileIter == end) throw PBException("ran out of tiles for unpacking");
#endif
                tilePtr = (*destTileIter)->Data16() + iframe * Tile::NumPixels;
#endif
            }
        }

#if 1
        size_t numBytes = reinterpret_cast<uint8_t*>(tilePtr) - reinterpret_cast<uint8_t*>(destPacket);
        return numBytes;
#else
        return numPixelStart > 0 ? (destTileIter - destTileIterStart + (ipixel ? 1 : 0)) : 0;
#endif

#endif
    }

/// \returns true if the CPU supports the intrinsics to do the AVX512 implementation
/// of the 12-bit to 16-bit unpacking. If false, the CPU should support the SSE
/// implementation.
/// Also has a backdoor for testing, to disable AVX512 optimization, by
/// setting the environment variable "FORCE_SSE" to anything.
    bool AVX512_TransposeSupported()
    {
#if defined(__AVX512BW__) && defined(__AVX512VL__)
        static bool cached = false;
        static bool supported = false;
        if (!cached)
        {
            supported = PacBio::Utilities::Sysinfo::SupportsCPUID_Feature(bit_AVX512BW) &&
                    PacBio::Utilities::Sysinfo::SupportsCPUID_Feature(bit_AVX512VL);
            if (getenv("FORCE_SSE")) supported=  false;
            cached = true;
        }
        return supported;
#else
        return false;
#endif
    }

/// Scatters 12 bit pixels to Tiles. It selects the best algorithm
/// for the current hardware platform.
/// \param rasterPixels - 12 bit data, packed together in memory, little endian order
/// \param numPixels - number of rasterPixels
/// \param destPacket - memory to write to
/// \returns number of bytes used
    size_t Expand12BitAndScatterSIMD(
            const void *rasterPixels,
            const uint64_t numPixels,
            void* destPacket
    )
    {
        return Expand12BitAndScatterSIMD_SSE_Unrolled(rasterPixels, numPixels, destPacket);
// TODO: see if we can squeeze some more with a AVX512 build.
//    if (AVX512_TransposeSupported())
//    {
//        return Expand12BitAndScatterSIMD_template<Expander_AVX512>(rasterPixels,numPixels,destFrameOffset,destTileIter);
//    }
//    else
//    {
//        return Expand12BitAndScatterSIMD_template<Expander_SSE>(rasterPixels,numPixels,destFrameOffset,destTileIter);
//    }
    }

    static void test()
    {
        const uint64_t numPixels = 100 * (1ULL<<25);

        const void* pattern;
        void* dest;
        if (false)
        {
#if 1
            pattern = malloc(numPixels * 12 / 8);
            dest = malloc(numPixels * 2);
#else
            std::vector <uint8_t> testPattern(numPixels * 12 / 8);
            for (auto &b : testPattern) b = 0x11;

            std::vector <int16_t> packet(numPixels);

            pattern = testPattern.data();
            dest = packet.data();
#endif
        }
        else
        {
            pattern = PacBio::HugePage::Malloc(numPixels * 12 / 8);
            dest =  PacBio::HugePage::Malloc(numPixels * 2);
        }
        Unpacker unpacker;
        double t0 = PacBio::Utilities::Time::GetMonotonicTime();
        size_t convertedBytes = unpacker.Expand12BitAndScatter(pattern,
                numPixels,
                                                                 dest
                );
        if (convertedBytes != sizeof(int16_t)*numPixels) throw PBException("miscount");
        double t1 = PacBio::Utilities::Time::GetMonotonicTime();
        PBLOG_INFO << "Expand12BitAndScatter time = " << (t1-t0) << " seconds";
        t0 = t1;

        convertedBytes = unpacker.Expand12BitAndScatterSIMD(pattern,
                                                                 numPixels,
                                                            dest
        );
        if (convertedBytes != sizeof(int16_t)*numPixels) throw PBException("miscount");
        t1 = PacBio::Utilities::Time::GetMonotonicTime();
        PBLOG_INFO << "Expand12BitAndScatterSIMD time = " << (t1-t0) << " seconds";
    }
};

}} // namespaces


#endif //PA_MONGO_UNPACKER_H
