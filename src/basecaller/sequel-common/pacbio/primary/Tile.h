#ifndef _PACBIO_PRIMARY_TILE_H_
#define _PACBIO_PRIMARY_TILE_H_

// Copyright (c) 2014, Pacific Biosciences of California, Inc.
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
//  C++ header to define the DMA memory unit known as a Tile
//
// Programmer: Mark Lakata

#ifdef __MIC
#include <immintrin.h>
#elif defined(__AVX__)
#include <immintrin.h>
#endif

#include <memory>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>
#include <iomanip>


#include <pacbio/PBException.h>

#ifndef TILE_DEPTH
#define TILE_DEPTH 512
#endif

#ifndef TILE_WIDTH
#define TILE_WIDTH 32
#endif

#ifndef PIXEL_SIZE
#define PIXEL_SIZE 2
#endif

#ifndef TILE_SIZE
#define TILE_SIZE TILE_DEPTH*TILE_WIDTH*PIXEL_SIZE
#endif


namespace PacBio {
namespace Primary {

  /// A Tile is the basis of all shared memory transactions. It is designed to be 32K
  /// to be granular enough to be flexible for rearrangement (ie transposes) but
  /// large enough to be efficient with caches and DMA transfers.
class Tile
{
protected:
    Tile() {} //this is to prohibit public construction. To construct this object, use the factory method 'make'
public:
    /// allocate a single Tile on the Heap.
    ///
    static Tile* make()
    {
        return new Tile;
    }

    /// allocate a single Tile on the Heap, and mark it as a header tile.
    ///
    static Tile* makeHeader()
    {
        auto x = new Tile;
        x->FormatHeader();
        return x;
    }

    /// formats a tile to look like a Header Tile. This is useful for emulation,
    /// as this normally done by the Wolverine.
    void FormatHeader()
    {
        memset(this, 0, sizeof(Tile));
        this->Header(0x8000);
        this->MagicWord(MAGIC_WORD);
        this->Version(VERSION_MAX);
        this->NumFramesInHeader(NumFrames);
    }
    /// Allocate an array of Tiles on the heap.
    ///
    static Tile* make(size_t count)
    {
        return new Tile[count];
    }

    //std::array<uint8_t,TILE_SIZE> data;
    uint8_t data[TILE_SIZE];
    static const uint32_t NumFrames = TILE_DEPTH;
    static const uint32_t NumPixels = TILE_WIDTH;
    static const size_t Alignment = 4096; // required by Intel Phi DMA
    static const uint32_t MAGIC_WORD = 0x00000003;
    static const uint32_t VERSION_MIN = 0x00000000;
    static const uint32_t VERSION_MAX = 0x00000000;

    int16_t* Data16() { return reinterpret_cast<int16_t*>(data); }
    const int16_t* Data16() const { return reinterpret_cast<const int16_t*>(data); }

    uint16_t HeaderWord(void) const { return Extract<uint16_t>(0);}

    bool IsHeader() const
    {
        return HeaderWord() == 0x8000;
    }

    uint32_t MagicWord(void) const { return Extract<uint32_t>(64); }

    uint32_t Version(void) const { return Extract<uint32_t>(64 + 4); }

    uint64_t FirstFrameIndex(void) const { return Extract<uint64_t>(128); }

    uint64_t FirstFrameTimeStamp(void) const { return Extract<uint64_t>(128 + 8); }

    uint32_t FirstFrameConfig(void) const { return Extract<uint32_t>(128 + 16); }

    uint64_t LastFrameIndex(void) const { return Extract<uint64_t>(192); }

    uint64_t LastFrameTimeStamp(void) const { return Extract<uint64_t>(192 + 8); }

    uint32_t LastFrameConfig(void) const { return Extract<uint32_t>(192 + 16); }

    uint32_t NumFramesInHeader(void) const { return Extract<uint32_t>(0x40 * sizeof(uint32_t)); }

    bool Flushed(void) const { return (Extract<uint8_t>(0x41 * sizeof(uint32_t)) & 0x00000001) != 0; }

    void Header(uint16_t val) { return Insert<uint16_t>(0, val); }

    void MagicWord(uint32_t val) { return Insert<uint32_t>(64, val); }

    void Version(uint32_t val) { return Insert<uint32_t>(64 + 4, val); }

    void FirstFrameIndex(uint64_t val) { return Insert<uint64_t>(128, val); }

    void FirstFrameTimeStamp(uint64_t val) { return Insert<uint64_t>(128 + 8, val); }

    void FirstFrameConfig(uint32_t val) { return Insert<uint32_t>(128 + 16, val); }

    void LastFrameIndex(uint64_t val) { return Insert<uint64_t>(192, val); }

    void LastFrameTimeStamp(uint64_t val) { return Insert<uint64_t>(192 + 8, val); }

    void LastFrameConfig(uint32_t val) { return Insert<uint32_t>(192 + 16, val); }

    void NumFramesInHeader(uint32_t val) { return Insert<uint32_t>(0x40 * sizeof(uint32_t), val); }

    bool VersionSupported() const
    {
        return ((int32_t) Version() >= (int32_t) Tile::VERSION_MIN && Version() <= Tile::VERSION_MAX);
    }

    std::vector<uint64_t> ErroredFrames() const
    {
        std::vector<uint64_t> errors;
        for (uint64_t i = 0; i < NumFrames / 64; i++)
        {
            uint64_t word = Extract<uint64_t>(1024U + i * sizeof(uint64_t));
            if (word != 0)
            {
                for (uint32_t bit = 0; bit < 64; bit++)
                {
                    if ((word >> bit) & 1)
                    {
                        uint64_t frame = i * 64 + bit + FirstFrameIndex();
                        errors.push_back(frame);
                    }
                }
            }
        }
        return errors;
    }

    /// fills in a iterable container with the frame offsets within the chunk that reported
    /// an error, and also returns the total number of errored frames. The number of error frames
    /// may be larger than the iterators allows.
    template<typename Iter>
    uint32_t ErroredFrames(Iter first, Iter last) const
    {
        uint32_t count = 0;
        for (uint64_t i = 0; i < NumFrames; i += 64)
        {
            uint64_t word = Extract<uint64_t>(1024U + (i / 64) * sizeof(uint64_t));
            if (word != 0)
            {
                for (uint32_t bit = 0; bit < 64; bit++)
                {
                    if ((word >> bit) & 1)
                    {
                        if (first != last)
                        {
                            *first = i + bit;
                            first++;
                        }
                        count++;
                    }
                }
            }
        }
        return count;
    }

    template<class T>
    T Extract(size_t offset) const
    {
        if (offset >= sizeof(Tile)) throw PBException("offset out of range");
        T t;
        memcpy(&t, data + offset, sizeof(T));
        return t;
    }

    /// Write an object into the memory space of the object
    template<class T>
    void Insert(size_t offset, T value)
    {
        if (offset >= sizeof(Tile)) throw PBException("offset out of range");
        memcpy(data + offset, &value, sizeof(T));
    }

    /// Initializes the current object as a header tile (non-data).
    ///
    void CreateHeaderTile(uint64_t firstFrameIndex, uint64_t firstFrameTimeStamp, uint32_t firstFrameConfig)
    {
        memset(this, 0, sizeof(Tile));
        Header(0x8000);
        MagicWord(MAGIC_WORD);
        Version(VERSION_MAX);
        FirstFrameIndex(firstFrameIndex);
        FirstFrameConfig(firstFrameConfig);
        FirstFrameTimeStamp(firstFrameTimeStamp);
        LastFrameIndex(firstFrameIndex + NumFrames - 1);
        LastFrameConfig(firstFrameConfig);
        LastFrameTimeStamp(firstFrameTimeStamp + (NumFrames - 1) * 10000); // microseconds
        NumFramesInHeader(NumFrames);
    }

    /// Set a static pattern onto all bytes using the lowest 8 bits of the argument.
    ///
    void SetPattern(int i)
    {
#ifdef __MIC
        // check if tile is SIMD aligned
        if ((reinterpret_cast<size_t>(data) & 0x3f) == 0)
        {
            __m512i patt = _mm512_set1_epi32(0x01010101 * (i&0xFF));
            for (uint32_t x = 0; x < sizeof(Tile); x += 64)
            {
                _mm512_store_epi32(data + x, patt);
            }
        }
        else
        {
            memset(data, i, sizeof(data));
        }
#else
        memset(data, i, sizeof(data));
#endif
    }

    void SetPattern2(int tileBaseline)
    {
        int16_t* data16 = reinterpret_cast<int16_t*>(data);
        for (uint32_t iframe = 0; iframe < NumFrames; iframe++)
        {
            for (uint32_t pixel = 0; pixel < NumPixels; pixel++)
            {
                int16_t v = static_cast<int16_t>(pixel + iframe * 10 + tileBaseline);
                data16[pixel + iframe * NumPixels] = v;
            }
        }
    }
    /// This pattern files with chunk as major, pixel as middle and frame index as minor
    /// parts of the pattern.  Thus the first ZMW will have values that look like (in time):
    /// (0,10),(1,11),(2,12),(3,13) ... (511,8)
    /// The second ZMWS will have
    /// (2,12),(3,13),(4,14),(5,15),... (1,10)

    /// pixel goes [0,2*numZMWs)
    /// frame goes [0,infinity]
    static int16_t GetPattern3Pixel(uint32_t pixel, uint32_t frame)
    {
        return static_cast<int16_t>((pixel * 10 + frame) & 0xFFFF);
    }

    void SetPattern3(uint32_t basePixel, uint32_t baseFrame)
    {
        int16_t* data16 = reinterpret_cast<int16_t*>(data);
        for (uint32_t iframe = 0; iframe < NumFrames; iframe++)
        {
            for (uint32_t pixel = 0; pixel < NumPixels; pixel++)
            {
                int16_t v = GetPattern3Pixel(basePixel + pixel, baseFrame + iframe);
                data16[pixel + iframe * NumPixels] = v;
            }
        }
    }

    /// verifies that all bytes match the low 8 bits of the argument
    void CheckPattern(int i) const
    {
#ifdef __MIC
        if ((reinterpret_cast<size_t>(data) & 0x3f) == 0)
        {
            __m512i patt = _mm512_set1_epi32(0x01010101 * (i&0xFF));
            for (size_t x = 0; x < sizeof(Tile); x += 64)
            {
                __m512i read = _mm512_load_epi32(data + x);
                __mmask16 c = _mm512_cmpeq_epi32_mask(read, patt);
                if (_mm512_mask2int(c) != 0xFFFF)
                {
                    std::stringstream s;

                    s << "data" << std::hex << x << ": (hex) ";
                    for(int j=0;j<64;j++) s << std::hex << (int)(data[x+j]) << " ";
                    s << std::endl;
                    s << "patt" << std::hex << x << ": (hex)";
                    for(int j=0;j<64;j++) s << std::hex << (i&0xFF) << " ";
                    s << std::dec << std::endl;
                    throw PBException("mismatch:" + s.str());
                }
            }
            return;
        }
#elif defined(__AVX__) && 0  // 32 bit test
        uint32_t patt = 0x01010101 * (i&0xFF);
        const uint32_t* data32 = (const uint32_t*)data;
        for (int x = 0; x < sizeof(Tile)/sizeof(uint32_t); x++)
        {
            uint32_t read1 = data32[x];

            if (read1 != patt)
            {
                std::cerr << "data" << std::hex << x << ": (hex) ";
                for(int j=0;j<4;j++) std::cerr << std::hex << (int)(data[x+j]) << " ";
                std::cerr << std::endl;
                std::cerr << "patt" << std::hex << x << ": (hex)";
                for(int j=0;j<4;j++) std::cerr << std::hex << (i&0xFF) << " ";
                std::cerr << std::dec << std::endl;
                throw PBException("mismatch");
            }
        }
        return;
#elif defined(__AVX__) && 0 // 64 bit test
        uint64_t patt = 0x010101010101010101010101ULL * (i&0xFF);
        const uint64_t* data64 = (const uint64_t*)data;
        for (int x = 0; x < sizeof(Tile)/sizeof(uint64_t); x++)
        {
            uint64_t read1 = data64[x];

            if (read1 != patt)
            {
                std::cerr << "data" << std::hex << x << ": (hex) ";
                for(int j=0;j<8;j++) std::cerr << std::hex << (int)(data[x+j]) << " ";
                std::cerr << std::endl;
                std::cerr << "patt" << std::hex << x << ": (hex)";
                for(int j=0;j<8;j++) std::cerr << std::hex << (i&0xFF) << " ";
                std::cerr << std::dec << std::endl;
                throw PBException("mismatch");
            }
        }
        return;
#elif defined(__AVX__)
        if ((reinterpret_cast<size_t>(data) & 0x3f) == 0)
        {
            __m256i patt = _mm256_set1_epi32(0x01010101 * (i&0xFF));
            for (uint32_t x = 0; x < sizeof(Tile); x += 64)
            {
                __m256i read1 = _mm256_load_si256((const __m256i*)(data + x));
                __m256i read2 = _mm256_load_si256((const __m256i*)(data + x + 32));

                __m256i c1 = (__m256i) _mm256_xor_ps((__m256)read1, (__m256)patt);
                __m256i c2 = (__m256i) _mm256_xor_ps((__m256)read2, (__m256)patt);

                int r1 =_mm256_testz_si256(c1,c1);
                int r2 =_mm256_testz_si256(c2,c2);
                if (r1 == 0)
                {
                    std::stringstream s;

                    s << "data" << std::hex << x << ": (hex) ";
                    for(int j=0;j<32;j++) s<< std::hex << (int)(data[x+j]) << " ";
                    s << std::endl;
                    s << "patt" << std::hex << x << ": (hex)";
                    for(int j=0;j<32;j++) s<< std::hex << (i&0xFF) << " ";
                    s << std::dec << std::endl;
                    throw PBException("mismatch:" + s.str());
                }
                if (r2 == 0)
                {
                    std::cerr << "data" << std::hex << x << ": (hex) ";
                    for(int j=0;j<32;j++) std::cerr << std::hex << (int)(data[x+j]) << " ";
                    std::cerr << std::endl;
                    std::cerr << "patt" << std::hex << x << ": (hex)";
                    for(int j=0;j<32;j++) std::cerr << std::hex << (i&0xFF) << " ";
                    std::cerr << std::dec << std::endl;
                    throw PBException("mismatch");
                }
            }
            return;
        }
#endif
        uint8_t pattern = (uint8_t) i;
        for (auto& x : data)
        {
            if (x != pattern)
            {
                std::stringstream s;
                s << "data " << std::hex << (int) x << std::dec << std::endl;
                s << "patt " << std::hex << (int) pattern << std::dec << std::endl;
                throw PBException("mismatch: " + s.str());
            }
        }
        return;
    }

    /// forces a cache update
    void Touch()
    {
        data[0] = 1;
    }

    std::string HeaderToString() const
    {
        std::stringstream ss;
        ss << "Magic Word: " << std::hex << MagicWord() << std::dec << std::endl;
        ss << "Version   : " << std::hex << Version() << std::dec << std::endl;
        ss << "First Frame Index: 0x" << std::hex << FirstFrameIndex() << " TimeStamp: 0x" << FirstFrameTimeStamp()
                << " Config: 0x" << FirstFrameConfig() << std::dec << std::endl;
        ss << "Last  Frame Index: 0x" << std::hex << LastFrameIndex() << " TimeStamp: 0x" << LastFrameTimeStamp()
                << " Config: 0x" << LastFrameConfig() << std::dec << std::endl;
        ss << "NumFramesInHeader: 0x" << std::hex << NumFramesInHeader() << std::dec << std::endl;
        return ss.str();
    }

    /// \returns a string with all the tiles written as UInt32 hex, each word prefixed by offset
    /// and every 16 words wrapped with a newline. Note that this will write pixel pairs in
    /// reverse order, as the architecture is little endian. For example, if pixel[0]=0xaaaa,
    /// pixel[1]=0xbbbb, the uint32 will display as bbbbaaaa.
    /// If you only care about pixel values, the other RawUInt16ToString or RawInt16ToString will
    /// be less confusing.
    /// \param numValues : a limit on the number of values to display. The default will
    ///                    display all the values of the Tile.
    std::string RawToString(uint32_t values=sizeof(Tile)/sizeof(uint32_t)) const
    {
        std::stringstream ss;
        for (uint32_t i = 0; i < values; i++)
        {
            if ((i % 16) == 0) ss << std::hex << " " << std::setw(8) << i << " : ";
            ss << std::hex << " " << std::setw(8) << Extract<uint32_t>(i * 4) << std::dec ;
            if ((i % 16) == 15) ss << std::endl;
        }
        return ss.str();
    }

    /// \returns a string with all the tiles written as UInt16 hex, each word prefixed by offset
    ///          and every 32 words wrapped with a newline
    /// \param numValues : a limit on the number of values to display. The default will
    ///                    display all the values of the Tile.
    std::string RawUInt16ToString(uint32_t numValues=sizeof(Tile)/sizeof(uint16_t)) const
    {
        std::stringstream ss;
        for (uint32_t i = 0; i < numValues; i++)
        {
            if (i % 32 == 0)
            {
                ss << "[" << i << "]: ";
            }
            ss << std::hex << Extract<uint16_t>(i*2) << std::dec << " ";
            if (i % 32 == 11)
            {
                ss << std::endl;
            }
        }
        return ss.str();
    }

    /// \returns a string with all the tiles written as Int16 decimal, each word prefixed by offset
    ///          and every 32 words wrapped with a newline.
    /// \param numValues : a limit on the number of values to display. The default will
    ///                    display all the values of the Tile.
    std::string RawInt16ToString(uint32_t numValues=sizeof(Tile)/sizeof(int16_t)) const
    {
        std::stringstream ss;
        for (uint32_t i = 0; i < numValues; i++)
        {
            if (i % 32 == 0)
            {
                ss << "[" << i << "]: ";
            }
            ss << Extract<int16_t>(i*2) << " ";
            if (i % 32 == 31)
            {
                ss << std::endl;
            }
        }
        return ss.str();
    }

    void* operator new[](std::size_t request)
    {
        return operator new(request);
    }

    void* operator new(std::size_t request)
    {
        static const size_t ptr_alloc = sizeof(void*);
        static const size_t align_size = 4096;
        static const size_t request_size = request + align_size;
        static const size_t needed = ptr_alloc + request_size;

        void* alloc = ::operator new(needed);
#if 1
        void* ptr = reinterpret_cast<void*>((reinterpret_cast<uint64_t>(alloc) + ptr_alloc + align_size - 1) &
                                            ~(align_size - 1));
#else
        void* ptr = std::align(align_size, sizeof(Tile),
                alloc+ptr_alloc, request_size);
#endif

        ((void**) ptr)[-1] = alloc; // save for delete calls to use
        return ptr;
    }

    void operator delete(void* ptr)
    {
        if (ptr) // 0 is valid, but a noop, so prevent passing negative memory
        {
            void* alloc = ((void**) ptr)[-1];
            ::operator delete(alloc);
        }
    }

    void operator delete[](void* ptr)
    {
        if (ptr) // 0 is valid, but a noop, so prevent passing negative memory
        {
            void* alloc = ((void**) ptr)[-1];
            ::operator delete(alloc);
        }
    }

    uint64_t Checksum() const
    {
        uint64_t sum = 0;
        const uint64_t* ptr = reinterpret_cast<const uint64_t*>(data);
        for (size_t i = 0; i < sizeof(Tile) / sizeof(uint64_t); i++)
        {
            sum += ptr[i];
        }
        return sum;
    }

    uint64_t ZeroPixelCnt(uint32_t frames)
    {
        uint64_t cnt = 0;
        int16_t* data16 = reinterpret_cast<int16_t*>(data);
        uint32_t offset = 0;
        for (uint32_t i = 0; i < frames; i++)
        {
            for (uint32_t j = 0; j < NumPixels; j++)
            {
                if (data16[offset] == 0) cnt++;
                offset++;
            }
        }
        return cnt;
    }

    int16_t GetPixel(uint32_t ipixel, uint32_t iframe) const
    {
        const int16_t* data16 = reinterpret_cast<const int16_t*>(data);
        uint32_t offset = ipixel + NumPixels * iframe;
        if (offset >= NumPixels * NumFrames) throw PBException("Out of range");
        return data16[offset];
    }

    void SetPixel(uint32_t ipixel, uint32_t iframe, int16_t value)
    {
        int16_t* data16 = reinterpret_cast<int16_t*>(data);
        uint32_t offset = ipixel + NumPixels * iframe;
        if (offset >= NumPixels * NumFrames) throw PBException("Out of range");
        data16[offset] = value;
    }

    /// \returns true if the tile data is the same as the other tile data
    bool Compare(const Tile& other, std::ostream& errorStream)
    {
        if (this->IsHeader() || other.IsHeader()) throw PBException("not implemented for header tiles");

        int errorCount = 0;
        for(uint32_t iframe=0; iframe < Tile::NumFrames;iframe++)
        {
            for (uint32_t ipixel = 0; ipixel < Tile::NumPixels; ipixel++)
            {
                int16_t thisValue = GetPixel(ipixel, iframe);
                int16_t thatValue = other.GetPixel(ipixel, iframe);
                if (thisValue != thatValue)
                {
                    if (true || errorCount ==0)
                    {
                        errorStream << "Mismatch [pixel=" << ipixel << ",frame:" << iframe
                            << "] " << thisValue << " != " << thatValue << std::endl;
                    }
                    errorCount++;
                }
            }
        }
        errorStream << "Total errors:" << errorCount << std::endl;
        return errorCount == 0;
    }
};

}}

#endif

