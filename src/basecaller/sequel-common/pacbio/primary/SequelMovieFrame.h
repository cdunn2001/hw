// Copyright (c) 2015, Pacific Biosciences of California, Inc.
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
/// \brief  declaration of frame object.


#ifndef PACBIO_SEQUEL_COMMON_SEQUELMOVIEFRAME_H
#define PACBIO_SEQUEL_COMMON_SEQUELMOVIEFRAME_H

#include <stdint.h>
#include <iostream>

#include <boost/numeric/conversion/cast.hpp>
#include <boost/algorithm/clamp.hpp>
#include <pacbio/PBException.h>
#include <pacbio/primary/FrameClass.h>

namespace H5 {
class DataSet;
}

namespace PacBio {
namespace Primary {

class SequelMovieFileBase;

class SequelRectangularROI;

template<class T>
class SequelMovieFrame
{
public:
    static bool FillMissingRows;

    /// Default constructors.
    SequelMovieFrame() : SequelMovieFrame(0,0) {}

    SequelMovieFrame(uint32_t rows , uint32_t cols );

    explicit SequelMovieFrame(const SequelRectangularROI& roi);

    /// Constructor from existing buffer.
    SequelMovieFrame(const int16_t* rawBuffer, uint32_t rows, uint32_t cols);

    /// Copy constructor (deep copy) (lvalue), different type
    template<class Y>
    explicit SequelMovieFrame(const SequelMovieFrame<Y>& frame);

    /// Copy constructor (deep copy) (lvalue), same type
    explicit SequelMovieFrame(const SequelMovieFrame<T>& frame);

    /// Copy constructor (with move) (rvalue).
    SequelMovieFrame(SequelMovieFrame<T>&& frame);

    /// copy assignment, different types
    template<typename Y>
    SequelMovieFrame& operator=(const SequelMovieFrame<Y>& frame);

    /// copy assignment, same types
    SequelMovieFrame& operator=(const SequelMovieFrame<T>& frame);

    /// move assignment
    template<typename Y>
    SequelMovieFrame& operator=(SequelMovieFrame<Y>&& frame);

    virtual ~SequelMovieFrame();

    void Resize(uint32_t rows, uint32_t cols);

    template<typename Y>
    void TiledReplicate(const SequelMovieFrame<Y>& frame);

    template<typename Y>
    void Insert(uint32_t rowOffset, uint32_t colOffset, const SequelMovieFrame<Y>& frame);

    T GetPixel(uint32_t row, uint32_t col) const
    {
        if (row >= NROW) throw PBException("row index " + std::to_string(row) + " GE than " + std::to_string(NROW));
        if (col >= NCOL) throw PBException("col index " + std::to_string(col) + " GE than " + std::to_string(NCOL));
        uint32_t offset = row * NCOL + col;
        return data[offset];
    }

    void SetPixel(uint32_t row, uint32_t col, const T value)
    {
        if (row >= NROW) throw PBException("row index " + std::to_string(row) + " GE than " + std::to_string(NROW));
        if (col >= NCOL) throw PBException("col index " + std::to_string(col) + " GE than " + std::to_string(NCOL));
        uint32_t offset = row * NCOL + col;
        data[offset] = value;
    }

    SequelMovieFrame<T>& SetDefaultValue(T defaultValue);

    void DumpSummary(std::ostream& s, const std::string& prefix = "", uint32_t rows = 3, uint32_t cols = 5) const;

    inline uint16_t GetPackedHeaderPduSize(void) const
    {
        return SequelMovieFrame::GetPackedHeaderPduSize(NROW, NCOL);
    }

    inline uint32_t GetPackedRowPduSize(uint32_t BitsPerPixel, FrameClass frameClass) const
    {
        return SequelMovieFrame::GetPackedRowPduSize(NROW, NCOL, BitsPerPixel, frameClass);
    }

    inline uint32_t GetPackedFrameSize(uint32_t BitsPerPixel, FrameClass frameClass) const
    {
        return SequelMovieFrame::GetPackedFrameSize(NROW, NCOL, BitsPerPixel, frameClass);
    }

    // extracts just the index
    static uint64_t PeekIndex(const void* streamBuffer0)
    {
        const uint8_t* streamBuffer = (const uint8_t*) streamBuffer0;
        uint32_t magicWord;
        memcpy(&magicWord, streamBuffer, 4);
        streamBuffer += 4;
        if (magicWord == 0x01000001)
        {
            uint64_t idx;
            memcpy(&idx, streamBuffer, 8);
            return idx;
        }
        else
        {
            return 0;
        }
    }

    // returns number of bytes parsed from streamBuffer0
    size_t Unpack(const void* streamBuffer0, uint32_t rowsTransfered, uint32_t maxLength);

    void Unpack12BitRow(const uint8_t*& streamBuffer, T* ptr);

    void Unpack10BitRow(const uint8_t*& streamBuffer, T* ptr);

    // packed format is
    // <number of words>
    // <words>
    // <number of words>
    // <words>
    // <number of words>
    // <words>
    // 0x00000000

    // returns number of bytes packed into streamBuffer
    size_t Pack(uint8_t* streamBuffer, size_t maxLen, uint32_t BitsPerPixel, FrameClass frameClass) const;

    uint32_t PackHeader(uint8_t* streamBuffer) const;

    static uint32_t PackHeaderStatic(uint8_t* streamBuffer, size_t headerSize, uint64_t frameindex, uint64_t timestamp,
                                     uint32_t cameraConfiguration);

    // assume streamBuffer is 32bit aligned
    uint32_t PackLine(int32_t row, uint8_t* streamBuffer, uint32_t BitsPerPixel, FrameClass frameClass) const;

    static int NumberOfSetBits(int i);
    static int CountBitMismatches(T a, T b);

    // returns number of errors found
    uint32_t Compare(const SequelMovieFrame& b, double tolerance = 0, bool dataOnly = false) const;

    static uint16_t GetPackedHeaderPduSize(uint32_t rows, uint32_t cols);

    static uint32_t GetPackedRowPduSize(uint32_t rows, uint32_t cols, uint32_t bitsPerPixel, FrameClass frameClass);

    static uint32_t GetPackedFrameSize(uint32_t rows, uint32_t cols, uint32_t bitsPerPixel, FrameClass frameClass);

    void CreateRandomPattern(uint32_t BitsPerPixel);

    void CreateRandomPatternForSequel(uint32_t BitsPerPixel);

    template<typename Y>
    void Add(const SequelMovieFrame<Y>& frame)
    {
        uint64_t n = NROW * NCOL;
        for (uint64_t i = 0; i < n; i++)
        {
            data[i] = static_cast<T>(data[i] + frame.data[i]);
        }
    }

    template<typename Y>
    void Add(Y offset)
    {
        uint64_t n = NROW * NCOL;
        for (uint64_t i = 0; i < n; i++)
        {
            data[i] += offset;
        }
    }

    template<typename Y>
    void Add(Y offset, T min, T max)
    {
        uint64_t n = NROW * NCOL;
        double offset_d = static_cast<double>(offset);
        for (uint64_t i = 0; i < n; i++)
        {
            double x = static_cast<double>(data[i]) + offset_d;
            if (x < min) x = min;
            if (x > max) x = max;
            data[i] = static_cast<T>(x);
        }
    }

    /// Subtracts a scalar value from all pixels. Note that the scalar must be the same type as the
    /// frame data (unlike the templated Add<Y> methods above. Because it is easy to overflow the range,
    /// the output pixels are saturated at the limits of <T>.
    /// \param offset - The subtractend that is subtracted from the original pixel data.
    template<typename Y>
    void SaturatedSubtract(Y offset)
    {
        uint64_t n = NROW * NCOL;
        for (uint64_t i = 0; i < n; i++)
        {
            double x = boost::numeric_cast<double>(data[i]) - boost::numeric_cast<double>(offset);
            static constexpr int64_t kintmin = std::numeric_limits<T>::min();
            static constexpr int64_t kintmax = std::numeric_limits<T>::max();
            // I found that that just calling `output = clamp<T>(x,...)` is a bad idea.
            // The problem with that is
            // that the argument to `clamp` (`x` in this case) is first cast to `T`.
            // The problem with that is you can't cast some values of `x` to `T`. Example,
            //  x = +32768 and T=int16_t. If you cast to T first, you get x=-32768, and the
            // clamp has no effect. And you get the worst possible result (what you want is
            // +32767, and you get -32768). If we do the clamp all in double, we are ok.
            // if T and Y are double, then we still have a corner case where we have UB, but
            // I'm not worried about that.
            double clampedX = boost::algorithm::clamp<double>(x, kintmin,kintmax);
            // If we've written the code above correctly (I saw "we" because Curtis had me
            // fix this function for this boundary condition), then this cast should never ever throw.
            data[i] = boost::numeric_cast<T>(clampedX);
        }
    }

    template<typename Y>
    void RectReplace(Y replacementPixel, uint32_t orow, uint32_t ocol, uint32_t height, uint32_t width)
    {
        uint64_t pos = 0;
        for (uint64_t i = 0; i < NROW; i++)
            for (uint64_t j = 0; j < NCOL; j++, pos++)
                if ((i >= orow) && (i < orow + height) && (j >= ocol) && (j < ocol + width))
                    data[pos] = static_cast<T>(replacementPixel);
    }

    template<typename Y>
    void Subtract(const SequelMovieFrame<Y>& frame)
    {
        uint64_t nIn = frame.NROW * frame.NCOL;
        uint64_t nOut = NROW * NCOL;
        if (nIn != nOut)
        {
            std::stringstream ss;
            ss << "Frame sizes unequal: this.ROWxCOL:" << NROW << "x" << NCOL
               << " that.ROWxCOL:" << frame.NROW << "x" << frame.NCOL;
            throw PBException(ss.str());
        }
        for (uint64_t i = 0; i < nOut; i++)
        {
            data[i] -= frame.data[i];
        }
    }

    template<typename Y>
    void AddSquare(const SequelMovieFrame<Y>& frame)
    {
        uint64_t n = NROW * NCOL;
        for (uint64_t i = 0; i < n; i++)
        {
            const Y value = frame.data[i];
            data[i] += value * value;
        }
    }

    void Scale(const double factor);

    void Scale(const double factor, T min, T max);

    void Square();

    // Clamps to extreme end (based on sign of argument).  
    // This simple function has multiple purposes:
    // 1. For signed pixel data, clampVal will be set to 0x8001 to reserve 0x8000 for use as the header tile marker.
    // 2. This function is also used to limit the range, when the pixels are constrained to 12-bits of range.
    // Thus, if given a negative value, this function will clamp the low end.  Yet when given a positive value, it clamps
    // the high end.
    // Eventually, this will be partially replaced with a 12-bit type, but more work is required to support that.
    void Clamp(const T clampVal);

    // this performs the last stage of a linear regression
    // http://stackoverflow.com/questions/5083465/fast-efficient-least-squares-fit-algorithm-in-c
    template<typename Y>
    void LinearFitYIntercept(
            const SequelMovieFrame<Y>& sumy,
            const SequelMovieFrame<Y>& sumx2,
            const SequelMovieFrame<Y>& sumx,
            const SequelMovieFrame<Y>& sumxy,
            double recip)
    {
        uint64_t n = NROW * NCOL;
        for (uint64_t i = 0; i < n; i++)
        {
            data[i] = (T) ((sumy.data[i] * sumx2.data[i] - sumx.data[i] * sumxy.data[i]) * recip);
        }
    }

    template<typename Y>
    void LinearFitYIntercept(
            const SequelMovieFrame<Y>& sumy,
            const Y sumx2,
            const Y sumx,
            const SequelMovieFrame<Y>& sumxy,
            double recip)
    {
        uint64_t n = NROW * NCOL;
        for (uint64_t i = 0; i < n; i++)
        {
            data[i] = (T) ((sumy.data[i] * sumx2 - sumx * sumxy.data[i]) * recip);
        }
    }

    template<typename Y>
    void LinearFitSlope(
            const int nPoints,
            const SequelMovieFrame<Y>& sumxy,
            const Y sumx,
            const SequelMovieFrame<Y>& sumy,
            double recip)
    {
        uint64_t n = NROW * NCOL;
        for (uint64_t i = 0; i < n; i++)
        {
            data[i] = (T) ((nPoints * sumxy.data[i] - sumx * sumy.data[i]) * recip);
        }
    }

    /// \returns size in bytes
    size_t DataSize() const;

    /// \returns size in pixels
    size_t NumPixels() const
    {
        return NROW * NCOL;
    }
    uint64_t Index() const { return index; }
    uint64_t Timestamp() const { return timestamp; }
    uint32_t CameraConfiguration() const { return cameraConfiguration;}
    void GetMinMax(T* min, T* max) const;

    /// sets the pixels to pattern1
    /// \param bitsPerPixel - the bit depth of the pixels
    void SetPattern1(uint32_t bitsPerPixel);

    /// throws an exception if the frame doesn't match pattern1
    /// \param bitsPerPixel - the bit depth of the pixels
    void ValidatePattern1(uint32_t bitsPerPixel, std::ostream* os = nullptr);

    /// throws an exception if the frame doesn't match the pattern created by the functor
    void ValidatePattern(uint32_t frame, std::function<T(uint32_t,uint32_t,uint32_t)> pattern) const;

    uint32_t NROW;
    uint32_t NCOL;

    /// an arbitrary 64 bit integer, but is understood to increase monotonically by +1
    /// during an acquisition.
    uint64_t index;
    /// an arbitrary 64 bit integer, but is understood to be the encoding the Linux epoch
    /// (elapsed time since Jan 1, 1970  0:00 UTC) in microseconds.
    uint64_t timestamp;
    /// An arbitrary 32 bit value, encoded by ICS into the sensor.
    uint32_t cameraConfiguration;
    /// Pointer to internal data. (Use with caution)
    T* Data() { return data;}
    const T* Data() const { return data;}
    /// Pointer to internal vector. (Use with caution)
    const std::vector<T> DataVector() const { return std::vector<T>(data, data+NumPixels()); }

    T* data; // TODO make this a private member
    bool valid;
private:
    friend SequelMovieFileBase;

    static int Compare(uint64_t a, uint64_t b, const char* field);
};

template<typename T>
std::ostream& operator<<(std::ostream& s, const SequelMovieFrame<T>& f)
{
    f.DumpSummary(s, " ", 20, 20);
    return s;
}

H5::DataSet& operator<<(H5::DataSet& ds, const SequelMovieFrame<float>& value);

const H5::DataSet& operator>>(const H5::DataSet& ds, SequelMovieFrame<float>& value);

}}

#endif //PACBIO_SEQUEL_COMMON_SEQUELMOVIEFRAME_H
