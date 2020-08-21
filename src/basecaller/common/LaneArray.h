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

// This file is the entry point for using the LaneArray concept, though
// all code is defined elsewhere.  LaneArray provides a generic
// template class that can treat a fixed-length array of arithmetic
// types that behave mathematically the same as their scalar counterparts.
// These classes are vectorized, and should have their best performance on
// an AVX512 compilation.  An SSE implementations exist, but it has a few
// software emulations for missing intrinsics, plus it's fundamentally
// limited in that it takes 4 registers to do what AVX512 can do with
// one, which for some code will cause more pushing/popping data to/from
// the stack.
//
// For the most part these types will behave as closely as possible to
// their scalar counterparts, including respecting concepts like
// "common type" and "integral promotions".  In simple terms this means
// you get expected behaviour where:
// LaneArray<int> + LaneArray<float> = LaneArray<float>.
//
// However there are a few notible exceptions worth being aware of:
// 1. LaneArray<short> + LaneArray<short> = LaneArray<short>
//    Normal 16 bit types get promoted to 32 bit in such a case but
//    we very much don't want that.
// 2. While LaneArray<short> * LaneArray<int> will result in
//    LaneArray<int>, LaneArray<short> * int will *not*.  This
//    is because it's very difficult to type 16 bit literals,
//    and even if you have a 16 bit variable it's too easy to
//    accidentally bump it up to 32 bit.  Without this exception
//    we'll be either be constantly casting back to 16 bit types
//    or else suffering from unecessary (and somewhat expensive)
//    conversion to arrays of 32 bit types.
//    2b. Scalar int and unsigned int are the *only* exceptions
//        to this rule.  LaneArray<int> * float will still result
//        in LaneArray<float>
// 3. No implicit "demotions".  You can't do something like
//    LaneArray<int> += float as that requires truncation.  Maybe
//    this is my personal preferences bleeding through but enabling
//    that did not seem a good idea even if primive types can
//    behave that way.  If someone decides this should be changed,
//    one should only have to tweak the definition of
//    `CompoundOpIsValid` in `LaneArrayTraits.h`
// 4. All mixed sign integral operations are disabled.  The compiler
//    can warn you if you do silly things like signed/unsigned
//    comparison, but that seemed difficult for me to emulate.
//    So those operators are all undefined, and if a user wants
//    to do something mixed type they will have to explicitly
//    cast one of their arguments.  If someone decides this should
//    be changed, one should only have to tweak the definition of
//    `CommonArrayType` in `LaneArrayTraits.h`
//
// I've tried to document the implementation thouroughly, but here's
// a big picture overview about how all the pieces fit together:
// There are 4 main classes, all with separate responsibilities
// 1. BaseArray: A CRTP base class at the top of the inheritance
//    chain.  It's responsiblity is the storage/construction/conversion
//    of the raw m512 array used.  By providing this functionality in
//    a central place it avoids a lot of duplicate logic that would
//    need to be in the various LaneArray types, though it comes at
//    the cost of being rather abstract.
//
//    This class provides a special constructor and two special functions
//    (Update and Reduce) that enable a lot of functionality in children
//    classes.  These functions follow a common trend where it accepts
//    a lambda function and an arbitrary set of arguments, and it
//    automatically iterates over all the arrays involved, applying the
//    lambda to each element.  This iteration process automatically
//    handles 32bit vs 16bit mismatches (as the natural stride is different
//    for each), as well as scalar arguments mixed in with the array
//    arguments.
//
//    Children classes leverage these functions for all their
//    defined operators/functions, as this allows them to have simple
//    definitions uncomplicated by the bit width and vector/scalar issues
//    mentioned.  In principle however these functions are publicly
//    available, and in some cases it may be more efficient to use
//    these functions directly if doing a complicated mathematical
//    operation.  This could be faster for the same reason that expression
//    templates (e.g. what Eigen uses) can be faster as your temporaries
//    use less register/stack space and access patterns are potentially
//    more cache friendly.  In fact, it should be possible to augment
//    this framework to actually use expression templates itself,
//    but doing so seemed a bit too far "out of scope" at the time
//    this was written.  Still, it's an interesting idea for the future.
//
// 2. LaneMask: This is a relatively simple class that handles arrays of
//    booleans.  It's only real (minor) complication is that as an
//    array of booleans is not bitwise compatible with our m512b types,
//    it must handle some of the creation/conversion logic that otherwise
//    would have been handled in BaseArray.
//
// 3. ArithmeticArray: This class extends BaseArray and is in turn
//    meant to be the parent of other LaneArray classes.  It generically
//    provides the common functionality one would expect from arithmetic
//    types (barring the exceptions mentioned above).  At face value
//    it is very simple, though it does rely on the details of ADL
//    and SFINAE to produce a comprehenive overload set that is seemless
//    to use.  I've tried to document such "magic" thoroughly, but also
//    to write things in such a way that things appear simple and these
//    details can be glossed over unless one cares about them.
//
// 4. LaneArray: This is the main externally visible class.  There are
//    specializations for every supported primitive type, where those
//    specializations can be used to provide type-specific functionality
//    (e.g. isnan).
//    Note: There currently is no support for 64 bit types. (and it would
//          not be trivial to add)

#ifndef mongo_common_LaneArray_H_
#define mongo_common_LaneArray_H_

#include <common/LaneArray_fwd.h>
#include <common/simd/LaneArrayImpl.h>

#endif  // mongo_common_LaneArray_H_
