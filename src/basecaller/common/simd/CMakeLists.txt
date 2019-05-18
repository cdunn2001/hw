
cmake_minimum_required(VERSION 3.13)

add_library(libMongoCommonSimd  STATIC
    ArrayUnion.h
    SimdVectorTypes.h
    m32f.h
    m32s.h
    m512b.h
    m512b_LOOP.h
    m512b_SSE.h
    m512b_AVX512.h
    m512i.h
    m512i_LOOP.h
    m512i_SSE.h
    m512i_AVX512.h
    m512f.h
    m512f_LOOP.h
    m512f_SSE.h
    m512f_AVX512.h
    m512s.h
    m512s_LOOP.h
    m512s_SSE.h
    m512s_AVX512.h
    SimdConvTraits.h
    SimdTypeTraits.h
    xcompile.h
    sse_mathfun.h
    mm_blendv_si128.h
    xcompile.cpp
)

target_include_directories(libMongoCommonSimd
    PUBLIC
        ${MONGO_PROJECT_DIR}
)
