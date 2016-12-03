/* arch/x86/dispatch.c -- dispatch x86 optimized functions
 * Copyright (C) 2016 Matthieu Darbois
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

/* @(#) $Id$ */

#include "zutil.h"
#include "deflate.h"

#include <cpuid.h>

/* flags definitions */
#define ZL_INITIALIZED 0x0001U
#define ZL_SSE2        0x0002U
#define ZL_SSSE3       0x0004U
#define ZL_SSE41       0x0008U
#define ZL_SSE42       0x0010U
#define ZL_CLMUL       0x0020U
#define ZL_AVX2        0x0040U

/* Features in ecx for level 1 */
#define ZL_BIT_CLMUL   0x00000002U
#define ZL_BIT_SSSE3   0x00000200U
#define ZL_BIT_SSE41   0x00080000U
#define ZL_BIT_SSE42   0x00100000U
#define ZL_BIT_XSAVE   0x04000000U
#define ZL_BIT_OSXSAVE 0x08000000U

/* Features in edx for level 1 */
#define ZL_BIT_SSE2    0x04000000U

/* Features in ebx for level 7 subleaf 0 */
#define ZL_BIT_AVX2    0x00000020U

/* function prototypes */
local int zl_cpuid_max OF(());
local void zl_cpuid OF((unsigned int level, unsigned int subleaf,
                        unsigned int *eax, unsigned int *ebx,
                        unsigned int *ecx, unsigned int *edx));
local void zl_xgetbv OF((unsigned int index,
                         unsigned int *eax, unsigned int *edx));

local unsigned int zl_get_feature_flags_once OF(());
local unsigned int zl_get_feature_flags OF(());
local volatile unsigned int zl_feature_flags = 0U;

local uLong zl_adler32_dispatch_init      OF((uLong adler, const Bytef *buf, uInt len));
local uLong zl_adler32_copy_dispatch_init OF((uLong adler, const Bytef *buf, uInt len, Bytef *dest));
local uLong zl_crc32_dispatch_init        OF((uLong crc,   const Bytef *buf, uInt len));
local uLong zl_crc32_copy_dispatch_init   OF((uLong crc, const Bytef *buf, uInt len, Bytef *dest));
local void  zl_fill_window_dispatch_init  OF((deflate_state *s));

typedef uLong (*zl_adler32_func)      OF((uLong adler, const Bytef *buf, uInt len));
typedef uLong (*zl_adler32_copy_func) OF((uLong adler, const Bytef *buf, uInt len, Bytef *dest));
typedef uLong (*zl_crc32_func)        OF((uLong crc,   const Bytef *buf, uInt len));
typedef uLong (*zl_crc32_copy_func)   OF((uLong crc, const Bytef *buf, uInt len, Bytef *dest));
typedef void  (*zl_fill_window_func)  OF((deflate_state *s));

local volatile zl_adler32_func      zl_adler32_dispatch      = zl_adler32_dispatch_init;
local volatile zl_adler32_copy_func zl_adler32_copy_dispatch = zl_adler32_copy_dispatch_init;
local volatile zl_crc32_func        zl_crc32_dispatch        = zl_crc32_dispatch_init;
local volatile zl_crc32_copy_func   zl_crc32_copy_dispatch   = zl_crc32_copy_dispatch_init;
local volatile zl_fill_window_func  zl_fill_window_dispatch  = zl_fill_window_dispatch_init;

local int zl_cpuid_max()
{
    return __get_cpuid_max(0, NULL);
}

local void zl_cpuid (level, subleaf, eax, ebx, ecx, edx)
    unsigned int level;
    unsigned int subleaf;
    unsigned int *eax;
    unsigned int *ebx;
    unsigned int *ecx;
    unsigned int *edx;
{
    __cpuid_count(level, subleaf, *eax, *ebx, *ecx, *edx);
}

local void zl_xgetbv(index, eax, edx)
    unsigned int index;
    unsigned int *eax;
    unsigned int *edx;
{
    __asm__ __volatile__("xgetbv" : "=a"(*eax), "=d"(*edx) : "c"(index));
}

local unsigned int zl_get_feature_flags_once()
{
    unsigned int result = ZL_INITIALIZED;
    int maxLevel = zl_cpuid_max();

    if (maxLevel) {
        unsigned int avxEnabled = 0U;
        unsigned int eax;
        unsigned int ebx;
        unsigned int ecx;
        unsigned int edx;

        zl_cpuid(1U, 0U, &eax, &ebx, &ecx, &edx);

        if (edx & ZL_BIT_SSE2) {
            result |= ZL_SSE2;
        }
        if (ecx & ZL_BIT_SSSE3) {
            result |= ZL_SSSE3;
        }
        if (ecx & ZL_BIT_SSE41) {
            result |= ZL_SSE41;
        }
        if (ecx & ZL_BIT_SSE42) {
            result |= ZL_SSE42;
        }
        if (ecx & ZL_BIT_CLMUL) {
            result |= ZL_CLMUL;
        }

        /* for AVX / AVX2, need to check support support from OS & support for xgetbv */
        if ((ecx & (ZL_BIT_XSAVE | ZL_BIT_OSXSAVE)) == (ZL_BIT_XSAVE | ZL_BIT_OSXSAVE)) {
            zl_xgetbv(0, &eax, &edx);
            /* check XMM/YMM register are saved / restored properly */
            if ((eax & (2U | 4U)) == (2U | 4U)) {
                avxEnabled = 1U;
            }
        }

        if ((maxLevel >= 7U) && avxEnabled) {
            zl_cpuid(7U, 0U, &eax, &ebx, &ecx, &edx);

            if (ebx & ZL_BIT_AVX2) {
                result |= ZL_AVX2;
            }
        }
    }
    return result;
}

local unsigned int zl_get_feature_flags()
{
    /* TODO atomic read */
    unsigned int result = zl_feature_flags;

    if (result == 0U) {
        result = zl_get_feature_flags_once();
        /* TODO atomic operation */
        zl_feature_flags = result;
    }

    return result;
}

ZLIB_INTERNAL uLong adler32_generic OF((uLong adler, const Bytef *buf, uInt len));
ZLIB_INTERNAL uLong adler32_sse2    OF((uLong adler, const Bytef *buf, uInt len));
ZLIB_INTERNAL uLong adler32_avx2    OF((uLong adler, const Bytef *buf, uInt len));

local uLong zl_adler32_dispatch_init(adler, buf, len)
    uLong adler;
    const Bytef *buf;
    uInt len;
{
    zl_adler32_func function = adler32_generic;
    unsigned int features = zl_get_feature_flags();

    if (features & ZL_AVX2) {
        function = adler32_avx2;
    } else if (features & ZL_SSE2) {
        function = adler32_sse2;
    }
    /* TODO atomic */
    zl_adler32_dispatch = function;
    return function(adler, buf, len);
}
ZLIB_INTERNAL uLong adler32_dispatch(adler, buf, len)
    uLong adler;
    const Bytef *buf;
    uInt len;
{
    /* TODO atomic read */
    return zl_adler32_dispatch(adler, buf, len);
}

ZLIB_INTERNAL uLong adler32_copy_generic OF((uLong adler, const Bytef *buf, uInt len, Bytef *dest));
ZLIB_INTERNAL uLong adler32_copy_sse2    OF((uLong adler, const Bytef *buf, uInt len, Bytef *dest));
ZLIB_INTERNAL uLong adler32_copy_avx2    OF((uLong adler, const Bytef *buf, uInt len, Bytef *dest));

local uLong zl_adler32_copy_dispatch_init(adler, buf, len, dest)
    uLong adler;
    const Bytef *buf;
    uInt len;
    Bytef* dest;
{
    zl_adler32_copy_func function = adler32_copy_generic;
    unsigned int features = zl_get_feature_flags();

    if (features & ZL_AVX2) {
        function = adler32_copy_avx2;
    } else if (features & ZL_SSE2) {
        function = adler32_copy_sse2;
    }
    /* TODO atomic */
    zl_adler32_copy_dispatch = function;
    return function(adler, buf, len, dest);
}
ZLIB_INTERNAL uLong adler32_copy_dispatch(adler, buf, len, dest)
    uLong adler;
    const Bytef *buf;
    uInt len;
    Bytef *dest;
{
    /* TODO atomic read */
    return zl_adler32_copy_dispatch(adler, buf, len, dest);
}

ZLIB_INTERNAL uLong crc32_generic OF((uLong crc, const Bytef *buf, uInt len));
ZLIB_INTERNAL uLong crc32_pclmulqdq OF((uLong crc, const Bytef *buf, uInt len));

local uLong zl_crc32_dispatch_init(crc, buf, len)
    uLong crc;
    const Bytef *buf;
    uInt len;
{
    zl_crc32_func function = crc32_generic;
    unsigned int features = zl_get_feature_flags();

    if (features & ZL_CLMUL) {
        function = crc32_pclmulqdq;
    }

    /* TODO atomic */
    zl_crc32_dispatch = function;
    return function(crc, buf, len);
}

ZLIB_INTERNAL uLong crc32_dispatch(adler, buf, len)
    uLong adler;
    const Bytef *buf;
    uInt len;
{
    /* TODO atomic read */
    return zl_crc32_dispatch(adler, buf, len);
}

ZLIB_INTERNAL uLong crc32_copy_generic OF((uLong crc, const Bytef *buf, uInt len, Bytef *dest));
ZLIB_INTERNAL uLong crc32_copy_pclmulqdq OF((uLong crc, const Bytef *buf, uInt len, Bytef *dest));

local uLong zl_crc32_copy_dispatch_init(crc, buf, len, dest)
    uLong crc;
    const Bytef *buf;
    uInt len;
    Bytef *dest;
{
    zl_crc32_copy_func function = crc32_copy_generic;
    unsigned int features = zl_get_feature_flags();

    if (features & ZL_CLMUL) {
        function = crc32_copy_pclmulqdq;
    }

    /* TODO atomic */
    zl_crc32_copy_dispatch = function;
    return function(crc, buf, len, dest);
}

ZLIB_INTERNAL uLong crc32_copy_dispatch(adler, buf, len, dest)
    uLong adler;
    const Bytef *buf;
    uInt len;
    Bytef *dest;
{
    /* TODO atomic read */
    return zl_crc32_copy_dispatch(adler, buf, len, dest);
}

ZLIB_INTERNAL void fill_window_generic OF((deflate_state *s));
ZLIB_INTERNAL void fill_window_sse2 OF((deflate_state *s));

local void zl_fill_window_dispatch_init(s)
    deflate_state *s;
{
    zl_fill_window_func function = fill_window_generic;
    unsigned int features = zl_get_feature_flags();

    if (features & ZL_SSE2) {
        function = fill_window_sse2;
    }

    /* TODO atomic */
    zl_fill_window_dispatch = function;
    return function(s);
}

ZLIB_INTERNAL void fill_window_dispatch(s)
    deflate_state *s;
{
    /* TODO atomic read */
    zl_fill_window_dispatch(s);
}
