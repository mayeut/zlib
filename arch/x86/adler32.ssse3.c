/* arch/x86/adler32.avx2.c -- compute the Adler-32 checksum of a data stream using AVX2 SIMD extensions
 * Copyright (C) 1995-2011 Mark Adler
 * Copyright (C) 2016 Matthieu Darbois
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

/* @(#) $Id$ */

#include "zutil.h"
#include <immintrin.h>

#define BASE 65521      /* largest prime smaller than 65536 */
#define NMAX 5552
/* NMAX is the largest n such that 255n(n+1)/2 + (n+1)(BASE-1) <= 2^32-1 */

#define DO1(buf,i)  {adler += (buf)[i]; sum2 += adler;}
#define DO2(buf,i)  DO1(buf,i); DO1(buf,i+1);
#define DO4(buf,i)  DO2(buf,i); DO2(buf,i+2);
#define DO8(buf,i)  DO4(buf,i); DO4(buf,i+4);
#define DO16(buf)   DO8(buf,0); DO8(buf,8);

#define MOD(a) a %= BASE
#define MOD28(a) a %= BASE
#define MOD63(a) a %= BASE


/* ========================================================================= */
ZLIB_INTERNAL uLong adler32_ssse3(iadler, buf, len)
    uLong iadler;
    const Bytef *buf;
    uInt len;
{
    unsigned adler = (unsigned)iadler;
    unsigned sum2;
    unsigned n;
    const __m128i c_zero   = _mm_setzero_si128();
    const __m128i c_sum2hi = _mm_set_epi8( 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16); /* weighs for sum2 addend elements */
    const __m128i c_sum2lo = _mm_set_epi8(17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32); /* weighs for sum2 addend elements */

    /* split Adler-32 into component sums */
    sum2 = (adler >> 16) & 0xffffU;
    adler &= 0xffffU;

    /* in case user likes doing a byte at a time, keep it fast */
    if (len == 1) {
        adler += buf[0];
        if (adler >= BASE)
            adler -= BASE;
        sum2 += adler;
        if (sum2 >= BASE)
            sum2 -= BASE;
        return adler | (sum2 << 16);
    }

    /* initial Adler-32 value (deferred check for len == 1 speed) */
    if (buf == Z_NULL)
        return 1L;

    /* in case short lengths are provided, keep it somewhat fast */
    if (len < 16) {
        while (len--) {
            adler += *buf++;
            sum2 += adler;
        }
        if (adler >= BASE)
            adler -= BASE;
        MOD28(sum2);            /* only added so many BASE's */
        return adler | (sum2 << 16);
    }

    /* do length NMAX blocks -- requires just one modulo operation */
    while (len >= NMAX) {
        len -= NMAX;
        n = NMAX / 32;          /* NMAX is divisible by 16 */

        do {
            __m128i srclo_0, srchi_0;
            __m128i adlerlo_0, adlerhi_0;
            __m128i sum2lo_0, sum2hi_0;

            srclo_0   = _mm_loadu_si128((const __m128i*)(buf +  0));
            srchi_0   = _mm_loadu_si128((const __m128i*)(buf + 16));
            adlerlo_0 = _mm_sad_epu8(srclo_0, c_zero); /* compute sums of elements in upper / lower qwords */
            adlerhi_0 = _mm_sad_epu8(srchi_0, c_zero); /* compute sums of elements in upper / lower qwords */
            sum2lo_0  = _mm_maddubs_epi16(srclo_0, c_sum2lo); /* compute weighs of each element in sum2 addend */
            sum2hi_0  = _mm_maddubs_epi16(srchi_0, c_sum2hi); /* compute weighs of each element in sum2 addend */

            adlerlo_0 = _mm_add_epi16(adlerlo_0, adlerhi_0);
            adlerhi_0 = _mm_srli_si128(adlerlo_0, 8); /* get hi part of adler addend in the low qword */
            adlerlo_0 = _mm_add_epi16(adlerlo_0, adlerhi_0); /* reduce adler addend to 1*16bits */

            sum2lo_0  = _mm_add_epi16(sum2lo_0, sum2hi_0); /* start reducing sum2 addend */

            sum2hi_0  = _mm_unpackhi_epi16(sum2lo_0, c_zero);
            sum2lo_0  = _mm_unpacklo_epi16(sum2lo_0, c_zero);
            sum2lo_0  = _mm_add_epi32(sum2lo_0, sum2hi_0); /* reduce to 4*32bits */
            sum2hi_0  = _mm_srli_si128(sum2lo_0, 8);
            sum2lo_0  = _mm_add_epi32(sum2lo_0, sum2hi_0); /* reduce to 2*32bits */
            sum2hi_0  = _mm_srli_si128(sum2lo_0, 4);
            sum2lo_0  = _mm_add_epi32(sum2lo_0, sum2hi_0); /* reduce to 1*32bits */

            sum2  += 32 * adler + (unsigned)_mm_cvtsi128_si32(sum2lo_0);
            adler += (unsigned)_mm_cvtsi128_si32(adlerlo_0); /* Update adler sum */
            
            buf += 32;
        } while (--n);

        if ((NMAX / 16U) & 1U) {
            __m128i srclo_0;
            __m128i adlerlo_0, adlerhi_0;
            __m128i sum2lo_0, sum2hi_0;

            srclo_0   = _mm_loadu_si128((const __m128i*)(buf +  0));
            adlerlo_0 = _mm_sad_epu8(srclo_0, c_zero); /* compute sums of elements in upper / lower qwords */
            sum2lo_0  = _mm_maddubs_epi16(srclo_0, c_sum2hi); /* compute weighs of each element in sum2 addend */

            adlerhi_0 = _mm_srli_si128(adlerlo_0, 8); /* get hi part of adler addend in the low qword */
            adlerlo_0 = _mm_add_epi16(adlerlo_0, adlerhi_0); /* reduce adler addend to 1*16bits */

            sum2hi_0  = _mm_unpackhi_epi16(sum2lo_0, c_zero);
            sum2lo_0  = _mm_unpacklo_epi16(sum2lo_0, c_zero);
            sum2lo_0  = _mm_add_epi32(sum2lo_0, sum2hi_0); /* reduce to 4*32bits */
            sum2hi_0 = _mm_srli_si128(sum2lo_0, 8);
            sum2lo_0 = _mm_add_epi32(sum2lo_0, sum2hi_0); /* reduce to 2*32bits */
            sum2hi_0 = _mm_srli_si128(sum2lo_0, 4);
            sum2lo_0 = _mm_add_epi32(sum2lo_0, sum2hi_0); /* reduce to 1*32bits */

            sum2  += 16 * adler + (unsigned)_mm_cvtsi128_si32(sum2lo_0);
            adler += (unsigned)_mm_cvtsi128_si32(adlerlo_0); /* Update adler sum */
            buf += 16;
        }

        MOD(adler);
        MOD(sum2);
    }

    /* do remaining bytes (less than NMAX, still just one modulo) */
    if (len) {                  /* avoid modulos if none remaining */
        while (len >= 32) {
            __m128i srclo_0, srchi_0;
            __m128i adlerlo_0, adlerhi_0;
            __m128i sum2lo_0, sum2hi_0;

            srclo_0   = _mm_loadu_si128((const __m128i*)(buf +  0));
            srchi_0   = _mm_loadu_si128((const __m128i*)(buf + 16));
            adlerlo_0 = _mm_sad_epu8(srclo_0, c_zero); /* compute sums of elements in upper / lower qwords */
            adlerhi_0 = _mm_sad_epu8(srchi_0, c_zero); /* compute sums of elements in upper / lower qwords */
            sum2lo_0  = _mm_maddubs_epi16(srclo_0, c_sum2lo); /* compute weighs of each element in sum2 addend */
            sum2hi_0  = _mm_maddubs_epi16(srchi_0, c_sum2hi); /* compute weighs of each element in sum2 addend */

            adlerlo_0 = _mm_add_epi16(adlerlo_0, adlerhi_0);
            adlerhi_0 = _mm_srli_si128(adlerlo_0, 8); /* get hi part of adler addend in the low qword */
            adlerlo_0 = _mm_add_epi16(adlerlo_0, adlerhi_0); /* reduce adler addend to 1*16bits */

            sum2lo_0  = _mm_add_epi16(sum2lo_0, sum2hi_0); /* start reducing sum2 addend */

            sum2hi_0  = _mm_unpackhi_epi16(sum2lo_0, c_zero);
            sum2lo_0  = _mm_unpacklo_epi16(sum2lo_0, c_zero);
            sum2lo_0  = _mm_add_epi32(sum2lo_0, sum2hi_0); /* reduce to 4*32bits */
            sum2hi_0  = _mm_srli_si128(sum2lo_0, 8);
            sum2lo_0  = _mm_add_epi32(sum2lo_0, sum2hi_0); /* reduce to 2*32bits */
            sum2hi_0  = _mm_srli_si128(sum2lo_0, 4);
            sum2lo_0  = _mm_add_epi32(sum2lo_0, sum2hi_0); /* reduce to 1*32bits */

            sum2  += 32 * adler + (unsigned)_mm_cvtsi128_si32(sum2lo_0);
            adler += (unsigned)_mm_cvtsi128_si32(adlerlo_0); /* Update adler sum */

            len -= 32;
            buf += 32;
        }
        if (len >= 16) {
            __m128i srclo_0;
            __m128i adlerlo_0, adlerhi_0;
            __m128i sum2lo_0, sum2hi_0;

            srclo_0   = _mm_loadu_si128((const __m128i*)(buf +  0));
            adlerlo_0 = _mm_sad_epu8(srclo_0, c_zero); /* compute sums of elements in upper / lower qwords */
            sum2lo_0  = _mm_maddubs_epi16(srclo_0, c_sum2hi); /* compute weighs of each element in sum2 addend */

            adlerhi_0 = _mm_srli_si128(adlerlo_0, 8); /* get hi part of adler addend in the low qword */
            adlerlo_0 = _mm_add_epi16(adlerlo_0, adlerhi_0); /* reduce adler addend to 1*16bits */

            sum2hi_0  = _mm_unpackhi_epi16(sum2lo_0, c_zero);
            sum2lo_0  = _mm_unpacklo_epi16(sum2lo_0, c_zero);
            sum2lo_0  = _mm_add_epi32(sum2lo_0, sum2hi_0); /* reduce to 4*32bits */
            sum2hi_0 = _mm_srli_si128(sum2lo_0, 8);
            sum2lo_0 = _mm_add_epi32(sum2lo_0, sum2hi_0); /* reduce to 2*32bits */
            sum2hi_0 = _mm_srli_si128(sum2lo_0, 4);
            sum2lo_0 = _mm_add_epi32(sum2lo_0, sum2hi_0); /* reduce to 1*32bits */

            sum2  += 16 * adler + (unsigned)_mm_cvtsi128_si32(sum2lo_0);
            adler += (unsigned)_mm_cvtsi128_si32(adlerlo_0); /* Update adler sum */

            len -= 16;
            buf += 16;
        }
        while (len--) {
            adler += *buf++;
            sum2 += adler;
        }
        MOD(adler);
        MOD(sum2);
    }

    /* return recombined sums */
    return adler | (sum2 << 16);
}
