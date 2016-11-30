/* adler32.c -- compute the Adler-32 checksum of a data stream
 * Copyright (C) 1995-2011 Mark Adler
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

/* @(#) $Id$ */

#include "zutil.h"
#include <immintrin.h>

#define local static

#define BASE 65521      /* largest prime smaller than 65536 */
//#define NMAX 5552
#define NMAX 5504 /* we removed 48 to be divisible by 64 */
/* NMAX is the largest n such that 255n(n+1)/2 + (n+1)(BASE-1) <= 2^32-1 */

#define MOD(a) a %= BASE
#define MOD28(a) a %= BASE
#define MOD63(a) a %= BASE

/* ========================================================================= */
ZLIB_INTERNAL uLong adler32_avx2(iadler, buf, len)
    uLong iadler;
    const Bytef *buf;
    uInt len;
{
    unsigned adler = (unsigned)iadler;
    unsigned sum2;
    unsigned n;
    const __m256i c_zero   = _mm256_setzero_si256();
    const __m128i c_sum2hi = _mm_set_epi8( 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16); /* weighs for sum2 addend elements */
    const __m128i c_sum2lo = _mm_set_epi8(17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32); /* weighs for sum2 addend elements */

    const __m256i c_sum2mul = _mm256_set_epi16(NMAX-15, NMAX-14, NMAX-13, NMAX-12, NMAX-11, NMAX-10, NMAX-9, NMAX-8, NMAX-7, NMAX-6, NMAX-5, NMAX-4, NMAX-3, NMAX-2, NMAX-1, NMAX);

    const __m256i c_sum2add = _mm256_set1_epi16(16);

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
	/* AVX2 processors shall have fast unaligned load */
    /* don't align buf */

    /* do length NMAX blocks -- requires just one modulo operation */
    while ( len >= NMAX ) {
        __m256i adler_acc = _mm256_setzero_si256();
        __m256i sum2_acc  = _mm256_setzero_si256();
        __m256i sum2mul   = _mm256_add_epi16(c_sum2mul, c_sum2add);

        len -= NMAX;

        n = NMAX / 64;          /* NMAX is divisible by 64 */
        do {
            __m256i srclo_0, srclo_1;
            __m256i adlerlo_0, adlerlo_1;
            __m256i sum2lo_0, sum2lo_1, sum2lo_2, sum2lo_3;
            __m256i sum2mul_0, sum2mul_1, sum2mul_2;

            srclo_0   = _mm256_loadu_si256((const __m256i*)(buf +  0));
            srclo_1   = _mm256_loadu_si256((const __m256i*)(buf + 32));

            sum2mul_0 = _mm256_sub_epi16(sum2mul, c_sum2add);
            sum2mul_1 = _mm256_sub_epi16(sum2mul_0, c_sum2add);
            sum2mul_2 = _mm256_sub_epi16(sum2mul_1, c_sum2add);
            sum2mul   = _mm256_sub_epi16(sum2mul_2, c_sum2add);

            adlerlo_0 = _mm256_sad_epu8(srclo_0, c_zero); /* compute sums of elements in upper / lower qwords */
            adlerlo_1 = _mm256_sad_epu8(srclo_1, c_zero); /* compute sums of elements in upper / lower qwords */

            sum2lo_0 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(srclo_0));
            sum2lo_1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(srclo_0, 1));
            sum2lo_2 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(srclo_1));
            sum2lo_3 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(srclo_1, 1));
            sum2lo_0 = _mm256_madd_epi16(sum2lo_0, sum2mul_0);
            sum2lo_1 = _mm256_madd_epi16(sum2lo_1, sum2mul_1);
            sum2lo_2 = _mm256_madd_epi16(sum2lo_2, sum2mul_2);
            sum2lo_3 = _mm256_madd_epi16(sum2lo_3, sum2mul);

            adlerlo_0 = _mm256_add_epi32(adlerlo_0, adlerlo_1);
            sum2lo_0  = _mm256_add_epi32(sum2lo_0, sum2lo_1);
            sum2lo_2  = _mm256_add_epi32(sum2lo_2, sum2lo_3);
            sum2lo_0  = _mm256_add_epi32(sum2lo_0, sum2lo_2);
            adler_acc = _mm256_add_epi32(adler_acc, adlerlo_0);
            sum2_acc  = _mm256_add_epi32(sum2_acc, sum2lo_0);

            buf += 64;
        } while (--n);

        /* reduce */
        {
            __m128i adlerlo;
            __m128i sum2lo, sum2hi;

            adlerlo = _mm_add_epi32(_mm256_castsi256_si128(adler_acc), _mm256_extracti128_si256(adler_acc, 1)); /* reduce to 2*32bits */
            adlerlo = _mm_add_epi32(adlerlo, _mm_srli_si128(adlerlo, 8)); /* reduce adler addend to 1*32bits */

            sum2lo = _mm_add_epi32(_mm256_castsi256_si128(sum2_acc), _mm256_extracti128_si256(sum2_acc, 1)); /* reduce to 4*32bits */
            sum2hi = _mm_srli_si128(sum2lo, 8);
            sum2lo = _mm_add_epi32(sum2lo, sum2hi); /* reduce to 2*32bits */
            sum2hi = _mm_srli_si128(sum2lo, 4);
            sum2lo = _mm_add_epi32(sum2lo, sum2hi); /* reduce to 1*32bits */

            sum2  += NMAX * adler + (unsigned)_mm_cvtsi128_si32(sum2lo);
            adler += (unsigned)_mm_cvtsi128_si32(adlerlo); /* Update adler sum */
        }

        MOD(adler);
        MOD(sum2);
    }

    /* do remaining bytes (less than NMAX, still just one modulo) */
    if (len) {                  /* avoid modulos if none remaining */
        if (len >= 64) {
            __m256i adler_acc = _mm256_setzero_si256();
            __m256i sum2_acc  = _mm256_setzero_si256();
            __m256i sum2mul   = _mm256_add_epi16(c_sum2mul, c_sum2add);
            uInt len64 = len & ~(uInt)63U;

            sum2mul = _mm256_sub_epi16(sum2mul, _mm256_set1_epi16(NMAX - len64));

            do {
                __m256i srclo_0, srclo_1;
                __m256i adlerlo_0, adlerlo_1;
                __m256i sum2lo_0, sum2lo_1, sum2lo_2, sum2lo_3;
                __m256i sum2mul_0, sum2mul_1, sum2mul_2;

                srclo_0   = _mm256_loadu_si256((const __m256i*)(buf +  0));
                srclo_1   = _mm256_loadu_si256((const __m256i*)(buf + 32));

                sum2mul_0 = _mm256_sub_epi16(sum2mul, c_sum2add);
                sum2mul_1 = _mm256_sub_epi16(sum2mul_0, c_sum2add);
                sum2mul_2 = _mm256_sub_epi16(sum2mul_1, c_sum2add);
                sum2mul   = _mm256_sub_epi16(sum2mul_2, c_sum2add);

                adlerlo_0 = _mm256_sad_epu8(srclo_0, c_zero); /* compute sums of elements in upper / lower qwords */
                adlerlo_1 = _mm256_sad_epu8(srclo_1, c_zero); /* compute sums of elements in upper / lower qwords */

                sum2lo_0 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(srclo_0));
                sum2lo_1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(srclo_0, 1));
                sum2lo_2 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(srclo_1));
                sum2lo_3 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(srclo_1, 1));
                sum2lo_0 = _mm256_madd_epi16(sum2lo_0, sum2mul_0);
                sum2lo_1 = _mm256_madd_epi16(sum2lo_1, sum2mul_1);
                sum2lo_2 = _mm256_madd_epi16(sum2lo_2, sum2mul_2);
                sum2lo_3 = _mm256_madd_epi16(sum2lo_3, sum2mul);

                adlerlo_0 = _mm256_add_epi32(adlerlo_0, adlerlo_1);
                sum2lo_0  = _mm256_add_epi32(sum2lo_0, sum2lo_1);
                sum2lo_2  = _mm256_add_epi32(sum2lo_2, sum2lo_3);
                sum2lo_0  = _mm256_add_epi32(sum2lo_0, sum2lo_2);
                adler_acc = _mm256_add_epi32(adler_acc, adlerlo_0);
                sum2_acc  = _mm256_add_epi32(sum2_acc, sum2lo_0);

                len -= 64;
                buf += 64;
            } while (len >= 64);
            /* reduce */
            {
                __m128i adlerlo;
                __m128i sum2lo, sum2hi;

                adlerlo = _mm_add_epi32(_mm256_castsi256_si128(adler_acc), _mm256_extracti128_si256(adler_acc, 1)); /* reduce to 2*32bits */
                adlerlo = _mm_add_epi32(adlerlo, _mm_srli_si128(adlerlo, 8)); /* reduce adler addend to 1*32bits */

                sum2lo = _mm_add_epi32(_mm256_castsi256_si128(sum2_acc), _mm256_extracti128_si256(sum2_acc, 1)); /* reduce to 4*32bits */
                sum2hi = _mm_srli_si128(sum2lo, 8);
                sum2lo = _mm_add_epi32(sum2lo, sum2hi); /* reduce to 2*32bits */
                sum2hi = _mm_srli_si128(sum2lo, 4);
                sum2lo = _mm_add_epi32(sum2lo, sum2hi); /* reduce to 1*32bits */

                sum2  += len64 * adler + (unsigned)_mm_cvtsi128_si32(sum2lo);
                adler += (unsigned)_mm_cvtsi128_si32(adlerlo); /* Update adler sum */
            }
        }
        if (len & 32U) {
            __m128i srclo_0, srchi_0;
            __m128i adlerlo_0, adlerhi_0;
            __m128i sum2lo_0, sum2hi_0;

            srclo_0   = _mm_loadu_si128((const __m128i*)(buf +  0));
            srchi_0   = _mm_loadu_si128((const __m128i*)(buf + 16));
            adlerlo_0 = _mm_sad_epu8(srclo_0, _mm256_castsi256_si128(c_zero)); /* compute sums of elements in upper / lower qwords */
            adlerhi_0 = _mm_sad_epu8(srchi_0, _mm256_castsi256_si128(c_zero)); /* compute sums of elements in upper / lower qwords */
            sum2lo_0  = _mm_maddubs_epi16(srclo_0, c_sum2lo); /* compute weighs of each element in sum2 addend */
            sum2hi_0  = _mm_maddubs_epi16(srchi_0, c_sum2hi); /* compute weighs of each element in sum2 addend */

            adlerlo_0 = _mm_add_epi16(adlerlo_0, adlerhi_0);
            adlerhi_0 = _mm_srli_si128(adlerlo_0, 8); /* get hi part of adler addend in the low qword */
            adlerlo_0 = _mm_add_epi16(adlerlo_0, adlerhi_0); /* reduce adler addend to 1*16bits */

            sum2lo_0  = _mm_add_epi16(sum2lo_0, sum2hi_0); /* start reducing sum2 addend */

            sum2hi_0  = _mm_unpackhi_epi16(sum2lo_0, _mm256_castsi256_si128(c_zero));
            sum2lo_0  = _mm_unpacklo_epi16(sum2lo_0, _mm256_castsi256_si128(c_zero));
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
        if (len & 16U) {
            len -= 16;
            __m128i srclo_0;
            __m128i adlerlo_0, adlerhi_0;
            __m128i sum2lo_0, sum2hi_0;

            srclo_0   = _mm_loadu_si128((const __m128i*)(buf +  0));
            adlerlo_0 = _mm_sad_epu8(srclo_0, _mm256_castsi256_si128(c_zero)); /* compute sums of elements in upper / lower qwords */
            sum2lo_0  = _mm_maddubs_epi16(srclo_0, c_sum2hi); /* compute weighs of each element in sum2 addend */

            adlerhi_0 = _mm_srli_si128(adlerlo_0, 8); /* get hi part of adler addend in the low qword */
            adlerlo_0 = _mm_add_epi16(adlerlo_0, adlerhi_0); /* reduce adler addend to 1*16bits */

            sum2hi_0  = _mm_unpackhi_epi16(sum2lo_0, _mm256_castsi256_si128(c_zero));
            sum2lo_0  = _mm_unpacklo_epi16(sum2lo_0, _mm256_castsi256_si128(c_zero));
            sum2lo_0  = _mm_add_epi32(sum2lo_0, sum2hi_0); /* reduce to 4*32bits */
            sum2hi_0 = _mm_srli_si128(sum2lo_0, 8);
            sum2lo_0 = _mm_add_epi32(sum2lo_0, sum2hi_0); /* reduce to 2*32bits */
            sum2hi_0 = _mm_srli_si128(sum2lo_0, 4);
            sum2lo_0 = _mm_add_epi32(sum2lo_0, sum2hi_0); /* reduce to 1*32bits */

            sum2  += 16 * adler + (unsigned)_mm_cvtsi128_si32(sum2lo_0);
            adler += (unsigned)_mm_cvtsi128_si32(adlerlo_0); /* Update adler sum */

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
