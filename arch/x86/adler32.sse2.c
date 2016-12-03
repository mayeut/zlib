/* arch/x86/adler32.sse2.c -- compute the Adler-32 checksum of a data stream using SSE2 SIMD extensions
 * Copyright (C) 1995-2011 Mark Adler
 * Copyright (C) 2016 Matthieu Darbois
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

/* @(#) $Id$ */

#include "zutil.h"
#include <emmintrin.h>

#define BASE 65521      /* largest prime smaller than 65536 */

//#define NMAX 5552
#define NMAX 5536 /* we removed 16 to be divisible by 32 */
/* NMAX is the largest n such that 255n(n+1)/2 + (n+1)(BASE-1) <= 2^32-1 */

#define MOD(a) a %= BASE
#define MOD28(a) a %= BASE
#define MOD63(a) a %= BASE

/* ========================================================================= */
ZLIB_INTERNAL uLong adler32_sse2(iadler, buf, len)
    uLong iadler;
    const Bytef *buf;
    uInt len;
{
    unsigned adler = (unsigned)iadler;
    unsigned sum2;
    unsigned n;
    const __m128i c_zero   = _mm_setzero_si128();
    const __m128i c_sum2hi = _mm_set_epi16(1,  2,  3,  4,  5,  6,  7,  8); /* weighs for sum2 addend elements */
    const __m128i c_sum2lo = _mm_set_epi16(9, 10, 11, 12, 13, 14, 15, 16); /* weighs for sum2 addend elements */
    const __m128i c_sum2mul = _mm_set_epi16(NMAX-7, NMAX-6, NMAX-5, NMAX-4, NMAX-3, NMAX-2, NMAX-1, NMAX);
    const __m128i c_sum2add = _mm_set1_epi16(8);

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
    else {
        /* align buf on 16bytes boundary */
        uInt align = (16U - ((uInt)buf & 15U)) & 15U;
        if (align) {
            while (align--) {
                adler += *buf++;
                sum2 += adler;
                len--;
            }
            /* NMAX is in fact NMAX - 16, no need for modulo here */
        }
    }

    /* do length NMAX blocks -- requires just one modulo operation */
    while (len >= NMAX) {
        __m128i adler_acc = _mm_setzero_si128();
        __m128i sum2_acc  = _mm_setzero_si128();
        __m128i sum2mul   = _mm_add_epi16(c_sum2mul, c_sum2add);

        len -= NMAX;
        n = NMAX / 32;          /* NMAX is divisible by 16 */
        do {
            __m128i src_0;
            __m128i adlerlo_0;
            __m128i sum2lo_0, sum2hi_0;
            __m128i src_1;
            __m128i adlerlo_1;
            __m128i sum2lo_1, sum2hi_1;
            __m128i sum2mul_0, sum2mul_1, sum2mul_2;

            src_0     = _mm_load_si128((const __m128i*)(buf +  0));
            src_1     = _mm_load_si128((const __m128i*)(buf + 16));

            adlerlo_0 = _mm_sad_epu8(src_0, c_zero); /* compute sums of elements in upper / lower qwords */
            adlerlo_1 = _mm_sad_epu8(src_1, c_zero); /* compute sums of elements in upper / lower qwords */

            sum2lo_0 = _mm_unpacklo_epi8(src_0, c_zero); /* convert lower 8*8bits to 8*16bits words */
            sum2hi_0 = _mm_unpackhi_epi8(src_0, c_zero); /* convert upper 8*8bits to 8*16bits words */
            sum2lo_1 = _mm_unpacklo_epi8(src_1, c_zero); /* convert lower 8*8bits to 8*16bits words */
            sum2hi_1 = _mm_unpackhi_epi8(src_1, c_zero); /* convert upper 8*8bits to 8*16bits words */

            sum2mul_0 = _mm_sub_epi16(sum2mul, c_sum2add);
            sum2mul_1 = _mm_sub_epi16(sum2mul_0, c_sum2add);
            sum2mul_2 = _mm_sub_epi16(sum2mul_1, c_sum2add);
            sum2mul   = _mm_sub_epi16(sum2mul_2, c_sum2add);

            sum2lo_0 = _mm_madd_epi16(sum2lo_0, sum2mul_0);
            sum2hi_0 = _mm_madd_epi16(sum2hi_0, sum2mul_1);
            sum2lo_1 = _mm_madd_epi16(sum2lo_1, sum2mul_2);
            sum2hi_1 = _mm_madd_epi16(sum2hi_1, sum2mul);

            adlerlo_0 = _mm_add_epi32(adlerlo_0, adlerlo_1);
            sum2lo_0  = _mm_add_epi32(sum2lo_0, sum2hi_0);
            sum2lo_1  = _mm_add_epi32(sum2lo_1, sum2hi_1);
            sum2lo_0  = _mm_add_epi32(sum2lo_0, sum2lo_1);
            adler_acc = _mm_add_epi32(adler_acc, adlerlo_0);
            sum2_acc  = _mm_add_epi32(sum2_acc, sum2lo_0);

            buf += 32;
        } while (--n);

        /* reduce */
        {
            __m128i sum2hi;

            adler_acc = _mm_add_epi32(adler_acc, _mm_srli_si128(adler_acc, 8)); /* reduce adler addend to 1*32bits */

            sum2hi = _mm_srli_si128(sum2_acc, 8);
            sum2_acc = _mm_add_epi32(sum2_acc, sum2hi); /* reduce to 2*32bits */
            sum2hi = _mm_srli_si128(sum2_acc, 4);
            sum2_acc = _mm_add_epi32(sum2_acc, sum2hi); /* reduce to 1*32bits */

            sum2  += NMAX * adler + (unsigned)_mm_cvtsi128_si32(sum2_acc);
            adler += (unsigned)_mm_cvtsi128_si32(adler_acc); /* Update adler sum */
        }

        MOD(adler);
        MOD(sum2);
    }

    /* do remaining bytes (less than NMAX, still just one modulo) */
    if (len) {                  /* avoid modulos if none remaining */
        if (len >= 32) {
            __m128i adler_acc = _mm_setzero_si128();
            __m128i sum2_acc  = _mm_setzero_si128();
            __m128i sum2mul   = _mm_add_epi16(c_sum2mul, c_sum2add);
            uInt len32 = len & ~(uInt)31U;

            sum2mul = _mm_sub_epi16(sum2mul, _mm_set1_epi16(NMAX - len32));

            do {
                __m128i src_0;
                __m128i adlerlo_0;
                __m128i sum2lo_0, sum2hi_0;
                __m128i src_1;
                __m128i adlerlo_1;
                __m128i sum2lo_1, sum2hi_1;
                __m128i sum2mul_0, sum2mul_1, sum2mul_2;

                src_0     = _mm_load_si128((const __m128i*)(buf +  0));
                src_1     = _mm_load_si128((const __m128i*)(buf + 16));

                adlerlo_0 = _mm_sad_epu8(src_0, c_zero); /* compute sums of elements in upper / lower qwords */
                adlerlo_1 = _mm_sad_epu8(src_1, c_zero); /* compute sums of elements in upper / lower qwords */

                sum2lo_0 = _mm_unpacklo_epi8(src_0, c_zero); /* convert lower 8*8bits to 8*16bits words */
                sum2hi_0 = _mm_unpackhi_epi8(src_0, c_zero); /* convert upper 8*8bits to 8*16bits words */
                sum2lo_1 = _mm_unpacklo_epi8(src_1, c_zero); /* convert lower 8*8bits to 8*16bits words */
                sum2hi_1 = _mm_unpackhi_epi8(src_1, c_zero); /* convert upper 8*8bits to 8*16bits words */

                sum2mul_0 = _mm_sub_epi16(sum2mul, c_sum2add);
                sum2mul_1 = _mm_sub_epi16(sum2mul_0, c_sum2add);
                sum2mul_2 = _mm_sub_epi16(sum2mul_1, c_sum2add);
                sum2mul   = _mm_sub_epi16(sum2mul_2, c_sum2add);

                sum2lo_0 = _mm_madd_epi16(sum2lo_0, sum2mul_0);
                sum2hi_0 = _mm_madd_epi16(sum2hi_0, sum2mul_1);
                sum2lo_1 = _mm_madd_epi16(sum2lo_1, sum2mul_2);
                sum2hi_1 = _mm_madd_epi16(sum2hi_1, sum2mul);

                adlerlo_0 = _mm_add_epi32(adlerlo_0, adlerlo_1);
                sum2lo_0  = _mm_add_epi32(sum2lo_0, sum2hi_0);
                sum2lo_1  = _mm_add_epi32(sum2lo_1, sum2hi_1);
                sum2lo_0  = _mm_add_epi32(sum2lo_0, sum2lo_1);
                adler_acc = _mm_add_epi32(adler_acc, adlerlo_0);
                sum2_acc  = _mm_add_epi32(sum2_acc, sum2lo_0);

                len -= 32;
                buf += 32;
            } while (len >= 32);
            /* reduce */
            {
                __m128i sum2hi;

                adler_acc = _mm_add_epi32(adler_acc, _mm_srli_si128(adler_acc, 8)); /* reduce adler addend to 1*32bits */

                sum2hi = _mm_srli_si128(sum2_acc, 8);
                sum2_acc = _mm_add_epi32(sum2_acc, sum2hi); /* reduce to 2*32bits */
                sum2hi = _mm_srli_si128(sum2_acc, 4);
                sum2_acc = _mm_add_epi32(sum2_acc, sum2hi); /* reduce to 1*32bits */

                sum2  += len32 * adler + (unsigned)_mm_cvtsi128_si32(sum2_acc);
                adler += (unsigned)_mm_cvtsi128_si32(adler_acc); /* Update adler sum */
            }
        }
        if (len & 16U) {
            __m128i src     = _mm_load_si128((const __m128i*)buf);
            __m128i adlerlo = _mm_sad_epu8(src, c_zero); /* compute sums of elements in upper / lower qwords */
            __m128i adlerhi =  _mm_srli_si128(adlerlo, 8); /* get hi part of adler addend in the low qword */
            __m128i adlerr  = _mm_add_epi16(adlerlo, adlerhi); /* reduce adler addend to 1*16bits */

            __m128i srclo   = _mm_unpacklo_epi8(src, c_zero); /* convert lower 8*8bits to 8*16bits words */
            __m128i srchi   = _mm_unpackhi_epi8(src, c_zero); /* convert upper 8*8bits to 8*16bits words */
            __m128i sum2lo  = _mm_madd_epi16(srclo, c_sum2lo); /* compute weighs of each element in sum2 addend */
            __m128i sum2hi  = _mm_madd_epi16(srchi, c_sum2hi); /* compute weighs of each element in sum2 addend */
            __m128i sum2r   = _mm_add_epi16(sum2lo, sum2hi); /* start reducing sum2 addend */

            sum2r = _mm_add_epi32(sum2r, _mm_srli_si128(sum2r, 8)); /* reduce to 2*32bits */
            sum2r = _mm_add_epi32(sum2r, _mm_srli_si128(sum2r, 4)); /* reduce to 1*32bits */

            sum2  += 16 * adler + (unsigned)_mm_cvtsi128_si32(sum2r);
            adler += (unsigned)_mm_cvtsi128_si32(adlerr); /* Update adler sum */

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

ZLIB_INTERNAL uLong adler32_copy_sse2(iadler, buf, len, dest)
    uLong iadler;
    const Bytef *buf;
    uInt len;
    Bytef *dest;
{
    unsigned adler = (unsigned)iadler;
    unsigned sum2;
    unsigned n;
    const __m128i c_zero   = _mm_setzero_si128();
    const __m128i c_sum2hi = _mm_set_epi16(1,  2,  3,  4,  5,  6,  7,  8); /* weighs for sum2 addend elements */
    const __m128i c_sum2lo = _mm_set_epi16(9, 10, 11, 12, 13, 14, 15, 16); /* weighs for sum2 addend elements */
    const __m128i c_sum2mul = _mm_set_epi16(NMAX-7, NMAX-6, NMAX-5, NMAX-4, NMAX-3, NMAX-2, NMAX-1, NMAX);
    const __m128i c_sum2add = _mm_set1_epi16(8);

    /* split Adler-32 into component sums */
    sum2 = (adler >> 16) & 0xffffU;
    adler &= 0xffffU;

    /* in case user likes doing a byte at a time, keep it fast */
    if (len == 1) {
        dest[0] = buf[0];
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
            unsigned val = *buf++;
            *dest++ = val;
            adler += val;
            sum2 += adler;
        }
        if (adler >= BASE)
            adler -= BASE;
        MOD28(sum2);            /* only added so many BASE's */
        return adler | (sum2 << 16);
    }
    else {
        /* align buf on 16bytes boundary */
        uInt align = (16U - ((uInt)buf & 15U)) & 15U;
        if (align) {
            while (align--) {
                unsigned val = *buf++;
                *dest++ = val;
                adler += val;
                sum2 += adler;
                len--;
            }
            /* NMAX is in fact NMAX - 16, no need for modulo here */
        }
    }

    /* do length NMAX blocks -- requires just one modulo operation */
    while (len >= NMAX) {
        __m128i adler_acc = _mm_setzero_si128();
        __m128i sum2_acc  = _mm_setzero_si128();
        __m128i sum2mul   = _mm_add_epi16(c_sum2mul, c_sum2add);

        len -= NMAX;
        n = NMAX / 32;          /* NMAX is divisible by 16 */
        do {
            __m128i src_0;
            __m128i adlerlo_0;
            __m128i sum2lo_0, sum2hi_0;
            __m128i src_1;
            __m128i adlerlo_1;
            __m128i sum2lo_1, sum2hi_1;
            __m128i sum2mul_0, sum2mul_1, sum2mul_2;

            src_0     = _mm_load_si128((const __m128i*)(buf +  0));
            src_1     = _mm_load_si128((const __m128i*)(buf + 16));

            _mm_storeu_si128((__m128i*)(dest +  0), src_0);
            _mm_storeu_si128((__m128i*)(dest + 16), src_1);

            adlerlo_0 = _mm_sad_epu8(src_0, c_zero); /* compute sums of elements in upper / lower qwords */
            adlerlo_1 = _mm_sad_epu8(src_1, c_zero); /* compute sums of elements in upper / lower qwords */

            sum2lo_0 = _mm_unpacklo_epi8(src_0, c_zero); /* convert lower 8*8bits to 8*16bits words */
            sum2hi_0 = _mm_unpackhi_epi8(src_0, c_zero); /* convert upper 8*8bits to 8*16bits words */
            sum2lo_1 = _mm_unpacklo_epi8(src_1, c_zero); /* convert lower 8*8bits to 8*16bits words */
            sum2hi_1 = _mm_unpackhi_epi8(src_1, c_zero); /* convert upper 8*8bits to 8*16bits words */

            sum2mul_0 = _mm_sub_epi16(sum2mul, c_sum2add);
            sum2mul_1 = _mm_sub_epi16(sum2mul_0, c_sum2add);
            sum2mul_2 = _mm_sub_epi16(sum2mul_1, c_sum2add);
            sum2mul   = _mm_sub_epi16(sum2mul_2, c_sum2add);

            sum2lo_0 = _mm_madd_epi16(sum2lo_0, sum2mul_0);
            sum2hi_0 = _mm_madd_epi16(sum2hi_0, sum2mul_1);
            sum2lo_1 = _mm_madd_epi16(sum2lo_1, sum2mul_2);
            sum2hi_1 = _mm_madd_epi16(sum2hi_1, sum2mul);

            adlerlo_0 = _mm_add_epi32(adlerlo_0, adlerlo_1);
            sum2lo_0  = _mm_add_epi32(sum2lo_0, sum2hi_0);
            sum2lo_1  = _mm_add_epi32(sum2lo_1, sum2hi_1);
            sum2lo_0  = _mm_add_epi32(sum2lo_0, sum2lo_1);
            adler_acc = _mm_add_epi32(adler_acc, adlerlo_0);
            sum2_acc  = _mm_add_epi32(sum2_acc, sum2lo_0);

            buf += 32;
            dest += 32;
        } while (--n);

        /* reduce */
        {
            __m128i sum2hi;

            adler_acc = _mm_add_epi32(adler_acc, _mm_srli_si128(adler_acc, 8)); /* reduce adler addend to 1*32bits */

            sum2hi = _mm_srli_si128(sum2_acc, 8);
            sum2_acc = _mm_add_epi32(sum2_acc, sum2hi); /* reduce to 2*32bits */
            sum2hi = _mm_srli_si128(sum2_acc, 4);
            sum2_acc = _mm_add_epi32(sum2_acc, sum2hi); /* reduce to 1*32bits */

            sum2  += NMAX * adler + (unsigned)_mm_cvtsi128_si32(sum2_acc);
            adler += (unsigned)_mm_cvtsi128_si32(adler_acc); /* Update adler sum */
        }

        MOD(adler);
        MOD(sum2);
    }

    /* do remaining bytes (less than NMAX, still just one modulo) */
    if (len) {                  /* avoid modulos if none remaining */
        if (len >= 32) {
            __m128i adler_acc = _mm_setzero_si128();
            __m128i sum2_acc  = _mm_setzero_si128();
            __m128i sum2mul   = _mm_add_epi16(c_sum2mul, c_sum2add);
            uInt len32 = len & ~(uInt)31U;

            sum2mul = _mm_sub_epi16(sum2mul, _mm_set1_epi16(NMAX - len32));

            do {
                __m128i src_0;
                __m128i adlerlo_0;
                __m128i sum2lo_0, sum2hi_0;
                __m128i src_1;
                __m128i adlerlo_1;
                __m128i sum2lo_1, sum2hi_1;
                __m128i sum2mul_0, sum2mul_1, sum2mul_2;

                src_0     = _mm_load_si128((const __m128i*)(buf +  0));
                src_1     = _mm_load_si128((const __m128i*)(buf + 16));

                _mm_storeu_si128((__m128i*)(dest +  0), src_0);
                _mm_storeu_si128((__m128i*)(dest + 16), src_1);

                adlerlo_0 = _mm_sad_epu8(src_0, c_zero); /* compute sums of elements in upper / lower qwords */
                adlerlo_1 = _mm_sad_epu8(src_1, c_zero); /* compute sums of elements in upper / lower qwords */

                sum2lo_0 = _mm_unpacklo_epi8(src_0, c_zero); /* convert lower 8*8bits to 8*16bits words */
                sum2hi_0 = _mm_unpackhi_epi8(src_0, c_zero); /* convert upper 8*8bits to 8*16bits words */
                sum2lo_1 = _mm_unpacklo_epi8(src_1, c_zero); /* convert lower 8*8bits to 8*16bits words */
                sum2hi_1 = _mm_unpackhi_epi8(src_1, c_zero); /* convert upper 8*8bits to 8*16bits words */

                sum2mul_0 = _mm_sub_epi16(sum2mul, c_sum2add);
                sum2mul_1 = _mm_sub_epi16(sum2mul_0, c_sum2add);
                sum2mul_2 = _mm_sub_epi16(sum2mul_1, c_sum2add);
                sum2mul   = _mm_sub_epi16(sum2mul_2, c_sum2add);

                sum2lo_0 = _mm_madd_epi16(sum2lo_0, sum2mul_0);
                sum2hi_0 = _mm_madd_epi16(sum2hi_0, sum2mul_1);
                sum2lo_1 = _mm_madd_epi16(sum2lo_1, sum2mul_2);
                sum2hi_1 = _mm_madd_epi16(sum2hi_1, sum2mul);

                adlerlo_0 = _mm_add_epi32(adlerlo_0, adlerlo_1);
                sum2lo_0  = _mm_add_epi32(sum2lo_0, sum2hi_0);
                sum2lo_1  = _mm_add_epi32(sum2lo_1, sum2hi_1);
                sum2lo_0  = _mm_add_epi32(sum2lo_0, sum2lo_1);
                adler_acc = _mm_add_epi32(adler_acc, adlerlo_0);
                sum2_acc  = _mm_add_epi32(sum2_acc, sum2lo_0);

                len -= 32;
                buf += 32;
                dest += 32;
            } while (len >= 32);
            /* reduce */
            {
                __m128i sum2hi;

                adler_acc = _mm_add_epi32(adler_acc, _mm_srli_si128(adler_acc, 8)); /* reduce adler addend to 1*32bits */

                sum2hi = _mm_srli_si128(sum2_acc, 8);
                sum2_acc = _mm_add_epi32(sum2_acc, sum2hi); /* reduce to 2*32bits */
                sum2hi = _mm_srli_si128(sum2_acc, 4);
                sum2_acc = _mm_add_epi32(sum2_acc, sum2hi); /* reduce to 1*32bits */

                sum2  += len32 * adler + (unsigned)_mm_cvtsi128_si32(sum2_acc);
                adler += (unsigned)_mm_cvtsi128_si32(adler_acc); /* Update adler sum */
            }
        }
        if (len & 16U) {
            __m128i src     = _mm_load_si128((const __m128i*)buf);
            __m128i adlerlo = _mm_sad_epu8(src, c_zero); /* compute sums of elements in upper / lower qwords */
            __m128i adlerhi =  _mm_srli_si128(adlerlo, 8); /* get hi part of adler addend in the low qword */
            __m128i adlerr  = _mm_add_epi16(adlerlo, adlerhi); /* reduce adler addend to 1*16bits */

            __m128i srclo   = _mm_unpacklo_epi8(src, c_zero); /* convert lower 8*8bits to 8*16bits words */
            __m128i srchi   = _mm_unpackhi_epi8(src, c_zero); /* convert upper 8*8bits to 8*16bits words */
            __m128i sum2lo  = _mm_madd_epi16(srclo, c_sum2lo); /* compute weighs of each element in sum2 addend */
            __m128i sum2hi  = _mm_madd_epi16(srchi, c_sum2hi); /* compute weighs of each element in sum2 addend */
            __m128i sum2r   = _mm_add_epi16(sum2lo, sum2hi); /* start reducing sum2 addend */

            _mm_storeu_si128((__m128i*)(dest +  0), src);

            sum2r = _mm_add_epi32(sum2r, _mm_srli_si128(sum2r, 8)); /* reduce to 2*32bits */
            sum2r = _mm_add_epi32(sum2r, _mm_srli_si128(sum2r, 4)); /* reduce to 1*32bits */

            sum2  += 16 * adler + (unsigned)_mm_cvtsi128_si32(sum2r);
            adler += (unsigned)_mm_cvtsi128_si32(adlerr); /* Update adler sum */
            
            len -= 16;
            buf += 16;
            dest += 16;
        }
        while (len--) {
            unsigned val = *buf++;
            *dest++ = val;
            adler += val;
            sum2 += adler;
        }
        MOD(adler);
        MOD(sum2);
    }
    
    /* return recombined sums */
    return adler | (sum2 << 16);
}
