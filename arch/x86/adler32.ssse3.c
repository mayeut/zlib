/* arch/x86/adler32.ssse3.c -- compute the Adler-32 checksum of a data stream using SSSE3 SIMD extensions
 * Copyright (C) 1995-2011 Mark Adler
 * Copyright (C) 2016-2017 Matthieu Darbois
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

/* @(#) $Id$ */

#include "zutil.h"
#include <immintrin.h>

#define BASE 65521      /* largest prime smaller than 65536 */
//#define NMAX 5552
#define NMAX 5536 /* we removed 16 to be divisible by 32 */
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
    const __m128i c_one    = _mm_set1_epi16(1);
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
        __m128i adler_acc = _mm_cvtsi32_si128(adler);
        __m128i sum2_acc  = _mm_cvtsi32_si128(sum2);

        len -= NMAX;
        n = NMAX / 32;          /* NMAX is divisible by 16 */

        do {
            __m128i src_0, src_1;
            __m128i adler_0, adler_1;
            __m128i sum2_0, sum2_1;

            src_0   = _mm_load_si128((const __m128i*)(buf +  0));
            src_1   = _mm_load_si128((const __m128i*)(buf + 16));

            sum2_acc = _mm_add_epi32(sum2_acc, _mm_slli_epi32(adler_acc, 5)); /* sum2 += 32 * adler */

            adler_0 = _mm_sad_epu8(src_0, c_zero); /* compute sums of elements in upper / lower qwords */
            adler_1 = _mm_sad_epu8(src_1, c_zero); /* compute sums of elements in upper / lower qwords */

            sum2_0  = _mm_maddubs_epi16(src_0, c_sum2lo); /* compute weighs of each element in sum2 addend */
            sum2_1  = _mm_maddubs_epi16(src_1, c_sum2hi); /* compute weighs of each element in sum2 addend */

            sum2_0 = _mm_madd_epi16(sum2_0, c_one); /* pairwise extend/add */
            sum2_1 = _mm_madd_epi16(sum2_1, c_one);

            adler_0 = _mm_add_epi32(adler_0, adler_1);
            sum2_0  = _mm_add_epi32(sum2_0, sum2_1);
            adler_acc = _mm_add_epi32(adler_acc, adler_0);
            sum2_acc  = _mm_add_epi32(sum2_acc, sum2_0);

            buf += 32;
        } while (--n);

        /* reduce */
        {
            __m128i tmp;

            adler_acc = _mm_add_epi32(adler_acc, _mm_srli_si128(adler_acc, 8)); /* reduce adler addend to 1*32bits */

            tmp = _mm_srli_si128(sum2_acc, 8);
            sum2_acc = _mm_add_epi32(sum2_acc, tmp); /* reduce to 2*32bits */
            tmp = _mm_srli_si128(sum2_acc, 4);
            sum2_acc = _mm_add_epi32(sum2_acc, tmp); /* reduce to 1*32bits */

            sum2  = (unsigned)_mm_cvtsi128_si32(sum2_acc);
            adler = (unsigned)_mm_cvtsi128_si32(adler_acc); /* Update adler sum */
        }

        MOD(adler);
        MOD(sum2);
    }

    /* do remaining bytes (less than NMAX, still just one modulo) */
    if (len) {                  /* avoid modulos if none remaining */
        if (len >= 16) {
            __m128i adler_acc = _mm_cvtsi32_si128(adler);
            __m128i sum2_acc  = _mm_cvtsi32_si128(sum2);
            while (len >= 32) {
                __m128i src_0, src_1;
                __m128i adler_0, adler_1;
                __m128i sum2_0, sum2_1;

                src_0   = _mm_load_si128((const __m128i*)(buf +  0));
                src_1   = _mm_load_si128((const __m128i*)(buf + 16));

                sum2_acc = _mm_add_epi32(sum2_acc, _mm_slli_epi32(adler_acc, 5)); /* sum2 += 32 * adler */

                adler_0 = _mm_sad_epu8(src_0, c_zero); /* compute sums of elements in upper / lower qwords */
                adler_1 = _mm_sad_epu8(src_1, c_zero); /* compute sums of elements in upper / lower qwords */

                sum2_0  = _mm_maddubs_epi16(src_0, c_sum2lo); /* compute weighs of each element in sum2 addend */
                sum2_1  = _mm_maddubs_epi16(src_1, c_sum2hi); /* compute weighs of each element in sum2 addend */

                sum2_0 = _mm_madd_epi16(sum2_0, c_one); /* pairwise extend/add */
                sum2_1 = _mm_madd_epi16(sum2_1, c_one);

                adler_0 = _mm_add_epi32(adler_0, adler_1);
                sum2_0  = _mm_add_epi32(sum2_0, sum2_1);
                adler_acc = _mm_add_epi32(adler_acc, adler_0);
                sum2_acc  = _mm_add_epi32(sum2_acc, sum2_0);

                len -= 32;
                buf += 32;
            }
            if (len >= 16) {
                __m128i srclo_0;
                __m128i adlerlo_0;
                __m128i sum2lo_0;

                srclo_0   = _mm_load_si128((const __m128i*)(buf +  0));

                sum2_acc = _mm_add_epi32(sum2_acc, _mm_slli_epi32(adler_acc, 4)); /* sum2 += 16 * adler */

                adlerlo_0 = _mm_sad_epu8(srclo_0, c_zero); /* compute sums of elements in upper / lower qwords */

                sum2lo_0  = _mm_maddubs_epi16(srclo_0, c_sum2hi); /* compute weighs of each element in sum2 addend */

                sum2lo_0 = _mm_madd_epi16(sum2lo_0, c_one); /* pairwise extend/add */

                adler_acc = _mm_add_epi32(adler_acc, adlerlo_0);
                sum2_acc  = _mm_add_epi32(sum2_acc, sum2lo_0);

                len -= 16;
                buf += 16;
            }
            /* reduce */
            {
                __m128i tmp;

                adler_acc = _mm_add_epi32(adler_acc, _mm_srli_si128(adler_acc, 8)); /* reduce adler addend to 1*32bits */

                tmp = _mm_srli_si128(sum2_acc, 8);
                sum2_acc = _mm_add_epi32(sum2_acc, tmp); /* reduce to 2*32bits */
                tmp = _mm_srli_si128(sum2_acc, 4);
                sum2_acc = _mm_add_epi32(sum2_acc, tmp); /* reduce to 1*32bits */

                sum2  = (unsigned)_mm_cvtsi128_si32(sum2_acc);
                adler = (unsigned)_mm_cvtsi128_si32(adler_acc); /* Update adler sum */
            }
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


ZLIB_INTERNAL uLong adler32_copy_ssse3(iadler, buf, len, dest)
uLong iadler;
const Bytef *buf;
uInt len;
Bytef *dest;
{
    unsigned adler = (unsigned)iadler;
    unsigned sum2;
    unsigned n;
    const __m128i c_zero   = _mm_setzero_si128();
    const __m128i c_one    = _mm_set1_epi16(1);
    const __m128i c_sum2hi = _mm_set_epi8( 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16); /* weighs for sum2 addend elements */
    const __m128i c_sum2lo = _mm_set_epi8(17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32); /* weighs for sum2 addend elements */

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
        __m128i adler_acc = _mm_cvtsi32_si128(adler);
        __m128i sum2_acc  = _mm_cvtsi32_si128(sum2);

        len -= NMAX;
        n = NMAX / 32;          /* NMAX is divisible by 16 */
        do {
            __m128i src_0, src_1;
            __m128i adler_0, adler_1;
            __m128i sum2_0, sum2_1;

            src_0   = _mm_load_si128((const __m128i*)(buf +  0));
            src_1   = _mm_load_si128((const __m128i*)(buf + 16));

            _mm_storeu_si128((__m128i*)(dest +  0), src_0);
            _mm_storeu_si128((__m128i*)(dest + 16), src_1);

            sum2_acc = _mm_add_epi32(sum2_acc, _mm_slli_epi32(adler_acc, 5)); /* sum2 += 32 * adler */

            adler_0 = _mm_sad_epu8(src_0, c_zero); /* compute sums of elements in upper / lower qwords */
            adler_1 = _mm_sad_epu8(src_1, c_zero); /* compute sums of elements in upper / lower qwords */

            sum2_0  = _mm_maddubs_epi16(src_0, c_sum2lo); /* compute weighs of each element in sum2 addend */
            sum2_1  = _mm_maddubs_epi16(src_1, c_sum2hi); /* compute weighs of each element in sum2 addend */

            sum2_0 = _mm_madd_epi16(sum2_0, c_one); /* pairwise extend/add */
            sum2_1 = _mm_madd_epi16(sum2_1, c_one);

            adler_0 = _mm_add_epi32(adler_0, adler_1);
            sum2_0  = _mm_add_epi32(sum2_0, sum2_1);
            adler_acc = _mm_add_epi32(adler_acc, adler_0);
            sum2_acc  = _mm_add_epi32(sum2_acc, sum2_0);

            buf += 32;
            dest += 32;
        } while (--n);

        /* reduce */
        {
            __m128i tmp;

            adler_acc = _mm_add_epi32(adler_acc, _mm_srli_si128(adler_acc, 8)); /* reduce adler addend to 1*32bits */

            tmp = _mm_srli_si128(sum2_acc, 8);
            sum2_acc = _mm_add_epi32(sum2_acc, tmp); /* reduce to 2*32bits */
            tmp = _mm_srli_si128(sum2_acc, 4);
            sum2_acc = _mm_add_epi32(sum2_acc, tmp); /* reduce to 1*32bits */

            sum2  = (unsigned)_mm_cvtsi128_si32(sum2_acc);
            adler = (unsigned)_mm_cvtsi128_si32(adler_acc); /* Update adler sum */
        }

        MOD(adler);
        MOD(sum2);
    }

    /* do remaining bytes (less than NMAX, still just one modulo) */
    if (len) {                  /* avoid modulos if none remaining */
        if (len >= 16) {
            __m128i adler_acc = _mm_cvtsi32_si128(adler);
            __m128i sum2_acc  = _mm_cvtsi32_si128(sum2);

            while (len >= 32) {
                __m128i src_0, src_1;
                __m128i adler_0, adler_1;
                __m128i sum2_0, sum2_1;

                src_0   = _mm_load_si128((const __m128i*)(buf +  0));
                src_1   = _mm_load_si128((const __m128i*)(buf + 16));

                _mm_storeu_si128((__m128i*)(dest +  0), src_0);
                _mm_storeu_si128((__m128i*)(dest + 16), src_1);

                sum2_acc = _mm_add_epi32(sum2_acc, _mm_slli_epi32(adler_acc, 5)); /* sum2 += 32 * adler */

                adler_0 = _mm_sad_epu8(src_0, c_zero); /* compute sums of elements in upper / lower qwords */
                adler_1 = _mm_sad_epu8(src_1, c_zero); /* compute sums of elements in upper / lower qwords */

                sum2_0  = _mm_maddubs_epi16(src_0, c_sum2lo); /* compute weighs of each element in sum2 addend */
                sum2_1  = _mm_maddubs_epi16(src_1, c_sum2hi); /* compute weighs of each element in sum2 addend */

                sum2_0 = _mm_madd_epi16(sum2_0, c_one); /* pairwise extend/add */
                sum2_1 = _mm_madd_epi16(sum2_1, c_one);

                adler_0 = _mm_add_epi32(adler_0, adler_1);
                sum2_0  = _mm_add_epi32(sum2_0, sum2_1);
                adler_acc = _mm_add_epi32(adler_acc, adler_0);
                sum2_acc  = _mm_add_epi32(sum2_acc, sum2_0);

                len -= 32;
                buf += 32;
                dest += 32;
            }

            if (len & 16U) {
                __m128i src_0;
                __m128i adler_0;
                __m128i sum2_0;

                src_0   = _mm_load_si128((const __m128i*)(buf +  0));

                _mm_storeu_si128((__m128i*)(dest +  0), src_0);

                sum2_acc = _mm_add_epi32(sum2_acc, _mm_slli_epi32(adler_acc, 4)); /* sum2 += 16 * adler */

                adler_0 = _mm_sad_epu8(src_0, c_zero); /* compute sums of elements in upper / lower qwords */

                sum2_0  = _mm_maddubs_epi16(src_0, c_sum2hi); /* compute weighs of each element in sum2 addend */

                sum2_0 = _mm_madd_epi16(sum2_0, c_one); /* pairwise extend/add */

                adler_acc = _mm_add_epi32(adler_acc, adler_0);
                sum2_acc  = _mm_add_epi32(sum2_acc, sum2_0);
            
                len -= 16;
                buf += 16;
                dest += 16;
            }
            /* reduce */
            {
                __m128i tmp;

                adler_acc = _mm_add_epi32(adler_acc, _mm_srli_si128(adler_acc, 8)); /* reduce adler addend to 1*32bits */

                tmp = _mm_srli_si128(sum2_acc, 8);
                sum2_acc = _mm_add_epi32(sum2_acc, tmp); /* reduce to 2*32bits */
                tmp = _mm_srli_si128(sum2_acc, 4);
                sum2_acc = _mm_add_epi32(sum2_acc, tmp); /* reduce to 1*32bits */

                sum2  = (unsigned)_mm_cvtsi128_si32(sum2_acc);
                adler = (unsigned)_mm_cvtsi128_si32(adler_acc); /* Update adler sum */
            }
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
