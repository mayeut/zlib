/* arch/x86/adler32.sse2.c -- compute the Adler-32 checksum of a data stream using SSE2 SIMD extensions
 * Copyright (C) 1995-2011 Mark Adler
 * Copyright (C) 2016 Matthieu Darbois
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

/* @(#) $Id$ */

#include "zutil.h"
#include <arm_neon.h>

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
ZLIB_INTERNAL uLong adler32_neon(iadler, buf, len)
    uLong iadler;
    const Bytef *buf;
    uInt len;
{
    unsigned adler = (unsigned)iadler;
    unsigned sum2;
    unsigned n;
    static const uint8_t c_sum2_u8[] = {
        32, 31, 30, 29, 28, 27, 26, 25,
        24, 23, 22, 21, 20, 19, 18, 17,
        16, 15, 14, 13, 12, 11, 10,  9,
         8,  7,  6,  5,  4,  3,  2,  1};
    const uint8x16x2_t c_sum2 = vld1q_u8_x2(c_sum2_u8); /* weighs for sum2 addend elements */

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
        len -= NMAX;
        n = NMAX / 32;          /* NMAX is divisible by 16 */
        do {
            uint8x16x2_t src_0;
            uint16x8_t adler_0, adler_1;
            uint16x8_t sum2_0, sum2_1;
            uint32x4_t sum2x4, adlerx4;

            src_0     = vld1q_u8_x2(buf +  0);

            adler_0 = vaddl_u8(vget_low_u8(src_0.val[0]), vget_low_u8(src_0.val[1]));
            adler_1 = vaddl_high_u8(src_0.val[0], src_0.val[1]);
            adler_0 = vaddq_u16(adler_0, adler_1);
            adlerx4 = vpaddlq_u16(adler_0);

            sum2_0 = vmull_u8(vget_low_u8(src_0.val[0]), vget_low_u8(c_sum2.val[0]));
            sum2_1 = vmull_high_u8(src_0.val[0], c_sum2.val[0]);
            sum2_0 = vmlal_u8(sum2_0, vget_low_u8(src_0.val[1]), vget_low_u8(c_sum2.val[1]));
            sum2_1 = vmlal_high_u8(sum2_1, src_0.val[1], c_sum2.val[1]);
            sum2_0 = vaddq_u16(sum2_0, sum2_1);
            sum2x4 = vpaddlq_u16(sum2_0);

            sum2  += 32 * adler + (unsigned)vaddvq_u32(sum2x4);
            adler += (unsigned)vaddvq_u32(adlerx4); /* Update adler sum */

            buf += 32;
        } while (--n);

        MOD(adler);
        MOD(sum2);
    }

    /* do remaining bytes (less than NMAX, still just one modulo) */
    if (len) {                  /* avoid modulos if none remaining */
        while (len >= 32) {
            uint8x16x2_t src_0;
            uint16x8_t adler_0, adler_1;
            uint16x8_t sum2_0, sum2_1;
            uint32x4_t sum2x4, adlerx4;

            src_0     = vld1q_u8_x2(buf +  0);

            adler_0 = vaddl_u8(vget_low_u8(src_0.val[0]), vget_low_u8(src_0.val[1]));
            adler_1 = vaddl_high_u8(src_0.val[0], src_0.val[1]);
            adler_0 = vaddq_u16(adler_0, adler_1);
            adlerx4 = vpaddlq_u16(adler_0);

            sum2_0 = vmull_u8(vget_low_u8(src_0.val[0]), vget_low_u8(c_sum2.val[0]));
            sum2_1 = vmull_high_u8(src_0.val[0], c_sum2.val[0]);
            sum2_0 = vmlal_u8(sum2_0, vget_low_u8(src_0.val[1]), vget_low_u8(c_sum2.val[1]));
            sum2_1 = vmlal_high_u8(sum2_1, src_0.val[1], c_sum2.val[1]);
            sum2_0 = vaddq_u16(sum2_0, sum2_1);
            sum2x4 = vpaddlq_u16(sum2_0);

            sum2  += 32 * adler + (unsigned)vaddvq_u32(sum2x4);
            adler += (unsigned)vaddvq_u32(adlerx4); /* Update adler sum */

            buf += 32;
            len -= 32;
        }
        if (len & 16U) {
            uint8x8x2_t src_0;
            uint16x8_t adler_0;
            uint16x8_t sum2_0, sum2_1;
            uint32x4_t sum2x4, adlerx4;

            src_0     = vld1_u8_x2(buf +  0);

            adler_0 = vaddl_u8(src_0.val[0], src_0.val[1]);
            adlerx4 = vpaddlq_u16(adler_0);

            sum2_0 = vmull_u8(src_0.val[0], vget_low_u8(c_sum2.val[1]));
            sum2_1 = vmull_u8(src_0.val[1], vget_high_u8(c_sum2.val[1]));
            sum2_0 = vaddq_u16(sum2_0, sum2_1);
            sum2x4 = vpaddlq_u16(sum2_0);

            sum2  += 16 * adler + (unsigned)vaddvq_u32(sum2x4);
            adler += (unsigned)vaddvq_u32(adlerx4); /* Update adler sum */
            
            buf += 16;
            len -= 16;
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

ZLIB_INTERNAL uLong adler32_copy_neon(iadler, buf, len, dest)
    uLong iadler;
    const Bytef *buf;
    uInt len;
    Bytef *dest;
{
    unsigned adler = (unsigned)iadler;
    unsigned sum2;
    unsigned n;
    static const uint8_t c_sum2_u8[] = {
        32, 31, 30, 29, 28, 27, 26, 25,
        24, 23, 22, 21, 20, 19, 18, 17,
        16, 15, 14, 13, 12, 11, 10,  9,
        8,  7,  6,  5,  4,  3,  2,  1};
    const uint8x16x2_t c_sum2 = vld1q_u8_x2(c_sum2_u8); /* weighs for sum2 addend elements */

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
        len -= NMAX;
        n = NMAX / 32;          /* NMAX is divisible by 16 */
        do {
            uint8x16x2_t src_0;
            uint16x8_t adler_0, adler_1;
            uint16x8_t sum2_0, sum2_1;
            uint32x4_t sum2x4, adlerx4;

            src_0     = vld1q_u8_x2(buf +  0);

            adler_0 = vaddl_u8(vget_low_u8(src_0.val[0]), vget_low_u8(src_0.val[1]));
            adler_1 = vaddl_high_u8(src_0.val[0], src_0.val[1]);
            adler_0 = vaddq_u16(adler_0, adler_1);
            adlerx4 = vpaddlq_u16(adler_0);

            vst1q_u8_x2(dest +  0, src_0);

            sum2_0 = vmull_u8(vget_low_u8(src_0.val[0]), vget_low_u8(c_sum2.val[0]));
            sum2_1 = vmull_high_u8(src_0.val[0], c_sum2.val[0]);
            sum2_0 = vmlal_u8(sum2_0, vget_low_u8(src_0.val[1]), vget_low_u8(c_sum2.val[1]));
            sum2_1 = vmlal_high_u8(sum2_1, src_0.val[1], c_sum2.val[1]);
            sum2_0 = vaddq_u16(sum2_0, sum2_1);
            sum2x4 = vpaddlq_u16(sum2_0);

            sum2  += 32 * adler + (unsigned)vaddvq_u32(sum2x4);
            adler += (unsigned)vaddvq_u32(adlerx4); /* Update adler sum */

            buf += 32;
            dest += 32;
        } while (--n);

        MOD(adler);
        MOD(sum2);
    }

    /* do remaining bytes (less than NMAX, still just one modulo) */
    if (len) {                  /* avoid modulos if none remaining */
        while (len >= 32) {
            uint8x16x2_t src_0;
            uint16x8_t adler_0, adler_1;
            uint16x8_t sum2_0, sum2_1;
            uint32x4_t sum2x4, adlerx4;

            src_0     = vld1q_u8_x2(buf +  0);

            adler_0 = vaddl_u8(vget_low_u8(src_0.val[0]), vget_low_u8(src_0.val[1]));
            adler_1 = vaddl_high_u8(src_0.val[0], src_0.val[1]);
            adler_0 = vaddq_u16(adler_0, adler_1);
            adlerx4 = vpaddlq_u16(adler_0);

            vst1q_u8_x2(dest +  0, src_0);

            sum2_0 = vmull_u8(vget_low_u8(src_0.val[0]), vget_low_u8(c_sum2.val[0]));
            sum2_1 = vmull_high_u8(src_0.val[0], c_sum2.val[0]);
            sum2_0 = vmlal_u8(sum2_0, vget_low_u8(src_0.val[1]), vget_low_u8(c_sum2.val[1]));
            sum2_1 = vmlal_high_u8(sum2_1, src_0.val[1], c_sum2.val[1]);
            sum2_0 = vaddq_u16(sum2_0, sum2_1);
            sum2x4 = vpaddlq_u16(sum2_0);

            sum2  += 32 * adler + (unsigned)vaddvq_u32(sum2x4);
            adler += (unsigned)vaddvq_u32(adlerx4); /* Update adler sum */

            buf += 32;
            dest += 32;
            len -= 32;
        }
        if (len & 16U) {
            uint8x8x2_t src_0;
            uint16x8_t adler_0;
            uint16x8_t sum2_0, sum2_1;
            uint32x4_t sum2x4, adlerx4;

            src_0     = vld1_u8_x2(buf +  0);

            adler_0 = vaddl_u8(src_0.val[0], src_0.val[1]);
            adlerx4 = vpaddlq_u16(adler_0);

            vst1_u8_x2(dest +  0, src_0);

            sum2_0 = vmull_u8(src_0.val[0], vget_low_u8(c_sum2.val[1]));
            sum2_1 = vmull_u8(src_0.val[1], vget_high_u8(c_sum2.val[1]));
            sum2_0 = vaddq_u16(sum2_0, sum2_1);
            sum2x4 = vpaddlq_u16(sum2_0);

            sum2  += 16 * adler + (unsigned)vaddvq_u32(sum2x4);
            adler += (unsigned)vaddvq_u32(adlerx4); /* Update adler sum */

            buf += 16;
            dest += 16;
            len -= 16;
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
