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
static void NEON_accum32(uint32_t *s, const unsigned char *buf, unsigned int len)
{
    static const uint8_t taps[32] = {
        32, 31, 30, 29, 28, 27, 26, 25,
        24, 23, 22, 21, 20, 19, 18, 17,
        16, 15, 14, 13, 12, 11, 10, 9,
        8, 7, 6, 5, 4, 3, 2, 1 };

    uint8x16_t t0 = vld1q_u8(taps), t1 = vld1q_u8(taps + 16);

    uint32x4_t adacc = vdupq_n_u32(0), s2acc = vdupq_n_u32(0);
    adacc = vsetq_lane_u32(s[0], adacc, 0);
    s2acc = vsetq_lane_u32(s[1], s2acc, 0);

    while (len >= 2) {
        uint8x16_t d0 = vld1q_u8(buf), d1 = vld1q_u8(buf + 16);
        uint16x8_t adler, sum2;
        s2acc = vaddq_u32(s2acc, vshlq_n_u32(adacc, 5));
        adler = vpaddlq_u8(       d0);
        adler = vpadalq_u8(adler, d1);
        sum2 = vmull_u8(      vget_low_u8(t0), vget_low_u8(d0));
        sum2 = vmlal_u8(sum2, vget_high_u8(t0), vget_high_u8(d0));
        sum2 = vmlal_u8(sum2, vget_low_u8(t1), vget_low_u8(d1));
        sum2 = vmlal_u8(sum2, vget_high_u8(t1), vget_high_u8(d1));
        adacc = vpadalq_u16(adacc, adler);
        s2acc = vpadalq_u16(s2acc, sum2);
        len -= 2;
        buf += 32;
    }


    while (len > 0) {
        uint8x16_t d0 = vld1q_u8(buf);
        uint16x8_t adler, sum2;
        s2acc = vaddq_u32(s2acc, vshlq_n_u32(adacc, 4));
        adler = vpaddlq_u8(d0);
        sum2 = vmull_u8(      vget_low_u8(t1), vget_low_u8(d0));
        sum2 = vmlal_u8(sum2, vget_high_u8(t1), vget_high_u8(d0));
        adacc = vpadalq_u16(adacc, adler);
        s2acc = vpadalq_u16(s2acc, sum2);
        buf += 16;
        len--;
    }

    {
        uint32x2_t adacc2 = vpadd_u32(vget_low_u32(adacc), vget_high_u32(adacc));
        uint32x2_t s2acc2 = vpadd_u32(vget_low_u32(s2acc), vget_high_u32(s2acc));
        uint32x2_t as = vpadd_u32(adacc2, s2acc2);
        s[0] = vget_lane_u32(as, 0);
        s[1] = vget_lane_u32(as, 1);
    }
}

static void NEON_handle_tail(uint32_t *pair, const unsigned char *buf,
                             unsigned int len)
{
    /* Oldie K&R code integration. */
    unsigned int i;
    for (i = 0; i < len; ++i) {
        pair[0] += buf[i];
        pair[1] += pair[0];
    }
}

unsigned long adler32_neon_5(unsigned long adler, const unsigned char *buf,
                           const unsigned int len)
{
    /* The largest prime smaller than 65536. */
    const uint32_t M_BASE = 65521;
    /* This is the threshold where doing accumulation may overflow. */
    const int M_NMAX = 5552;

    unsigned long sum2;
    uint32_t pair[2];
    int n = M_NMAX;
    unsigned int done = 0;
    /* Oldie K&R code integration. */
    unsigned int i;

    /* initial Adler-32 value (deferred check for len == 1 speed) */
    if (buf == Z_NULL)
        return 1L;

    /* Split Adler-32 into component sums, it can be supplied by
     * the caller sites (e.g. in a PNG file).
     */
    sum2 = (adler >> 16) & 0xffff;
    adler &= 0xffff;
    pair[0] = adler;
    pair[1] = sum2;

    for (i = 0; i < len; i += n) {
        if ((i + n) > len)
            n = len - i;

        if (n < 16)
            break;

        NEON_accum32(pair, buf + i, n / 16);
        pair[0] %= M_BASE;
        pair[1] %= M_BASE;

        done += (n / 16) * 16;
    }

    /* Handle the tail elements. */
    if (done < len) {
        NEON_handle_tail(pair, (buf + done), len - done);
        pair[0] %= M_BASE;
        pair[1] %= M_BASE;
    }

    /* D = B * 65536 + A, see: https://en.wikipedia.org/wiki/Adler-32. */
    return (pair[1] << 16) | pair[0];
}
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
        uint32x4_t sum2x4, adlerx4;

        len -= NMAX;
        n = NMAX / 32;          /* NMAX is divisible by 16 */

        sum2x4 = vdupq_n_u32(0U);
        adlerx4 = vdupq_n_u32(0U);

        sum2x4 = vsetq_lane_u32(sum2, sum2x4, 0);
        adlerx4 = vsetq_lane_u32(adler, adlerx4, 0);

        do {
            uint8x16x2_t src_0;
            uint16x8_t adler_0;
            uint16x8_t sum2_0, sum2_1;

            src_0     = vld1q_u8_x2(buf +  0);

            sum2x4 = vaddq_u32(sum2x4, vshlq_n_u32(adlerx4, 5));

            adler_0 = vpaddlq_u8(src_0.val[0]);
            adler_0 = vpadalq_u8(adler_0, src_0.val[1]);
            adlerx4 = vpadalq_u16(adlerx4, adler_0);

            sum2_0 = vmull_u8(vget_low_u8(src_0.val[0]), vget_low_u8(c_sum2.val[0]));
            sum2_1 = vmull_high_u8(src_0.val[0], c_sum2.val[0]);
            sum2_0 = vmlal_u8(sum2_0, vget_low_u8(src_0.val[1]), vget_low_u8(c_sum2.val[1]));
            sum2_1 = vmlal_high_u8(sum2_1, src_0.val[1], c_sum2.val[1]);
            sum2_0 = vaddq_u16(sum2_0, sum2_1);
            sum2x4 = vpadalq_u16(sum2x4, sum2_0);

            buf += 32;
        } while (--n);

        sum2  = (unsigned)vaddvq_u32(sum2x4);
        adler = (unsigned)vaddvq_u32(adlerx4); /* Update adler sum */

        MOD(adler);
        MOD(sum2);
    }

    /* do remaining bytes (less than NMAX, still just one modulo) */
    if (len) {                  /* avoid modulos if none remaining */
        while (len >= 32) {
            uint8x16x2_t src_0;
            uint16x8_t adler_0;
            uint16x8_t sum2_0, sum2_1;
            uint32x4_t sum2x4, adlerx4;

            src_0     = vld1q_u8_x2(buf +  0);

            adler_0 = vpaddlq_u8(src_0.val[0]);
            adler_0 = vpadalq_u8(adler_0, src_0.val[1]);
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
