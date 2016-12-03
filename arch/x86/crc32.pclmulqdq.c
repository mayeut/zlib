/* arch/x86/crc32.pclmulqdq.c -- compute the CRC32 using a parallelized
 * folding approach with the PCLMULQDQ instruction.
 * Copyright (C) 2013 Intel Corporation. All rights reserved.
 * Authors:
 * 	Wajdi Feghali   <wajdi.k.feghali@intel.com>
 * 	Jim Guilford    <james.guilford@intel.com>
 * 	Vinodh Gopal    <vinodh.gopal@intel.com>
 * 	Erdinc Ozturk   <erdinc.ozturk@intel.com>
 * 	Jim Kukunas     <james.t.kukunas@linux.intel.com>
 * Copyright (C) 2016 Matthieu Darbois
 * For conditions of distribution and use, see copyright notice in zlib.h
 *
 * A white paper describing this algorithm can be found at:
 * http://www.intel.com/content/dam/www/public/us/en/documents/white-papers/fast-crc-computation-generic-polynomials-pclmulqdq-paper.pdf
 */

/* @(#) $Id$ */

#include "zutil.h"

#include <inttypes.h>
#include <immintrin.h>
#include <wmmintrin.h>

#define FOLD_8

#if defined(FOLD_8)
local void fold_8(__m128i *xmm_crc0, __m128i *xmm_crc1,
                  __m128i *xmm_crc2, __m128i *xmm_crc3,
                  __m128i *xmm_crc4, __m128i *xmm_crc5,
                  __m128i *xmm_crc6, __m128i *xmm_crc7)
{
    z_const __m128i xmm_fold8 = _mm_set_epi32(
                                              0x00000001, 0x4a7fe880,
                                              0x00000001, 0xe88ef372
                                              );

    __m128i tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
    __m128i tmp8, tmp9, tmpA, tmpB, tmpC, tmpD, tmpE, tmpF;

    tmp0 = _mm_clmulepi64_si128(*xmm_crc0, xmm_fold8, 0x00);
    tmp1 = _mm_clmulepi64_si128(*xmm_crc0, xmm_fold8, 0x11);

    tmp2 = _mm_clmulepi64_si128(*xmm_crc1, xmm_fold8, 0x00);
    tmp3 = _mm_clmulepi64_si128(*xmm_crc1, xmm_fold8, 0x11);

    tmp4 = _mm_clmulepi64_si128(*xmm_crc2, xmm_fold8, 0x00);
    tmp5 = _mm_clmulepi64_si128(*xmm_crc2, xmm_fold8, 0x11);

    tmp6 = _mm_clmulepi64_si128(*xmm_crc3, xmm_fold8, 0x00);
    tmp7 = _mm_clmulepi64_si128(*xmm_crc3, xmm_fold8, 0x11);

    tmp8 = _mm_clmulepi64_si128(*xmm_crc4, xmm_fold8, 0x00);
    tmp9 = _mm_clmulepi64_si128(*xmm_crc4, xmm_fold8, 0x11);

    tmpA = _mm_clmulepi64_si128(*xmm_crc5, xmm_fold8, 0x00);
    tmpB = _mm_clmulepi64_si128(*xmm_crc5, xmm_fold8, 0x11);

    tmpC = _mm_clmulepi64_si128(*xmm_crc6, xmm_fold8, 0x00);
    tmpD = _mm_clmulepi64_si128(*xmm_crc6, xmm_fold8, 0x11);

    tmpE = _mm_clmulepi64_si128(*xmm_crc7, xmm_fold8, 0x00);
    tmpF = _mm_clmulepi64_si128(*xmm_crc7, xmm_fold8, 0x11);

    *xmm_crc0 = _mm_xor_si128((tmp0), (tmp1));
    *xmm_crc1 = _mm_xor_si128((tmp2), (tmp3));
    *xmm_crc2 = _mm_xor_si128((tmp4), (tmp5));
    *xmm_crc3 = _mm_xor_si128((tmp6), (tmp7));
    *xmm_crc4 = _mm_xor_si128((tmp8), (tmp9));
    *xmm_crc5 = _mm_xor_si128((tmpA), (tmpB));
    *xmm_crc6 = _mm_xor_si128((tmpC), (tmpD));
    *xmm_crc7 = _mm_xor_si128((tmpE), (tmpF));
}

local void fold_8_7(__m128i *xmm_crc0, __m128i *xmm_crc1,
                    __m128i *xmm_crc2, __m128i *xmm_crc3,
                    __m128i *xmm_crc4, __m128i *xmm_crc5,
                    __m128i *xmm_crc6, __m128i *xmm_crc7)
{
    z_const __m128i xmm_fold8 = _mm_set_epi32(
                                              0x00000001, 0x4a7fe880,
                                              0x00000001, 0xe88ef372
                                              );

    __m128i tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
    __m128i tmp8, tmp9, tmpA, tmpB, tmpC, tmpD;

    tmp0 = _mm_clmulepi64_si128(*xmm_crc0, xmm_fold8, 0x00);
    tmp1 = _mm_clmulepi64_si128(*xmm_crc0, xmm_fold8, 0x11);

    tmp2 = _mm_clmulepi64_si128(*xmm_crc1, xmm_fold8, 0x00);
    tmp3 = _mm_clmulepi64_si128(*xmm_crc1, xmm_fold8, 0x11);

    tmp4 = _mm_clmulepi64_si128(*xmm_crc2, xmm_fold8, 0x00);
    tmp5 = _mm_clmulepi64_si128(*xmm_crc2, xmm_fold8, 0x11);

    tmp6 = _mm_clmulepi64_si128(*xmm_crc3, xmm_fold8, 0x00);
    tmp7 = _mm_clmulepi64_si128(*xmm_crc3, xmm_fold8, 0x11);

    tmp8 = _mm_clmulepi64_si128(*xmm_crc4, xmm_fold8, 0x00);
    tmp9 = _mm_clmulepi64_si128(*xmm_crc4, xmm_fold8, 0x11);

    tmpA = _mm_clmulepi64_si128(*xmm_crc5, xmm_fold8, 0x00);
    tmpB = _mm_clmulepi64_si128(*xmm_crc5, xmm_fold8, 0x11);

    tmpC = _mm_clmulepi64_si128(*xmm_crc6, xmm_fold8, 0x00);
    tmpD = _mm_clmulepi64_si128(*xmm_crc6, xmm_fold8, 0x11);

    *xmm_crc0 = *xmm_crc7;
    *xmm_crc1 = _mm_xor_si128((tmp0), (tmp1));
    *xmm_crc2 = _mm_xor_si128((tmp2), (tmp3));
    *xmm_crc3 = _mm_xor_si128((tmp4), (tmp5));
    *xmm_crc4 = _mm_xor_si128((tmp6), (tmp7));
    *xmm_crc5 = _mm_xor_si128((tmp8), (tmp9));
    *xmm_crc6 = _mm_xor_si128((tmpA), (tmpB));
    *xmm_crc7 = _mm_xor_si128((tmpC), (tmpD));
}
#endif

local void fold_4(__m128i *xmm_crc0, __m128i *xmm_crc1,
                  __m128i *xmm_crc2, __m128i *xmm_crc3)
{
    z_const __m128i xmm_fold4 = _mm_set_epi32(
                                              0x00000001, 0xc6e41596,
                                              0x00000001, 0x54442bd4
                                              );

    __m128i x_tmp0, x_tmp1, x_tmp2, x_tmp3;
    __m128 ps_crc0, ps_crc1, ps_crc2, ps_crc3;
    __m128 ps_t0, ps_t1, ps_t2, ps_t3;
    __m128 ps_res0, ps_res1, ps_res2, ps_res3;

    x_tmp0 = *xmm_crc0;
    x_tmp1 = *xmm_crc1;
    x_tmp2 = *xmm_crc2;
    x_tmp3 = *xmm_crc3;

    *xmm_crc0 = _mm_clmulepi64_si128(*xmm_crc0, xmm_fold4, 0x00);
    x_tmp0 = _mm_clmulepi64_si128(x_tmp0, xmm_fold4, 0x11);
    ps_crc0 = _mm_castsi128_ps(*xmm_crc0);
    ps_t0 = _mm_castsi128_ps(x_tmp0);
    ps_res0 = _mm_xor_ps(ps_crc0, ps_t0);

    *xmm_crc1 = _mm_clmulepi64_si128(*xmm_crc1, xmm_fold4, 0x00);
    x_tmp1 = _mm_clmulepi64_si128(x_tmp1, xmm_fold4, 0x11);
    ps_crc1 = _mm_castsi128_ps(*xmm_crc1);
    ps_t1 = _mm_castsi128_ps(x_tmp1);
    ps_res1 = _mm_xor_ps(ps_crc1, ps_t1);

    *xmm_crc2 = _mm_clmulepi64_si128(*xmm_crc2, xmm_fold4, 0x00);
    x_tmp2 = _mm_clmulepi64_si128(x_tmp2, xmm_fold4, 0x11);
    ps_crc2 = _mm_castsi128_ps(*xmm_crc2);
    ps_t2 = _mm_castsi128_ps(x_tmp2);
    ps_res2 = _mm_xor_ps(ps_crc2, ps_t2);

    *xmm_crc3 = _mm_clmulepi64_si128(*xmm_crc3, xmm_fold4, 0x00);
    x_tmp3 = _mm_clmulepi64_si128(x_tmp3, xmm_fold4, 0x11);
    ps_crc3 = _mm_castsi128_ps(*xmm_crc3);
    ps_t3 = _mm_castsi128_ps(x_tmp3);
    ps_res3 = _mm_xor_ps(ps_crc3, ps_t3);

    *xmm_crc0 = _mm_castps_si128(ps_res0);
    *xmm_crc1 = _mm_castps_si128(ps_res1);
    *xmm_crc2 = _mm_castps_si128(ps_res2);
    *xmm_crc3 = _mm_castps_si128(ps_res3);
}
local void fold_4_3(__m128i *xmm_crc0, __m128i *xmm_crc1,
                    __m128i *xmm_crc2, __m128i *xmm_crc3)
{
    z_const __m128i xmm_fold4 = _mm_set_epi32(
                                              0x00000001, 0xc6e41596,
                                              0x00000001, 0x54442bd4
                                              );

    __m128i x_tmp3;
    __m128 ps_crc0, ps_crc1, ps_crc2, ps_crc3, ps_res32, ps_res21, ps_res10;

    x_tmp3 = *xmm_crc3;

    *xmm_crc3 = *xmm_crc2;
    *xmm_crc2 = _mm_clmulepi64_si128(*xmm_crc2, xmm_fold4, 0x00);
    *xmm_crc3 = _mm_clmulepi64_si128(*xmm_crc3, xmm_fold4, 0x11);
    ps_crc2 = _mm_castsi128_ps(*xmm_crc2);
    ps_crc3 = _mm_castsi128_ps(*xmm_crc3);
    ps_res32 = _mm_xor_ps(ps_crc2, ps_crc3);

    *xmm_crc2 = *xmm_crc1;
    *xmm_crc1 = _mm_clmulepi64_si128(*xmm_crc1, xmm_fold4, 0x00);
    *xmm_crc2 = _mm_clmulepi64_si128(*xmm_crc2, xmm_fold4, 0x11);
    ps_crc1 = _mm_castsi128_ps(*xmm_crc1);
    ps_crc2 = _mm_castsi128_ps(*xmm_crc2);
    ps_res21= _mm_xor_ps(ps_crc1, ps_crc2);

    *xmm_crc1 = *xmm_crc0;
    *xmm_crc0 = _mm_clmulepi64_si128(*xmm_crc0, xmm_fold4, 0x00);
    *xmm_crc1 = _mm_clmulepi64_si128(*xmm_crc1, xmm_fold4, 0x11);
    ps_crc0 = _mm_castsi128_ps(*xmm_crc0);
    ps_crc1 = _mm_castsi128_ps(*xmm_crc1);
    ps_res10= _mm_xor_ps(ps_crc0, ps_crc1);

    *xmm_crc0 = x_tmp3;
    *xmm_crc1 = _mm_castps_si128(ps_res10);
    *xmm_crc2 = _mm_castps_si128(ps_res21);
    *xmm_crc3 = _mm_castps_si128(ps_res32);
}

local void fold_2(__m128i *xmm_crc0, __m128i *xmm_crc1)
{
    z_const __m128i xmm_fold4 = _mm_set_epi32(
                                              0x00000001, 0x5a546366,
                                              0x00000000, 0xf1da05aa
                                              );

    __m128i x_tmp0, x_tmp1;
    __m128 ps_crc0, ps_crc1;
    __m128 ps_t0, ps_t1;
    __m128 ps_res0, ps_res1;

    x_tmp0 = *xmm_crc0;
    x_tmp1 = *xmm_crc1;

    *xmm_crc0 = _mm_clmulepi64_si128(*xmm_crc0, xmm_fold4, 0x00);
    x_tmp0 = _mm_clmulepi64_si128(x_tmp0, xmm_fold4, 0x11);
    ps_crc0 = _mm_castsi128_ps(*xmm_crc0);
    ps_t0 = _mm_castsi128_ps(x_tmp0);
    ps_res0 = _mm_xor_ps(ps_crc0, ps_t0);

    *xmm_crc1 = _mm_clmulepi64_si128(*xmm_crc1, xmm_fold4, 0x00);
    x_tmp1 = _mm_clmulepi64_si128(x_tmp1, xmm_fold4, 0x11);
    ps_crc1 = _mm_castsi128_ps(*xmm_crc1);
    ps_t1 = _mm_castsi128_ps(x_tmp1);
    ps_res1 = _mm_xor_ps(ps_crc1, ps_t1);

    *xmm_crc0 = _mm_castps_si128(ps_res0);
    *xmm_crc1 = _mm_castps_si128(ps_res1);
}

local void fold_1(__m128i *xmm_crc0)
{
    z_const __m128i xmm_fold128 = _mm_set_epi32(0x00000000, 0xccaa009e, 0x00000001, 0x751997d0);

    __m128i x_tmp0;
    __m128 ps_crc0;
    __m128 ps_t0;
    __m128 ps_res0;

    x_tmp0 = *xmm_crc0;
    *xmm_crc0 = _mm_clmulepi64_si128(*xmm_crc0, xmm_fold128, 0x00);
    x_tmp0 = _mm_clmulepi64_si128(x_tmp0, xmm_fold128, 0x11);
    ps_crc0 = _mm_castsi128_ps(*xmm_crc0);
    ps_t0 = _mm_castsi128_ps(x_tmp0);
    ps_res0 = _mm_xor_ps(ps_crc0, ps_t0);
    *xmm_crc0 = _mm_castps_si128(ps_res0);
}

local void partial_fold1(z_const size_t len,
                         __m128i *xmm_crc0,
                         __m128i *xmm_crc_part)
{
    static z_const unsigned __attribute__((aligned(32))) pshufb_shf_table[60] = {
        0x84838281,0x88878685,0x8c8b8a89,0x008f8e8d, /* shl 15 (16 - 1)/shr1 */
        0x85848382,0x89888786,0x8d8c8b8a,0x01008f8e, /* shl 14 (16 - 3)/shr2 */
        0x86858483,0x8a898887,0x8e8d8c8b,0x0201008f, /* shl 13 (16 - 4)/shr3 */
        0x87868584,0x8b8a8988,0x8f8e8d8c,0x03020100, /* shl 12 (16 - 4)/shr4 */
        0x88878685,0x8c8b8a89,0x008f8e8d,0x04030201, /* shl 11 (16 - 5)/shr5 */
        0x89888786,0x8d8c8b8a,0x01008f8e,0x05040302, /* shl 10 (16 - 6)/shr6 */
        0x8a898887,0x8e8d8c8b,0x0201008f,0x06050403, /* shl  9 (16 - 7)/shr7 */
        0x8b8a8988,0x8f8e8d8c,0x03020100,0x07060504, /* shl  8 (16 - 8)/shr8 */
        0x8c8b8a89,0x008f8e8d,0x04030201,0x08070605, /* shl  7 (16 - 9)/shr9 */
        0x8d8c8b8a,0x01008f8e,0x05040302,0x09080706, /* shl  6 (16 -10)/shr10*/
        0x8e8d8c8b,0x0201008f,0x06050403,0x0a090807, /* shl  5 (16 -11)/shr11*/
        0x8f8e8d8c,0x03020100,0x07060504,0x0b0a0908, /* shl  4 (16 -12)/shr12*/
        0x008f8e8d,0x04030201,0x08070605,0x0c0b0a09, /* shl  3 (16 -13)/shr13*/
        0x01008f8e,0x05040302,0x09080706,0x0d0c0b0a, /* shl  2 (16 -14)/shr14*/
        0x0201008f,0x06050403,0x0a090807,0x0e0d0c0b  /* shl  1 (16 -15)/shr15*/
    };

    z_const __m128i xmm_fold1 = _mm_set_epi32(0x00000000, 0xccaa009e, 0x00000001, 0x751997d0);
    z_const __m128i xmm_mask3 = _mm_set1_epi32(0x80808080);

    __m128i xmm_shl, xmm_shr;
    __m128i xmm_a0_0, xmm_a0_1;
    __m128 ps_crc3, psa0_0, psa0_1, ps_res;

    xmm_shl = _mm_load_si128((__m128i *)pshufb_shf_table + (len - 1));
    xmm_shr = xmm_shl;
    xmm_shr = _mm_xor_si128(xmm_shr, xmm_mask3);

    xmm_a0_0 = _mm_shuffle_epi8(*xmm_crc0, xmm_shl);

    *xmm_crc0 = _mm_shuffle_epi8(*xmm_crc0, xmm_shr);
    *xmm_crc_part = _mm_shuffle_epi8(*xmm_crc_part, xmm_shl);
    *xmm_crc0 = _mm_or_si128(*xmm_crc0, *xmm_crc_part);

    xmm_a0_1 = _mm_clmulepi64_si128(xmm_a0_0, xmm_fold1, 0x00);
    xmm_a0_0 = _mm_clmulepi64_si128(xmm_a0_0, xmm_fold1, 0x11);

    ps_crc3 = _mm_castsi128_ps(*xmm_crc0);
    psa0_0 = _mm_castsi128_ps(xmm_a0_0);
    psa0_1 = _mm_castsi128_ps(xmm_a0_1);

    ps_res = _mm_xor_ps(ps_crc3, psa0_0);
    ps_res = _mm_xor_ps(ps_res, psa0_1);

    *xmm_crc0 = _mm_castps_si128(ps_res);
}

local unsigned long crc_fold_128to32(__m128i xmm_crc0)
{
    static z_const unsigned __attribute__((aligned(16))) crc_k[] = {
        0xccaa009e, 0x00000000, /* rk5 */
        0x63cd6124, 0x00000001, /* rk6 */
        0xf7011641, 0x00000001, /* rk7 */
        0xdb710641, 0x00000001  /* rk8 */
    };

    unsigned int crc;
    __m128i x_tmp0, x_tmp1, crc_fold;

    /*
     * k1
     */
    crc_fold = _mm_load_si128((__m128i *)crc_k);

    x_tmp0   = xmm_crc0;
    x_tmp0   = _mm_clmulepi64_si128(x_tmp0, crc_fold, 0x00);
    xmm_crc0 = _mm_srli_si128(xmm_crc0, 8);
    xmm_crc0 = _mm_xor_si128(xmm_crc0, x_tmp0);

    x_tmp0   = _mm_castps_si128( _mm_move_ss(_mm_setzero_ps(), _mm_castsi128_ps(xmm_crc0)));
    xmm_crc0 = _mm_srli_si128(xmm_crc0, 4);
    x_tmp0   = _mm_clmulepi64_si128(x_tmp0, crc_fold, 0x10);
    xmm_crc0 = _mm_xor_si128(xmm_crc0, x_tmp0);

    crc_fold = _mm_load_si128((__m128i *)crc_k + 1);
    x_tmp0   = _mm_castps_si128( _mm_move_ss(_mm_setzero_ps(), _mm_castsi128_ps(xmm_crc0)));
    x_tmp0   = _mm_clmulepi64_si128(x_tmp0, crc_fold, 0x00);
    x_tmp1   = _mm_castps_si128( _mm_move_ss(_mm_setzero_ps(), _mm_castsi128_ps(x_tmp0)));
    x_tmp1   = _mm_clmulepi64_si128(x_tmp1, crc_fold, 0x10);
    xmm_crc0 = _mm_xor_si128(xmm_crc0, x_tmp1);
    xmm_crc0 = _mm_srli_si128(xmm_crc0, 4);

    crc = _mm_cvtsi128_si32(xmm_crc0);
    crc = ~crc;
    return (unsigned long)crc;
}

ZLIB_INTERNAL uLong crc32_generic OF((uLong crc, const Bytef *buf, uInt len));

ZLIB_INTERNAL uLong crc32_pclmulqdq(crc, buf, len)
uLong crc;
const Bytef *buf;
uInt len;
{
    uInt align = (16U - ((uInt)buf & 15U)) & 15U;
    if (len < (16 + align)) {
        crc = crc32_generic(crc, buf, len);
    } else {
        __m128i xmm_crc0, xmm_crc1, xmm_crc2, xmm_crc3;
#if defined(FOLD_8)
        __m128i xmm_crc4, xmm_crc5, xmm_crc6, xmm_crc7;
#endif
        __m128i c;

        if (align) {
            crc = crc32_generic(crc, buf, align);
            len -= align;
            buf += align;
        }
        c = _mm_cvtsi32_si128(~(z_crc_t)crc);

        xmm_crc0 = _mm_load_si128((__m128i *)(buf +  0));
        if (len < 32) {
            xmm_crc0 = _mm_xor_si128(xmm_crc0, c);
            len -= 16U;
            buf += 16;
            goto endloop16;
        }
        xmm_crc1 = _mm_load_si128((__m128i *)(buf + 16));
        if (len < 64) {
            xmm_crc0 = _mm_xor_si128(xmm_crc0, c);
            len -= 32U;
            buf += 32;
            goto endloop32;
        }
        xmm_crc2 = _mm_load_si128((__m128i *)(buf + 32));
        xmm_crc3 = _mm_load_si128((__m128i *)(buf + 48));
#if defined(FOLD_8)
        if (len < 128) {
            xmm_crc0 = _mm_xor_si128(xmm_crc0, c);
            len -= 64U;
            buf += 64;
            goto endloop64;
        }
        xmm_crc4 = _mm_load_si128((__m128i *)(buf +  64));
        xmm_crc5 = _mm_load_si128((__m128i *)(buf +  80));
        xmm_crc6 = _mm_load_si128((__m128i *)(buf +  96));
        xmm_crc7 = _mm_load_si128((__m128i *)(buf + 112));
        xmm_crc0 = _mm_xor_si128(xmm_crc0, c);
        len -= 128U;
        buf += 128;

        while (len >= 128U) {
            fold_8(&xmm_crc0, &xmm_crc1, &xmm_crc2, &xmm_crc3, &xmm_crc4, &xmm_crc5, &xmm_crc6, &xmm_crc7);

            xmm_crc0 = _mm_xor_si128(xmm_crc0, _mm_load_si128((__m128i *)(buf +   0)));
            xmm_crc1 = _mm_xor_si128(xmm_crc1, _mm_load_si128((__m128i *)(buf +  16)));
            xmm_crc2 = _mm_xor_si128(xmm_crc2, _mm_load_si128((__m128i *)(buf +  32)));
            xmm_crc3 = _mm_xor_si128(xmm_crc3, _mm_load_si128((__m128i *)(buf +  48)));
            xmm_crc4 = _mm_xor_si128(xmm_crc4, _mm_load_si128((__m128i *)(buf +  64)));
            xmm_crc5 = _mm_xor_si128(xmm_crc5, _mm_load_si128((__m128i *)(buf +  80)));
            xmm_crc6 = _mm_xor_si128(xmm_crc6, _mm_load_si128((__m128i *)(buf +  96)));
            xmm_crc7 = _mm_xor_si128(xmm_crc7, _mm_load_si128((__m128i *)(buf + 112)));

            len -= 128U;
            buf += 128;
        }
        if (len >= 112U) {
            fold_8_7(&xmm_crc0, &xmm_crc1, &xmm_crc2, &xmm_crc3, &xmm_crc4, &xmm_crc5, &xmm_crc6, &xmm_crc7);

            xmm_crc1 = _mm_xor_si128(xmm_crc1, _mm_load_si128((__m128i *)(buf +  0)));
            xmm_crc2 = _mm_xor_si128(xmm_crc2, _mm_load_si128((__m128i *)(buf + 16)));
            xmm_crc3 = _mm_xor_si128(xmm_crc3, _mm_load_si128((__m128i *)(buf + 32)));
            xmm_crc4 = _mm_xor_si128(xmm_crc4, _mm_load_si128((__m128i *)(buf + 48)));
            xmm_crc5 = _mm_xor_si128(xmm_crc5, _mm_load_si128((__m128i *)(buf + 64)));
            xmm_crc6 = _mm_xor_si128(xmm_crc6, _mm_load_si128((__m128i *)(buf + 80)));
            xmm_crc7 = _mm_xor_si128(xmm_crc7, _mm_load_si128((__m128i *)(buf + 96)));

            len -= 112U;
            buf += 112;
        }
        fold_4(&xmm_crc0, &xmm_crc1, &xmm_crc2, &xmm_crc3);
        xmm_crc0 = _mm_xor_si128(xmm_crc0, xmm_crc4);
        xmm_crc1 = _mm_xor_si128(xmm_crc1, xmm_crc5);
        xmm_crc2 = _mm_xor_si128(xmm_crc2, xmm_crc6);
        xmm_crc3 = _mm_xor_si128(xmm_crc3, xmm_crc7);
/* loop64: */
        if (len & 64U) {
#else
        xmm_crc0 = _mm_xor_si128(xmm_crc0, c);
        len -= 64U;
        buf += 64;
/* loop64: */
        while (len >= 64U) {
#endif
            fold_4(&xmm_crc0, &xmm_crc1, &xmm_crc2, &xmm_crc3);

            xmm_crc0 = _mm_xor_si128(xmm_crc0, _mm_load_si128((__m128i *)(buf +  0)));
            xmm_crc1 = _mm_xor_si128(xmm_crc1, _mm_load_si128((__m128i *)(buf + 16)));
            xmm_crc2 = _mm_xor_si128(xmm_crc2, _mm_load_si128((__m128i *)(buf + 32)));
            xmm_crc3 = _mm_xor_si128(xmm_crc3, _mm_load_si128((__m128i *)(buf + 48)));

            len -= 64U;
            buf += 64;
        }
#if defined(FOLD_8)
endloop64:
#endif
        if (len >= 48U) {
            fold_4_3(&xmm_crc0, &xmm_crc1, &xmm_crc2, &xmm_crc3);

            xmm_crc1 = _mm_xor_si128(xmm_crc1, _mm_load_si128((__m128i *)(buf +  0)));
            xmm_crc2 = _mm_xor_si128(xmm_crc2, _mm_load_si128((__m128i *)(buf + 16)));
            xmm_crc3 = _mm_xor_si128(xmm_crc3, _mm_load_si128((__m128i *)(buf + 32)));

            len -= 48U;
            buf += 48;
        }
        fold_2(&xmm_crc0, &xmm_crc1);
        xmm_crc0 = _mm_xor_si128(xmm_crc0, xmm_crc2);
        xmm_crc1 = _mm_xor_si128(xmm_crc1, xmm_crc3);
/* loop32: */
        if (len & 32U) {
            fold_2(&xmm_crc0, &xmm_crc1);

            xmm_crc0 = _mm_xor_si128(xmm_crc0, _mm_load_si128((__m128i *)(buf +  0)));
            xmm_crc1 = _mm_xor_si128(xmm_crc1, _mm_load_si128((__m128i *)(buf + 16)));

            len -= 32U;
            buf += 32;
        }
endloop32:
        fold_1(&xmm_crc0);
        xmm_crc0 = _mm_xor_si128(xmm_crc0, xmm_crc1);
/*loop16:*/
        if (len & 16U)
        {
            fold_1(&xmm_crc0);

            xmm_crc0 = _mm_xor_si128(xmm_crc0, _mm_load_si128((__m128i *)(buf +  0)));
            
            len -= 16U;
            buf += 16;
        }
endloop16:
        if (len) {
            unsigned char __attribute__((aligned(16))) partial_buf[16] = { 0 };
            union {
                unsigned char* c;
                uint16_t* u16;
                uint32_t* u32;
                uint64_t* u64;
            } dst;
            union {
                const unsigned char* c;
                const uint16_t* u16;
                const uint32_t* u32;
                const uint64_t* u64;
            } src;
            __m128i xmm_crc_part;

            src.c = buf;
            dst.c = partial_buf;

            if (len & 8) {
                *dst.u64++ = *src.u64++;
            }
            if (len & 4) {
                *dst.u32++ = *src.u32++;
            }
            if (len & 2) {
                *dst.u16++ = *src.u16++;
            }
            if (len & 1) {
                *dst.c++ = *src.c++;
            }
            xmm_crc_part = _mm_load_si128((const __m128i *)partial_buf);
            partial_fold1(len, &xmm_crc0, &xmm_crc_part);
        }
        crc = crc_fold_128to32(xmm_crc0);
    }
    return crc;
}

ZLIB_INTERNAL uLong crc32_copy_generic OF((uLong crc, const Bytef *buf, uInt len, Bytef *dest));

ZLIB_INTERNAL uLong crc32_copy_pclmulqdq(crc, buf, len, dest)
    uLong crc;
    const Bytef *buf;
    uInt len;
    Bytef *dest;
{
    uInt align = (16U - ((uInt)buf & 15U)) & 15U;
    if (len < (16 + align)) {
        crc = crc32_copy_generic(crc, buf, len, dest);
    } else {
        __m128i xmm_crc0, xmm_crc1, xmm_crc2, xmm_crc3;
        __m128i c;

        if (align) {
            crc = crc32_copy_generic(crc, buf, align, dest);
            len -= align;
            buf += align;
            dest += align;
        }
        c = _mm_cvtsi32_si128(~(z_crc_t)crc);

        xmm_crc0 = _mm_load_si128((__m128i *)(buf +  0));
        if (len < 32) {
            _mm_storeu_si128((__m128i *)(dest +  0), xmm_crc0);
            xmm_crc0 = _mm_xor_si128(xmm_crc0, c);
            len -= 16U;
            buf += 16;
            dest += 16;
            goto endloop16;
        }
        xmm_crc1 = _mm_load_si128((__m128i *)(buf + 16));
        if (len < 64) {
            _mm_storeu_si128((__m128i *)(dest +  0), xmm_crc0);
            _mm_storeu_si128((__m128i *)(dest + 16), xmm_crc1);
            xmm_crc0 = _mm_xor_si128(xmm_crc0, c);
            len -= 32U;
            buf += 32;
            dest += 32;
            goto endloop32;
        }
        xmm_crc2 = _mm_load_si128((__m128i *)(buf + 32));
        xmm_crc3 = _mm_load_si128((__m128i *)(buf + 48));
        _mm_storeu_si128((__m128i *)(dest +  0), xmm_crc0);
        _mm_storeu_si128((__m128i *)(dest + 16), xmm_crc1);
        _mm_storeu_si128((__m128i *)(dest + 32), xmm_crc2);
        _mm_storeu_si128((__m128i *)(dest + 48), xmm_crc3);

        xmm_crc0 = _mm_xor_si128(xmm_crc0, c);
        len -= 64U;
        buf += 64;
        dest += 64;
        /* loop64: */
        while (len >= 64U) {
            const __m128i src0 = _mm_load_si128((__m128i *)(buf +  0));
            const __m128i src1 = _mm_load_si128((__m128i *)(buf + 16));
            const __m128i src2 = _mm_load_si128((__m128i *)(buf + 32));
            const __m128i src3 = _mm_load_si128((__m128i *)(buf + 48));

            fold_4(&xmm_crc0, &xmm_crc1, &xmm_crc2, &xmm_crc3);

            _mm_storeu_si128((__m128i *)(dest +  0), src0);
            _mm_storeu_si128((__m128i *)(dest + 16), src1);
            _mm_storeu_si128((__m128i *)(dest + 32), src2);
            _mm_storeu_si128((__m128i *)(dest + 48), src3);

            xmm_crc0 = _mm_xor_si128(xmm_crc0, src0);
            xmm_crc1 = _mm_xor_si128(xmm_crc1, src1);
            xmm_crc2 = _mm_xor_si128(xmm_crc2, src2);
            xmm_crc3 = _mm_xor_si128(xmm_crc3, src3);

            len -= 64U;
            buf += 64;
            dest += 64;
        }
/* endloop64: */
        if (len >= 48U) {
            const __m128i src0 = _mm_load_si128((__m128i *)(buf +  0));
            const __m128i src1 = _mm_load_si128((__m128i *)(buf + 16));
            const __m128i src2 = _mm_load_si128((__m128i *)(buf + 32));

            fold_4_3(&xmm_crc0, &xmm_crc1, &xmm_crc2, &xmm_crc3);

            _mm_storeu_si128((__m128i *)(dest +  0), src0);
            _mm_storeu_si128((__m128i *)(dest + 16), src1);
            _mm_storeu_si128((__m128i *)(dest + 32), src2);

            xmm_crc1 = _mm_xor_si128(xmm_crc1, src0);
            xmm_crc2 = _mm_xor_si128(xmm_crc2, src1);
            xmm_crc3 = _mm_xor_si128(xmm_crc3, src2);

            len -= 48U;
            buf += 48;
            dest += 48;
        }
        fold_2(&xmm_crc0, &xmm_crc1);
        xmm_crc0 = _mm_xor_si128(xmm_crc0, xmm_crc2);
        xmm_crc1 = _mm_xor_si128(xmm_crc1, xmm_crc3);
/* loop32: */
        if (len & 32U) {
            const __m128i src0 = _mm_load_si128((__m128i *)(buf +  0));
            const __m128i src1 = _mm_load_si128((__m128i *)(buf + 16));

            fold_2(&xmm_crc0, &xmm_crc1);

            _mm_storeu_si128((__m128i *)(dest +  0), src0);
            _mm_storeu_si128((__m128i *)(dest + 16), src1);

            xmm_crc0 = _mm_xor_si128(xmm_crc0, src0);
            xmm_crc1 = _mm_xor_si128(xmm_crc1, src1);

            len -= 32U;
            buf += 32;
            dest += 32;
        }
endloop32:
        fold_1(&xmm_crc0);
        xmm_crc0 = _mm_xor_si128(xmm_crc0, xmm_crc1);
/*loop16:*/
        if (len & 16U)
        {
            const __m128i src0 = _mm_load_si128((__m128i *)(buf +  0));

            fold_1(&xmm_crc0);

            _mm_storeu_si128((__m128i *)(dest +  0), src0);
            xmm_crc0 = _mm_xor_si128(xmm_crc0, src0);

            len -= 16U;
            buf += 16;
            dest += 16;
        }
endloop16:
        if (len) {
            unsigned char __attribute__((aligned(16))) partial_buf[16] = { 0 };
            union {
                unsigned char* c;
                uint16_t* u16;
                uint32_t* u32;
                uint64_t* u64;
            } dst;
            union {
                unsigned char* c;
                uint16_t* u16;
                uint32_t* u32;
                uint64_t* u64;
            } dst2;
            union {
                const unsigned char* c;
                const uint16_t* u16;
                const uint32_t* u32;
                const uint64_t* u64;
            } src;
            __m128i xmm_crc_part;

            src.c = buf;
            dst.c = partial_buf;
            dst2.c = dest;

            if (len & 8) {
                *dst.u64++ = *src.u64;
                *dst2.u64++ = *src.u64++;
            }
            if (len & 4) {
                *dst.u32++ = *src.u32;
                *dst2.u32++ = *src.u32++;
            }
            if (len & 2) {
                *dst.u16++ = *src.u16;
                *dst2.u16++ = *src.u16++;
            }
            if (len & 1) {
                *dst.c++ = *src.c;
                *dst2.c++ = *src.c++;
            }
            xmm_crc_part = _mm_load_si128((const __m128i *)partial_buf);
            partial_fold1(len, &xmm_crc0, &xmm_crc_part);
        }
        crc = crc_fold_128to32(xmm_crc0);
    }
    return crc;
}

