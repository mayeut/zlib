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

local void fold_1(__m128i *xmm_crc0, __m128i *xmm_crc1,
                  __m128i *xmm_crc2, __m128i *xmm_crc3)
{
    z_const __m128i xmm_fold4 = _mm_set_epi32(
                                              0x00000001, 0x54442bd4,
                                              0x00000001, 0xc6e41596);

    __m128i x_tmp3;
    __m128 ps_crc0, ps_crc3, ps_res;

    x_tmp3 = *xmm_crc3;

    *xmm_crc3 = *xmm_crc0;
    *xmm_crc0 = _mm_clmulepi64_si128(*xmm_crc0, xmm_fold4, 0x01);
    *xmm_crc3 = _mm_clmulepi64_si128(*xmm_crc3, xmm_fold4, 0x10);
    ps_crc0 = _mm_castsi128_ps(*xmm_crc0);
    ps_crc3 = _mm_castsi128_ps(*xmm_crc3);
    ps_res = _mm_xor_ps(ps_crc0, ps_crc3);

    *xmm_crc0 = *xmm_crc1;
    *xmm_crc1 = *xmm_crc2;
    *xmm_crc2 = x_tmp3;
    *xmm_crc3 = _mm_castps_si128(ps_res);
}

local void fold_2(__m128i *xmm_crc0, __m128i *xmm_crc1,
                  __m128i *xmm_crc2, __m128i *xmm_crc3)
{
    z_const __m128i xmm_fold4 = _mm_set_epi32(
                                              0x00000001, 0x54442bd4,
                                              0x00000001, 0xc6e41596);

    __m128i x_tmp3, x_tmp2;
    __m128 ps_crc0, ps_crc1, ps_crc2, ps_crc3, ps_res31, ps_res20;

    x_tmp3 = *xmm_crc3;
    x_tmp2 = *xmm_crc2;

    *xmm_crc3 = *xmm_crc1;
    *xmm_crc1 = _mm_clmulepi64_si128(*xmm_crc1, xmm_fold4, 0x01);
    *xmm_crc3 = _mm_clmulepi64_si128(*xmm_crc3, xmm_fold4, 0x10);
    ps_crc3 = _mm_castsi128_ps(*xmm_crc3);
    ps_crc1 = _mm_castsi128_ps(*xmm_crc1);
    ps_res31= _mm_xor_ps(ps_crc3, ps_crc1);

    *xmm_crc2 = *xmm_crc0;
    *xmm_crc0 = _mm_clmulepi64_si128(*xmm_crc0, xmm_fold4, 0x01);
    *xmm_crc2 = _mm_clmulepi64_si128(*xmm_crc2, xmm_fold4, 0x10);
    ps_crc0 = _mm_castsi128_ps(*xmm_crc0);
    ps_crc2 = _mm_castsi128_ps(*xmm_crc2);
    ps_res20= _mm_xor_ps(ps_crc0, ps_crc2);

    *xmm_crc0 = x_tmp2;
    *xmm_crc1 = x_tmp3;
    *xmm_crc2 = _mm_castps_si128(ps_res20);
    *xmm_crc3 = _mm_castps_si128(ps_res31);
}

local void fold_3(__m128i *xmm_crc0, __m128i *xmm_crc1,
                  __m128i *xmm_crc2, __m128i *xmm_crc3)
{
    z_const __m128i xmm_fold4 = _mm_set_epi32(
                                              0x00000001, 0x54442bd4,
                                              0x00000001, 0xc6e41596);

    __m128i x_tmp3;
    __m128 ps_crc0, ps_crc1, ps_crc2, ps_crc3, ps_res32, ps_res21, ps_res10;

    x_tmp3 = *xmm_crc3;

    *xmm_crc3 = *xmm_crc2;
    *xmm_crc2 = _mm_clmulepi64_si128(*xmm_crc2, xmm_fold4, 0x01);
    *xmm_crc3 = _mm_clmulepi64_si128(*xmm_crc3, xmm_fold4, 0x10);
    ps_crc2 = _mm_castsi128_ps(*xmm_crc2);
    ps_crc3 = _mm_castsi128_ps(*xmm_crc3);
    ps_res32 = _mm_xor_ps(ps_crc2, ps_crc3);

    *xmm_crc2 = *xmm_crc1;
    *xmm_crc1 = _mm_clmulepi64_si128(*xmm_crc1, xmm_fold4, 0x01);
    *xmm_crc2 = _mm_clmulepi64_si128(*xmm_crc2, xmm_fold4, 0x10);
    ps_crc1 = _mm_castsi128_ps(*xmm_crc1);
    ps_crc2 = _mm_castsi128_ps(*xmm_crc2);
    ps_res21= _mm_xor_ps(ps_crc1, ps_crc2);

    *xmm_crc1 = *xmm_crc0;
    *xmm_crc0 = _mm_clmulepi64_si128(*xmm_crc0, xmm_fold4, 0x01);
    *xmm_crc1 = _mm_clmulepi64_si128(*xmm_crc1, xmm_fold4, 0x10);
    ps_crc0 = _mm_castsi128_ps(*xmm_crc0);
    ps_crc1 = _mm_castsi128_ps(*xmm_crc1);
    ps_res10= _mm_xor_ps(ps_crc0, ps_crc1);
    
    *xmm_crc0 = x_tmp3;
    *xmm_crc1 = _mm_castps_si128(ps_res10);
    *xmm_crc2 = _mm_castps_si128(ps_res21);
    *xmm_crc3 = _mm_castps_si128(ps_res32);
}

local void fold_4(__m128i *xmm_crc0, __m128i *xmm_crc1,
                  __m128i *xmm_crc2, __m128i *xmm_crc3)
{
    z_const __m128i xmm_fold4 = _mm_set_epi32(
                                              0x00000001, 0x54442bd4,
                                              0x00000001, 0xc6e41596);

    __m128i x_tmp0, x_tmp1, x_tmp2, x_tmp3;
    __m128 ps_crc0, ps_crc1, ps_crc2, ps_crc3;
    __m128 ps_t0, ps_t1, ps_t2, ps_t3;
    __m128 ps_res0, ps_res1, ps_res2, ps_res3;

    x_tmp0 = *xmm_crc0;
    x_tmp1 = *xmm_crc1;
    x_tmp2 = *xmm_crc2;
    x_tmp3 = *xmm_crc3;

    *xmm_crc0 = _mm_clmulepi64_si128(*xmm_crc0, xmm_fold4, 0x01);
    x_tmp0 = _mm_clmulepi64_si128(x_tmp0, xmm_fold4, 0x10);
    ps_crc0 = _mm_castsi128_ps(*xmm_crc0);
    ps_t0 = _mm_castsi128_ps(x_tmp0);
    ps_res0 = _mm_xor_ps(ps_crc0, ps_t0);

    *xmm_crc1 = _mm_clmulepi64_si128(*xmm_crc1, xmm_fold4, 0x01);
    x_tmp1 = _mm_clmulepi64_si128(x_tmp1, xmm_fold4, 0x10);
    ps_crc1 = _mm_castsi128_ps(*xmm_crc1);
    ps_t1 = _mm_castsi128_ps(x_tmp1);
    ps_res1 = _mm_xor_ps(ps_crc1, ps_t1);

    *xmm_crc2 = _mm_clmulepi64_si128(*xmm_crc2, xmm_fold4, 0x01);
    x_tmp2 = _mm_clmulepi64_si128(x_tmp2, xmm_fold4, 0x10);
    ps_crc2 = _mm_castsi128_ps(*xmm_crc2);
    ps_t2 = _mm_castsi128_ps(x_tmp2);
    ps_res2 = _mm_xor_ps(ps_crc2, ps_t2);

    *xmm_crc3 = _mm_clmulepi64_si128(*xmm_crc3, xmm_fold4, 0x01);
    x_tmp3 = _mm_clmulepi64_si128(x_tmp3, xmm_fold4, 0x10);
    ps_crc3 = _mm_castsi128_ps(*xmm_crc3);
    ps_t3 = _mm_castsi128_ps(x_tmp3);
    ps_res3 = _mm_xor_ps(ps_crc3, ps_t3);

    *xmm_crc0 = _mm_castps_si128(ps_res0);
    *xmm_crc1 = _mm_castps_si128(ps_res1);
    *xmm_crc2 = _mm_castps_si128(ps_res2);
    *xmm_crc3 = _mm_castps_si128(ps_res3);
}

local z_const unsigned __attribute__((aligned(32))) pshufb_shf_table[60] = {
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

local void partial_fold(z_const size_t len,
                        __m128i *xmm_crc0, __m128i *xmm_crc1,
                        __m128i *xmm_crc2, __m128i *xmm_crc3,
                        __m128i *xmm_crc_part)
{

    z_const __m128i xmm_fold4 = _mm_set_epi32(
                                              0x00000001, 0x54442bd4,
                                              0x00000001, 0xc6e41596);
    z_const __m128i xmm_mask3 = _mm_set1_epi32(0x80808080);

    __m128i xmm_shl, xmm_shr, xmm_tmp1, xmm_tmp2, xmm_tmp3;
    __m128i xmm_a0_0, xmm_a0_1;
    __m128 ps_crc3, psa0_0, psa0_1, ps_res;

    xmm_shl = _mm_load_si128((__m128i *)pshufb_shf_table + (len - 1));
    xmm_shr = xmm_shl;
    xmm_shr = _mm_xor_si128(xmm_shr, xmm_mask3);

    xmm_a0_0 = _mm_shuffle_epi8(*xmm_crc0, xmm_shl);

    *xmm_crc0 = _mm_shuffle_epi8(*xmm_crc0, xmm_shr);
    xmm_tmp1 = _mm_shuffle_epi8(*xmm_crc1, xmm_shl);
    *xmm_crc0 = _mm_or_si128(*xmm_crc0, xmm_tmp1);

    *xmm_crc1 = _mm_shuffle_epi8(*xmm_crc1, xmm_shr);
    xmm_tmp2 = _mm_shuffle_epi8(*xmm_crc2, xmm_shl);
    *xmm_crc1 = _mm_or_si128(*xmm_crc1, xmm_tmp2);

    *xmm_crc2 = _mm_shuffle_epi8(*xmm_crc2, xmm_shr);
    xmm_tmp3 = _mm_shuffle_epi8(*xmm_crc3, xmm_shl);
    *xmm_crc2 = _mm_or_si128(*xmm_crc2, xmm_tmp3);

    *xmm_crc3 = _mm_shuffle_epi8(*xmm_crc3, xmm_shr);
    *xmm_crc_part = _mm_shuffle_epi8(*xmm_crc_part, xmm_shl);
    *xmm_crc3 = _mm_or_si128(*xmm_crc3, *xmm_crc_part);

    xmm_a0_1 = _mm_clmulepi64_si128(xmm_a0_0, xmm_fold4, 0x10);
    xmm_a0_0 = _mm_clmulepi64_si128(xmm_a0_0, xmm_fold4, 0x01);

    ps_crc3 = _mm_castsi128_ps(*xmm_crc3);
    psa0_0 = _mm_castsi128_ps(xmm_a0_0);
    psa0_1 = _mm_castsi128_ps(xmm_a0_1);
    
    ps_res = _mm_xor_ps(ps_crc3, psa0_0);
    ps_res = _mm_xor_ps(ps_res, psa0_1);
    
    *xmm_crc3 = _mm_castps_si128(ps_res);
}

local unsigned long crc_fold_512to32(__m128i xmm_crc0, __m128i xmm_crc1, __m128i xmm_crc2, __m128i xmm_crc3 )
{
    static z_const unsigned __attribute__((aligned(16))) crc_k[] = {
        0xccaa009e, 0x00000000, /* rk1 */
        0x751997d0, 0x00000001, /* rk2 */
        0xccaa009e, 0x00000000, /* rk5 */
        0x63cd6124, 0x00000001, /* rk6 */
        0xf7011640, 0x00000001, /* rk7 */
        0xdb710640, 0x00000001  /* rk8 */
    };

    static z_const unsigned __attribute__((aligned(16))) crc_mask[4] = {
        0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 0x00000000
    };

    static z_const unsigned __attribute__((aligned(16))) crc_mask2[4] = {
        0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
    };

    z_const __m128i xmm_mask  = _mm_load_si128((__m128i *)crc_mask);
    z_const __m128i xmm_mask2 = _mm_load_si128((__m128i *)crc_mask2);

    unsigned int crc;
    __m128i x_tmp0, x_tmp1, x_tmp2, crc_fold;

    /*
     * k1
     */
    crc_fold = _mm_load_si128((__m128i *)crc_k);

    x_tmp0 = _mm_clmulepi64_si128(xmm_crc0, crc_fold, 0x10);
    xmm_crc0 = _mm_clmulepi64_si128(xmm_crc0, crc_fold, 0x01);
    xmm_crc1 = _mm_xor_si128(xmm_crc1, x_tmp0);
    xmm_crc1 = _mm_xor_si128(xmm_crc1, xmm_crc0);

    x_tmp1 = _mm_clmulepi64_si128(xmm_crc1, crc_fold, 0x10);
    xmm_crc1 = _mm_clmulepi64_si128(xmm_crc1, crc_fold, 0x01);
    xmm_crc2 = _mm_xor_si128(xmm_crc2, x_tmp1);
    xmm_crc2 = _mm_xor_si128(xmm_crc2, xmm_crc1);

    x_tmp2 = _mm_clmulepi64_si128(xmm_crc2, crc_fold, 0x10);
    xmm_crc2 = _mm_clmulepi64_si128(xmm_crc2, crc_fold, 0x01);
    xmm_crc3 = _mm_xor_si128(xmm_crc3, x_tmp2);
    xmm_crc3 = _mm_xor_si128(xmm_crc3, xmm_crc2);

    /*
     * k5
     */
    crc_fold = _mm_load_si128((__m128i *)crc_k + 1);

    xmm_crc0 = xmm_crc3;
    xmm_crc3 = _mm_clmulepi64_si128(xmm_crc3, crc_fold, 0);
    xmm_crc0 = _mm_srli_si128(xmm_crc0, 8);
    xmm_crc3 = _mm_xor_si128(xmm_crc3, xmm_crc0);

    xmm_crc0 = xmm_crc3;
    xmm_crc3 = _mm_slli_si128(xmm_crc3, 4);
    xmm_crc3 = _mm_clmulepi64_si128(xmm_crc3, crc_fold, 0x10);
    xmm_crc3 = _mm_xor_si128(xmm_crc3, xmm_crc0);
    xmm_crc3 = _mm_and_si128(xmm_crc3, xmm_mask2);

    /*
     * k7
     */
    xmm_crc1 = xmm_crc3;
    xmm_crc2 = xmm_crc3;
    crc_fold = _mm_load_si128((__m128i *)crc_k + 2);

    xmm_crc3 = _mm_clmulepi64_si128(xmm_crc3, crc_fold, 0);
    xmm_crc3 = _mm_xor_si128(xmm_crc3, xmm_crc2);
    xmm_crc3 = _mm_and_si128(xmm_crc3, xmm_mask);

    xmm_crc2 = xmm_crc3;
    xmm_crc3 = _mm_clmulepi64_si128(xmm_crc3, crc_fold, 0x10);
    xmm_crc3 = _mm_xor_si128(xmm_crc3, xmm_crc2);
    xmm_crc3 = _mm_xor_si128(xmm_crc3, xmm_crc1);

    crc = _mm_extract_epi32(xmm_crc3, 2);
    crc = ~crc;
    return (unsigned long)crc;
}

local uLong crc32_pclmulqdq_aligned64(crc, buf, len)
uLong crc;
const Bytef *buf;
uInt len;
{
    __m128i xmm_crc0 = _mm_load_si128((__m128i *)(buf +  0));
    __m128i xmm_crc1 = _mm_load_si128((__m128i *)(buf + 16));
    __m128i xmm_crc2 = _mm_load_si128((__m128i *)(buf + 32));
    __m128i xmm_crc3 = _mm_load_si128((__m128i *)(buf + 48));
    __m128i c = _mm_cvtsi32_si128((z_crc_t)crc);

    len -= 64U;
    buf += 64;

    xmm_crc0 = _mm_xor_si128(xmm_crc0, c);

    if (len >= 64) {
        do
        {

            fold_4(&xmm_crc0, &xmm_crc1, &xmm_crc2, &xmm_crc3);

            xmm_crc0 = _mm_xor_si128(xmm_crc0, _mm_load_si128((__m128i *)(buf +  0)));
            xmm_crc1 = _mm_xor_si128(xmm_crc1, _mm_load_si128((__m128i *)(buf + 16)));
            xmm_crc2 = _mm_xor_si128(xmm_crc2, _mm_load_si128((__m128i *)(buf + 32)));
            xmm_crc3 = _mm_xor_si128(xmm_crc3, _mm_load_si128((__m128i *)(buf + 48)));
            
            len -= 64U;
            buf += 64;
        } while (len >= 64U);
    }

    if (len >= 48U) {
        len -= 48U;

        fold_3(&xmm_crc0, &xmm_crc1, &xmm_crc2, &xmm_crc3);

        xmm_crc1 = _mm_xor_si128(xmm_crc1, _mm_load_si128((__m128i *)(buf +  0)));
        xmm_crc2 = _mm_xor_si128(xmm_crc2, _mm_load_si128((__m128i *)(buf + 16)));
        xmm_crc3 = _mm_xor_si128(xmm_crc3, _mm_load_si128((__m128i *)(buf + 32)));

        if (len == 0U)
            goto done;

        buf += 48;
    } else if (len >= 32U) {
        len -= 32U;

        fold_2(&xmm_crc0, &xmm_crc1, &xmm_crc2, &xmm_crc3);

        xmm_crc2 = _mm_xor_si128(xmm_crc2, _mm_load_si128((__m128i *)(buf +  0)));
        xmm_crc3 = _mm_xor_si128(xmm_crc3, _mm_load_si128((__m128i *)(buf + 16)));

        if (len == 0)
            goto done;

        buf += 32;
    } else if (len >= 16U) {
        len -= 16U;

        fold_1(&xmm_crc0, &xmm_crc1, &xmm_crc2, &xmm_crc3);

        xmm_crc3 = _mm_xor_si128(xmm_crc3, _mm_load_si128((__m128i *)(buf +  0)));

        if (len == 0)
            goto done;

        buf += 16;
    }
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
        partial_fold(len, &xmm_crc0, &xmm_crc1, &xmm_crc2, &xmm_crc3, &xmm_crc_part);
    }
done:
    return crc_fold_512to32(xmm_crc0, xmm_crc1, xmm_crc2, xmm_crc3);
}

local void fold128(__m128i *xmm_crc0)
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

local uLong crc32_pclmulqdq_aligned16(crc, buf, len)
uLong crc;
const Bytef *buf;
uInt len;
{
	__m128i xmm_crc0 = _mm_load_si128((__m128i *)(buf +  0));
	__m128i c = _mm_cvtsi32_si128((z_crc_t)crc);

	len -= 16U;
	buf += 16;

	xmm_crc0 = _mm_xor_si128(xmm_crc0, c);

	if (len >= 16) {
		do
		{

			fold128(&xmm_crc0);

			xmm_crc0 = _mm_xor_si128(xmm_crc0, _mm_load_si128((__m128i *)(buf +  0)));

			len -= 16U;
			buf += 16;
		} while (len >= 16U);
	}
#if 0
	if (len) {
		/* TODO */
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
		partial_fold(len, &xmm_crc0, &xmm_crc1, &xmm_crc2, &xmm_crc3, &xmm_crc_part);
	}
#endif
	return crc_fold_128to32(xmm_crc0);
}

ZLIB_INTERNAL uLong crc32_generic OF((uLong crc, const Bytef *buf, uInt len));

ZLIB_INTERNAL uLong crc32_pclmulqdq(crc, buf, len)
uLong crc;
const Bytef *buf;
uInt len;
{
	uInt align = (16U - ((uInt)buf & 15U)) & 15U;
	if (len >= (16U + align)) {
		if (align) {
			crc = crc32_generic(crc, buf, align);
			len -= align;
			buf += align;
		}
		if (len < 64U) {
			crc = crc32_pclmulqdq_aligned16(~crc, buf, len & ~(uInt)15U);
            buf += len & ~(uInt)15U;
            if (len & 15U) {
                return crc32_generic(crc, buf, len & 15U);
            }
            return crc;
		}
		return crc32_pclmulqdq_aligned64(~crc, buf, len);
	}
	return crc32_generic(crc, buf, len);
}

