/* arch/arm64/crc32.crypto -- compute the CRC32 using a parallelized
 * folding approach with the PMULL instruction.
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
#include <arm_neon.h>

local void fold_8(poly64x2x4_t fold_crc,
                  uint8x16_t *xmm_crc0, uint8x16_t *xmm_crc1,
                  uint8x16_t *xmm_crc2, uint8x16_t *xmm_crc3,
                  uint8x16_t *xmm_crc4, uint8x16_t *xmm_crc5,
                  uint8x16_t *xmm_crc6, uint8x16_t *xmm_crc7)
{
    poly128_t tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
    poly128_t tmp8, tmp9, tmpA, tmpB, tmpC, tmpD, tmpE, tmpF;

    tmp0 = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(*xmm_crc0)), (poly64_t)vget_low_p64(fold_crc.val[3]));
    tmp1 = vmull_high_p64(vreinterpretq_p64_u8(*xmm_crc0), fold_crc.val[3]);

    tmp2 = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(*xmm_crc1)), (poly64_t)vget_low_p64(fold_crc.val[3]));
    tmp3 = vmull_high_p64(vreinterpretq_p64_u8(*xmm_crc1), fold_crc.val[3]);

    tmp4 = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(*xmm_crc2)), (poly64_t)vget_low_p64(fold_crc.val[3]));
    tmp5 = vmull_high_p64(vreinterpretq_p64_u8(*xmm_crc2), fold_crc.val[3]);

    tmp6 = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(*xmm_crc3)), (poly64_t)vget_low_p64(fold_crc.val[3]));
    tmp7 = vmull_high_p64(vreinterpretq_p64_u8(*xmm_crc3), fold_crc.val[3]);

    tmp8 = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(*xmm_crc4)), (poly64_t)vget_low_p64(fold_crc.val[3]));
    tmp9 = vmull_high_p64(vreinterpretq_p64_u8(*xmm_crc4), fold_crc.val[3]);

    tmpA = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(*xmm_crc5)), (poly64_t)vget_low_p64(fold_crc.val[3]));
    tmpB = vmull_high_p64(vreinterpretq_p64_u8(*xmm_crc5), fold_crc.val[3]);

    tmpC = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(*xmm_crc6)), (poly64_t)vget_low_p64(fold_crc.val[3]));
    tmpD = vmull_high_p64(vreinterpretq_p64_u8(*xmm_crc6), fold_crc.val[3]);

    tmpE = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(*xmm_crc7)), (poly64_t)vget_low_p64(fold_crc.val[3]));
    tmpF = vmull_high_p64(vreinterpretq_p64_u8(*xmm_crc7), fold_crc.val[3]);

    *xmm_crc0 = veorq_u8(vreinterpretq_u8_p128(tmp0), vreinterpretq_u8_p128(tmp1));
    *xmm_crc1 = veorq_u8(vreinterpretq_u8_p128(tmp2), vreinterpretq_u8_p128(tmp3));
    *xmm_crc2 = veorq_u8(vreinterpretq_u8_p128(tmp4), vreinterpretq_u8_p128(tmp5));
    *xmm_crc3 = veorq_u8(vreinterpretq_u8_p128(tmp6), vreinterpretq_u8_p128(tmp7));
    *xmm_crc4 = veorq_u8(vreinterpretq_u8_p128(tmp8), vreinterpretq_u8_p128(tmp9));
    *xmm_crc5 = veorq_u8(vreinterpretq_u8_p128(tmpA), vreinterpretq_u8_p128(tmpB));
    *xmm_crc6 = veorq_u8(vreinterpretq_u8_p128(tmpC), vreinterpretq_u8_p128(tmpD));
    *xmm_crc7 = veorq_u8(vreinterpretq_u8_p128(tmpE), vreinterpretq_u8_p128(tmpF));
}

local void fold_8_7(poly64x2x4_t fold_crc,
                    uint8x16_t *xmm_crc0, uint8x16_t *xmm_crc1,
                    uint8x16_t *xmm_crc2, uint8x16_t *xmm_crc3,
                    uint8x16_t *xmm_crc4, uint8x16_t *xmm_crc5,
                    uint8x16_t *xmm_crc6, uint8x16_t *xmm_crc7)
{
    poly128_t tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
    poly128_t tmp8, tmp9, tmpA, tmpB, tmpC, tmpD;

    tmp0 = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(*xmm_crc0)), (poly64_t)vget_low_p64(fold_crc.val[3]));
    tmp1 = vmull_high_p64(vreinterpretq_p64_u8(*xmm_crc0), fold_crc.val[3]);

    tmp2 = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(*xmm_crc1)), (poly64_t)vget_low_p64(fold_crc.val[3]));
    tmp3 = vmull_high_p64(vreinterpretq_p64_u8(*xmm_crc1), fold_crc.val[3]);

    tmp4 = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(*xmm_crc2)), (poly64_t)vget_low_p64(fold_crc.val[3]));
    tmp5 = vmull_high_p64(vreinterpretq_p64_u8(*xmm_crc2), fold_crc.val[3]);

    tmp6 = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(*xmm_crc3)), (poly64_t)vget_low_p64(fold_crc.val[3]));
    tmp7 = vmull_high_p64(vreinterpretq_p64_u8(*xmm_crc3), fold_crc.val[3]);

    tmp8 = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(*xmm_crc4)), (poly64_t)vget_low_p64(fold_crc.val[3]));
    tmp9 = vmull_high_p64(vreinterpretq_p64_u8(*xmm_crc4), fold_crc.val[3]);

    tmpA = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(*xmm_crc5)), (poly64_t)vget_low_p64(fold_crc.val[3]));
    tmpB = vmull_high_p64(vreinterpretq_p64_u8(*xmm_crc5), fold_crc.val[3]);

    tmpC = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(*xmm_crc6)), (poly64_t)vget_low_p64(fold_crc.val[3]));
    tmpD = vmull_high_p64(vreinterpretq_p64_u8(*xmm_crc6), fold_crc.val[3]);

    *xmm_crc0 = *xmm_crc7;
    *xmm_crc1 = veorq_u8(vreinterpretq_u8_p128(tmp0), vreinterpretq_u8_p128(tmp1));
    *xmm_crc2 = veorq_u8(vreinterpretq_u8_p128(tmp2), vreinterpretq_u8_p128(tmp3));
    *xmm_crc3 = veorq_u8(vreinterpretq_u8_p128(tmp4), vreinterpretq_u8_p128(tmp5));
    *xmm_crc4 = veorq_u8(vreinterpretq_u8_p128(tmp6), vreinterpretq_u8_p128(tmp7));
    *xmm_crc5 = veorq_u8(vreinterpretq_u8_p128(tmp8), vreinterpretq_u8_p128(tmp9));
    *xmm_crc6 = veorq_u8(vreinterpretq_u8_p128(tmpA), vreinterpretq_u8_p128(tmpB));
    *xmm_crc7 = veorq_u8(vreinterpretq_u8_p128(tmpC), vreinterpretq_u8_p128(tmpD));
}

local void fold_4(poly64x2x4_t fold_crc,
                  uint8x16_t *xmm_crc0, uint8x16_t *xmm_crc1,
                  uint8x16_t *xmm_crc2, uint8x16_t *xmm_crc3)
{
    poly128_t tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;

    tmp0 = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(*xmm_crc0)), (poly64_t)vget_low_p64(fold_crc.val[2]));
    tmp1 = vmull_high_p64(vreinterpretq_p64_u8(*xmm_crc0), fold_crc.val[2]);

    tmp2 = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(*xmm_crc1)), (poly64_t)vget_low_p64(fold_crc.val[2]));
    tmp3 = vmull_high_p64(vreinterpretq_p64_u8(*xmm_crc1), fold_crc.val[2]);

    tmp4 = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(*xmm_crc2)), (poly64_t)vget_low_p64(fold_crc.val[2]));
    tmp5 = vmull_high_p64(vreinterpretq_p64_u8(*xmm_crc2), fold_crc.val[2]);

    tmp6 = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(*xmm_crc3)), (poly64_t)vget_low_p64(fold_crc.val[2]));
    tmp7 = vmull_high_p64(vreinterpretq_p64_u8(*xmm_crc3), fold_crc.val[2]);

    *xmm_crc0 = veorq_u8(vreinterpretq_u8_p128(tmp0), vreinterpretq_u8_p128(tmp1));
    *xmm_crc1 = veorq_u8(vreinterpretq_u8_p128(tmp2), vreinterpretq_u8_p128(tmp3));
    *xmm_crc2 = veorq_u8(vreinterpretq_u8_p128(tmp4), vreinterpretq_u8_p128(tmp5));
    *xmm_crc3 = veorq_u8(vreinterpretq_u8_p128(tmp6), vreinterpretq_u8_p128(tmp7));
}

local void fold_4_3(poly64x2x4_t fold_crc,
                    uint8x16_t *xmm_crc0, uint8x16_t *xmm_crc1,
                    uint8x16_t *xmm_crc2, uint8x16_t *xmm_crc3)
{
    poly128_t tmp0, tmp1, tmp2, tmp3, tmp4, tmp5;

    tmp0 = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(*xmm_crc0)), (poly64_t)vget_low_p64(fold_crc.val[2]));
    tmp1 = vmull_high_p64(vreinterpretq_p64_u8(*xmm_crc0), fold_crc.val[2]);

    tmp2 = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(*xmm_crc1)), (poly64_t)vget_low_p64(fold_crc.val[2]));
    tmp3 = vmull_high_p64(vreinterpretq_p64_u8(*xmm_crc1), fold_crc.val[2]);

    tmp4 = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(*xmm_crc2)), (poly64_t)vget_low_p64(fold_crc.val[2]));
    tmp5 = vmull_high_p64(vreinterpretq_p64_u8(*xmm_crc2), fold_crc.val[2]);

    *xmm_crc0 = *xmm_crc3;
    *xmm_crc1 = veorq_u8(vreinterpretq_u8_p128(tmp0), vreinterpretq_u8_p128(tmp1));
    *xmm_crc2 = veorq_u8(vreinterpretq_u8_p128(tmp2), vreinterpretq_u8_p128(tmp3));
    *xmm_crc3 = veorq_u8(vreinterpretq_u8_p128(tmp4), vreinterpretq_u8_p128(tmp5));
}

local void fold_2(poly64x2x4_t fold_crc, uint8x16_t *xmm_crc0, uint8x16_t *xmm_crc1)
{
    poly128_t tmp0, tmp1, tmp2, tmp3;

    tmp0 = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(*xmm_crc0)), (poly64_t)vget_low_p64(fold_crc.val[1]));
    tmp1 = vmull_high_p64(vreinterpretq_p64_u8(*xmm_crc0), fold_crc.val[1]);

    tmp2 = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(*xmm_crc1)), (poly64_t)vget_low_p64(fold_crc.val[1]));
    tmp3 = vmull_high_p64(vreinterpretq_p64_u8(*xmm_crc1), fold_crc.val[1]);

    *xmm_crc0 = veorq_u8(vreinterpretq_u8_p128(tmp0), vreinterpretq_u8_p128(tmp1));
    *xmm_crc1 = veorq_u8(vreinterpretq_u8_p128(tmp2), vreinterpretq_u8_p128(tmp3));
}

local void fold_1(poly64x2x4_t fold_crc, uint8x16_t *xmm_crc0)
{
    poly128_t tmp0;
    poly128_t tmp1;

    tmp0 = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(*xmm_crc0)), (poly64_t)vget_low_p64(fold_crc.val[0]));
    tmp1 = vmull_high_p64(vreinterpretq_p64_u8(*xmm_crc0), fold_crc.val[0]);

    *xmm_crc0 = veorq_u8(vreinterpretq_u8_p128(tmp0), vreinterpretq_u8_p128(tmp1));
}

local void partial_fold1(z_const size_t len,
                         poly64x2x4_t fold_crc,
                         uint8x16_t *xmm_crc0,
                         uint8x16_t *xmm_crc_part)
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

    z_const uint8x16_t xmm_mask3 = vdupq_n_u8(0x80U);

    uint8x16_t xmm_shl, xmm_shr;
    uint8x16_t xmm_a0_0;
    poly128_t tmp0, tmp1;

    xmm_shl = vld1q_u8((const uint8_t*)(pshufb_shf_table + 4 * (len - 1)));
    xmm_shr = xmm_shl;
    xmm_shr = veorq_u8(xmm_shr, xmm_mask3);

    xmm_a0_0 = vqtbl1q_u8(*xmm_crc0, xmm_shl);

    *xmm_crc0 = vqtbl1q_u8(*xmm_crc0, xmm_shr);
    *xmm_crc_part = vqtbl1q_u8(*xmm_crc_part, xmm_shl);
    *xmm_crc0 = vorrq_u8(*xmm_crc0, *xmm_crc_part);

    tmp0 = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(xmm_a0_0)), (poly64_t)vget_low_p64(fold_crc.val[0]));
    tmp1 = vmull_high_p64(vreinterpretq_p64_u8(xmm_a0_0), fold_crc.val[0]);

    *xmm_crc0 = veorq_u8(*xmm_crc0, vreinterpretq_u8_p128(tmp0));
    *xmm_crc0 = veorq_u8(*xmm_crc0, vreinterpretq_u8_p128(tmp1));
}

local unsigned long fold_128to32(uint8x16_t xmm_crc0)
{

    static z_const uint64_t crc_k[] = {
        0x00000000ccaa009eU, /* rk5 */
        0x0000000163cd6124U, /* rk6 */
        0x00000001f7011641U, /* rk7 */
        0x00000001db710641U  /* rk8 */
    };

    unsigned int crc;
    uint8x16_t x_tmp0, x_tmp1;
    poly64x1x4_t crc_fold;

    /*
     * k1
     */
    crc_fold = vld1_p64_x4(crc_k);

    x_tmp0   = vreinterpretq_u8_p128(vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(xmm_crc0)), (poly64_t)crc_fold.val[0]));

    xmm_crc0 = vextq_u8(xmm_crc0, vdupq_n_u8(0U), 8);
    xmm_crc0 = veorq_u8(xmm_crc0, x_tmp0);

    x_tmp0 = vreinterpretq_u8_u32(vcopyq_lane_u32(vdupq_n_u32(0U), 0, vget_low_u32(vreinterpretq_u32_u8(xmm_crc0)), 0));

    xmm_crc0 = vextq_u8(xmm_crc0, vdupq_n_u8(0U), 4);
    x_tmp0   = vreinterpretq_u8_p128(vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(x_tmp0)), (poly64_t)crc_fold.val[1]));
    xmm_crc0 = veorq_u8(xmm_crc0, x_tmp0);


    x_tmp0 = vreinterpretq_u8_u32(vcopyq_lane_u32(vdupq_n_u32(0U), 0, vget_low_u32(vreinterpretq_u32_u8(xmm_crc0)), 0));
    x_tmp0   = vreinterpretq_u8_p128(vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(x_tmp0)), (poly64_t)crc_fold.val[2]));
    x_tmp1 = vreinterpretq_u8_u32(vcopyq_lane_u32(vdupq_n_u32(0U), 0, vget_low_u32(vreinterpretq_u32_u8(x_tmp0)), 0));

    x_tmp1   = vreinterpretq_u8_p128(vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(x_tmp1)), (poly64_t)crc_fold.val[3]));

    xmm_crc0 = veorq_u8(xmm_crc0, x_tmp1);

    crc = vget_lane_u32(vget_low_u32(vreinterpretq_u32_u8(xmm_crc0)), 1);
    crc = ~crc;
    return (unsigned long)crc;
}

ZLIB_INTERNAL uLong crc32_generic OF((uLong crc, const Bytef *buf, uInt len));

ZLIB_INTERNAL uLong crc32_crypto(crc, buf, len)
uLong crc;
const Bytef *buf;
uInt len;
{
    if (len < 16) {
        crc = crc32_generic(crc, buf, len);
    } else {
        static z_const uint64_t fold1248_u64[] = {
            0x00000001751997d0U, /* fold by  128 bits */
            0x00000000ccaa009eU,
            0x00000000f1da05aaU, /* fold by  256 bits */
            0x000000015a546366U,
            0x0000000154442bd4U, /* fold by  512 bits */
            0x00000001c6e41596U,
            0x00000001e88ef372U, /* fold by 1024 bits */
            0x000000014a7fe880U
        };
        register z_const poly64x2x4_t crc_fold = vld1q_p64_x4(fold1248_u64);
        uint8x16_t c = vreinterpretq_u8_u32(vsetq_lane_u32(~(uint32_t)crc, vdupq_n_u32(0U), 0));

        uint8x16_t xmm_crc0, xmm_crc1, xmm_crc2, xmm_crc3;
        uint8x16_t xmm_crc4, xmm_crc5, xmm_crc6, xmm_crc7;

        xmm_crc0 = vld1q_u8(buf +   0);
        if (len < 32) {
            xmm_crc0 = veorq_u8(xmm_crc0, c);
            len -= 16U;
            buf += 16;
            goto endloop16;
        }
        xmm_crc1 = vld1q_u8(buf +  16);
        if (len < 64) {
            xmm_crc0 = veorq_u8(xmm_crc0, c);
            len -= 32U;
            buf += 32;
            goto endloop32;
        }
        xmm_crc2 = vld1q_u8(buf +  32);
        xmm_crc3 = vld1q_u8(buf +  48);
        if (len < 128) {
            xmm_crc0 = veorq_u8(xmm_crc0, c);
            len -= 64U;
            buf += 64;
            goto endloop64;
        }
        xmm_crc4 = vld1q_u8(buf +  64);
        xmm_crc5 = vld1q_u8(buf +  80);
        xmm_crc6 = vld1q_u8(buf +  96);
        xmm_crc7 = vld1q_u8(buf + 112);
        xmm_crc0 = veorq_u8(xmm_crc0, c);
        len -= 128U;
        buf += 128;

        while (len >= 128U) {
            poly128_t tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
            poly128_t tmp8, tmp9, tmpA, tmpB, tmpC, tmpD, tmpE, tmpF;

            tmp0 = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(xmm_crc0)), (poly64_t)vget_low_p64(crc_fold.val[3]));
            tmp1 = vmull_high_p64(vreinterpretq_p64_u8(xmm_crc0), crc_fold.val[3]);

            tmp2 = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(xmm_crc1)), (poly64_t)vget_low_p64(crc_fold.val[3]));
            tmp3 = vmull_high_p64(vreinterpretq_p64_u8(xmm_crc1), crc_fold.val[3]);

            tmp4 = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(xmm_crc2)), (poly64_t)vget_low_p64(crc_fold.val[3]));
            tmp5 = vmull_high_p64(vreinterpretq_p64_u8(xmm_crc2), crc_fold.val[3]);

            tmp6 = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(xmm_crc3)), (poly64_t)vget_low_p64(crc_fold.val[3]));
            tmp7 = vmull_high_p64(vreinterpretq_p64_u8(xmm_crc3), crc_fold.val[3]);

            tmp8 = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(xmm_crc4)), (poly64_t)vget_low_p64(crc_fold.val[3]));
            tmp9 = vmull_high_p64(vreinterpretq_p64_u8(xmm_crc4), crc_fold.val[3]);

            tmpA = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(xmm_crc5)), (poly64_t)vget_low_p64(crc_fold.val[3]));
            tmpB = vmull_high_p64(vreinterpretq_p64_u8(xmm_crc5), crc_fold.val[3]);

            tmpC = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(xmm_crc6)), (poly64_t)vget_low_p64(crc_fold.val[3]));
            tmpD = vmull_high_p64(vreinterpretq_p64_u8(xmm_crc6), crc_fold.val[3]);

            tmpE = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(xmm_crc7)), (poly64_t)vget_low_p64(crc_fold.val[3]));
            tmpF = vmull_high_p64(vreinterpretq_p64_u8(xmm_crc7), crc_fold.val[3]);

            xmm_crc0 = veorq_u8(vreinterpretq_u8_p128(tmp0), vreinterpretq_u8_p128(tmp1));
            xmm_crc1 = veorq_u8(vreinterpretq_u8_p128(tmp2), vreinterpretq_u8_p128(tmp3));
            xmm_crc2 = veorq_u8(vreinterpretq_u8_p128(tmp4), vreinterpretq_u8_p128(tmp5));
            xmm_crc3 = veorq_u8(vreinterpretq_u8_p128(tmp6), vreinterpretq_u8_p128(tmp7));
            xmm_crc4 = veorq_u8(vreinterpretq_u8_p128(tmp8), vreinterpretq_u8_p128(tmp9));
            xmm_crc5 = veorq_u8(vreinterpretq_u8_p128(tmpA), vreinterpretq_u8_p128(tmpB));
            xmm_crc6 = veorq_u8(vreinterpretq_u8_p128(tmpC), vreinterpretq_u8_p128(tmpD));
            xmm_crc7 = veorq_u8(vreinterpretq_u8_p128(tmpE), vreinterpretq_u8_p128(tmpF));

            xmm_crc0 = veorq_u8(xmm_crc0, vld1q_u8(buf +   0));
            xmm_crc1 = veorq_u8(xmm_crc1, vld1q_u8(buf +  16));
            xmm_crc2 = veorq_u8(xmm_crc2, vld1q_u8(buf +  32));
            xmm_crc3 = veorq_u8(xmm_crc3, vld1q_u8(buf +  48));
            xmm_crc4 = veorq_u8(xmm_crc4, vld1q_u8(buf +  64));
            xmm_crc5 = veorq_u8(xmm_crc5, vld1q_u8(buf +  80));
            xmm_crc6 = veorq_u8(xmm_crc6, vld1q_u8(buf +  96));
            xmm_crc7 = veorq_u8(xmm_crc7, vld1q_u8(buf + 112));

            len -= 128U;
            buf += 128;
        }
        if (len >= 112U) {
            fold_8_7(crc_fold, &xmm_crc0, &xmm_crc1, &xmm_crc2, &xmm_crc3, &xmm_crc4, &xmm_crc5, &xmm_crc6, &xmm_crc7);

            xmm_crc1 = veorq_u8(xmm_crc1, vld1q_u8(buf +  0));
            xmm_crc2 = veorq_u8(xmm_crc2, vld1q_u8(buf + 16));
            xmm_crc3 = veorq_u8(xmm_crc3, vld1q_u8(buf + 32));
            xmm_crc4 = veorq_u8(xmm_crc4, vld1q_u8(buf + 48));
            xmm_crc5 = veorq_u8(xmm_crc5, vld1q_u8(buf + 64));
            xmm_crc6 = veorq_u8(xmm_crc6, vld1q_u8(buf + 80));
            xmm_crc7 = veorq_u8(xmm_crc7, vld1q_u8(buf + 96));

            len -= 112U;
            buf += 112;
        }
        fold_4(crc_fold, &xmm_crc0, &xmm_crc1, &xmm_crc2, &xmm_crc3);
        xmm_crc0 = veorq_u8(xmm_crc0, xmm_crc4);
        xmm_crc1 = veorq_u8(xmm_crc1, xmm_crc5);
        xmm_crc2 = veorq_u8(xmm_crc2, xmm_crc6);
        xmm_crc3 = veorq_u8(xmm_crc3, xmm_crc7);
/* loop64: */
        if (len & 64U) {
            fold_4(crc_fold, &xmm_crc0, &xmm_crc1, &xmm_crc2, &xmm_crc3);

            xmm_crc0 = veorq_u8(xmm_crc0, vld1q_u8(buf +  0));
            xmm_crc1 = veorq_u8(xmm_crc1, vld1q_u8(buf + 16));
            xmm_crc2 = veorq_u8(xmm_crc2, vld1q_u8(buf + 32));
            xmm_crc3 = veorq_u8(xmm_crc3, vld1q_u8(buf + 48));

            len -= 64U;
            buf += 64;
        }
endloop64:
        if (len >= 48U) {
            fold_4_3(crc_fold, &xmm_crc0, &xmm_crc1, &xmm_crc2, &xmm_crc3);

            xmm_crc1 = veorq_u8(xmm_crc1, vld1q_u8(buf +  0));
            xmm_crc2 = veorq_u8(xmm_crc2, vld1q_u8(buf + 16));
            xmm_crc3 = veorq_u8(xmm_crc3, vld1q_u8(buf + 32));

            len -= 48U;
            buf += 48;
        }
        fold_2(crc_fold, &xmm_crc0, &xmm_crc1);
        xmm_crc0 = veorq_u8(xmm_crc0, xmm_crc2);
        xmm_crc1 = veorq_u8(xmm_crc1, xmm_crc3);
/* loop32: */
        if (len & 32U) {
            fold_2(crc_fold, &xmm_crc0, &xmm_crc1);

            xmm_crc0 = veorq_u8(xmm_crc0, vld1q_u8(buf +  0));
            xmm_crc1 = veorq_u8(xmm_crc1, vld1q_u8(buf + 16));

            len -= 32U;
            buf += 32;
        }
endloop32:
        fold_1(crc_fold, &xmm_crc0);
        xmm_crc0 = veorq_u8(xmm_crc0, xmm_crc1);
/*loop16:*/
        if (len & 16U)
        {
            fold_1(crc_fold, &xmm_crc0);

            xmm_crc0 = veorq_u8(xmm_crc0, vld1q_u8(buf +  0));

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
            uint8x16_t xmm_crc_part;

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
            xmm_crc_part = vld1q_u8(partial_buf);
            partial_fold1(len, crc_fold, &xmm_crc0, &xmm_crc_part);
        }
        crc = fold_128to32(xmm_crc0);
    }
    return crc;
}

