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
#include <arm_neon.h>

local void fold_10(uint8x16_t *xmm_crc0, uint8x16_t *xmm_crc1,
                  uint8x16_t *xmm_crc2, uint8x16_t *xmm_crc3)
{
    static z_const uint64_t xmm_fold128_u64[] = {
        0x0000000154442bd4U,
        0x00000001c6e41596U
    };
    z_const poly64x2_t xmm_fold4 = vld1q_p64(xmm_fold128_u64);

    poly128_t tmp0, tmp1;

    tmp0 = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(*xmm_crc0)), (poly64_t)vget_low_p64(xmm_fold4));
    tmp1 = vmull_high_p64(vreinterpretq_p64_u8(*xmm_crc0), xmm_fold4);

    *xmm_crc0 = *xmm_crc1;
    *xmm_crc1 = *xmm_crc2;
    *xmm_crc2 = *xmm_crc3;
    *xmm_crc3 = veorq_u8(vreinterpretq_u8_p128(tmp0), vreinterpretq_u8_p128(tmp1));
}

local void fold_20(uint8x16_t *xmm_crc0, uint8x16_t *xmm_crc1,
                  uint8x16_t *xmm_crc2, uint8x16_t *xmm_crc3)
{
    static z_const uint64_t xmm_fold128_u64[] = {
        0x0000000154442bd4U,
        0x00000001c6e41596U
    };
    z_const poly64x2_t xmm_fold4 = vld1q_p64(xmm_fold128_u64);

    poly128_t tmp0, tmp1, tmp2, tmp3;

    tmp0 = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(*xmm_crc0)), (poly64_t)vget_low_p64(xmm_fold4));
    tmp1 = vmull_high_p64(vreinterpretq_p64_u8(*xmm_crc0), xmm_fold4);

    tmp2 = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(*xmm_crc1)), (poly64_t)vget_low_p64(xmm_fold4));
    tmp3 = vmull_high_p64(vreinterpretq_p64_u8(*xmm_crc1), xmm_fold4);

    *xmm_crc0 = *xmm_crc2;
    *xmm_crc1 = *xmm_crc3;
    *xmm_crc2 = veorq_u8(vreinterpretq_u8_p128(tmp0), vreinterpretq_u8_p128(tmp1));
    *xmm_crc3 = veorq_u8(vreinterpretq_u8_p128(tmp2), vreinterpretq_u8_p128(tmp3));
}

local void fold_30(uint8x16_t *xmm_crc0, uint8x16_t *xmm_crc1,
                  uint8x16_t *xmm_crc2, uint8x16_t *xmm_crc3)
{
    static z_const uint64_t xmm_fold128_u64[] = {
        0x0000000154442bd4U,
        0x00000001c6e41596U
    };
    z_const poly64x2_t xmm_fold4 = vld1q_p64(xmm_fold128_u64);

    poly128_t tmp0, tmp1, tmp2, tmp3, tmp4, tmp5;

    tmp0 = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(*xmm_crc0)), (poly64_t)vget_low_p64(xmm_fold4));
    tmp1 = vmull_high_p64(vreinterpretq_p64_u8(*xmm_crc0), xmm_fold4);

    tmp2 = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(*xmm_crc1)), (poly64_t)vget_low_p64(xmm_fold4));
    tmp3 = vmull_high_p64(vreinterpretq_p64_u8(*xmm_crc1), xmm_fold4);

    tmp4 = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(*xmm_crc2)), (poly64_t)vget_low_p64(xmm_fold4));
    tmp5 = vmull_high_p64(vreinterpretq_p64_u8(*xmm_crc2), xmm_fold4);

    *xmm_crc0 = *xmm_crc3;
    *xmm_crc1 = veorq_u8(vreinterpretq_u8_p128(tmp0), vreinterpretq_u8_p128(tmp1));
    *xmm_crc2 = veorq_u8(vreinterpretq_u8_p128(tmp2), vreinterpretq_u8_p128(tmp3));
    *xmm_crc3 = veorq_u8(vreinterpretq_u8_p128(tmp4), vreinterpretq_u8_p128(tmp5));
}

local void fold_40(uint8x16_t *xmm_crc0, uint8x16_t *xmm_crc1,
                   uint8x16_t *xmm_crc2, uint8x16_t *xmm_crc3,
                   uint8x16_t *xmm_crc4, uint8x16_t *xmm_crc5,
                   uint8x16_t *xmm_crc6, uint8x16_t *xmm_crc7)
{
    static z_const uint64_t xmm_fold128_u64[] = {
        0x0000000154442bd4U,
        0x00000001c6e41596U
    };
    z_const poly64x2_t xmm_fold4 = vld1q_p64(xmm_fold128_u64);

    poly128_t tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
    poly128_t tmp8, tmp9, tmpA, tmpB, tmpC, tmpD, tmpE, tmpF;

    tmp0 = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(*xmm_crc0)), (poly64_t)vget_low_p64(xmm_fold4));
    tmp1 = vmull_high_p64(vreinterpretq_p64_u8(*xmm_crc0), xmm_fold4);

    tmp2 = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(*xmm_crc1)), (poly64_t)vget_low_p64(xmm_fold4));
    tmp3 = vmull_high_p64(vreinterpretq_p64_u8(*xmm_crc1), xmm_fold4);

    tmp4 = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(*xmm_crc2)), (poly64_t)vget_low_p64(xmm_fold4));
    tmp5 = vmull_high_p64(vreinterpretq_p64_u8(*xmm_crc2), xmm_fold4);

    tmp6 = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(*xmm_crc3)), (poly64_t)vget_low_p64(xmm_fold4));
    tmp7 = vmull_high_p64(vreinterpretq_p64_u8(*xmm_crc3), xmm_fold4);

    tmp8 = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(*xmm_crc4)), (poly64_t)vget_low_p64(xmm_fold4));
    tmp9 = vmull_high_p64(vreinterpretq_p64_u8(*xmm_crc4), xmm_fold4);

    tmpA = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(*xmm_crc5)), (poly64_t)vget_low_p64(xmm_fold4));
    tmpB = vmull_high_p64(vreinterpretq_p64_u8(*xmm_crc5), xmm_fold4);

    tmpC = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(*xmm_crc6)), (poly64_t)vget_low_p64(xmm_fold4));
    tmpD = vmull_high_p64(vreinterpretq_p64_u8(*xmm_crc6), xmm_fold4);

    tmpE = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(*xmm_crc7)), (poly64_t)vget_low_p64(xmm_fold4));
    tmpF = vmull_high_p64(vreinterpretq_p64_u8(*xmm_crc7), xmm_fold4);

    *xmm_crc0 = veorq_u8(vreinterpretq_u8_p128(tmp0), vreinterpretq_u8_p128(tmp1));
    *xmm_crc1 = veorq_u8(vreinterpretq_u8_p128(tmp2), vreinterpretq_u8_p128(tmp3));
    *xmm_crc2 = veorq_u8(vreinterpretq_u8_p128(tmp4), vreinterpretq_u8_p128(tmp5));
    *xmm_crc3 = veorq_u8(vreinterpretq_u8_p128(tmp6), vreinterpretq_u8_p128(tmp7));
}

local unsigned long crc_fold_1024to32(uint8x16_t xmm_crc0, uint8x16_t xmm_crc1, uint8x16_t xmm_crc2, uint8x16_t xmm_crc3 )
{
    static z_const uint64_t crc_k[] = {
        0x00000001751997d0U, /* rk2 */
        0x00000000ccaa009eU, /* rk1 */
        0x00000000ccaa009eU, /* rk5 */
        0x0000000163cd6124U, /* rk6 */
        0x00000001f7011640U, /* rk7 */
        0x00000001db710640U  /* rk8 */
    };

    static z_const uint32_t crc_mask[] = {
        0xFFFFFFFFU, 0xFFFFFFFFU, 0x00000000U, 0x00000000U,
        0x00000000U, 0xFFFFFFFFU, 0xFFFFFFFFU, 0xFFFFFFFFU
    };

    z_const uint8x16_t xmm_mask1 = vld1q_u8((uint8_t *)(crc_mask + 0));
    z_const uint8x16_t xmm_mask2 = vld1q_u8((uint8_t *)(crc_mask + 4));

    z_const poly64x2_t   crc_fold_k1k2 = vld1q_p64(crc_k + 0);
    z_const poly64x1x4_t crc_fold_k5k8 = vld1_p64_x4(crc_k + 2);

    unsigned int crc;
    poly128_t tmp0, tmp1;

    /*
     * k1
     */
    tmp0 = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(xmm_crc0)), (poly64_t)vget_low_p64(crc_fold_k1k2));
    tmp1 = vmull_high_p64(vreinterpretq_p64_u8(xmm_crc0), crc_fold_k1k2);
    xmm_crc1 = veorq_u8(xmm_crc1, vreinterpretq_u8_p128(tmp0));
    xmm_crc1 = veorq_u8(xmm_crc1, vreinterpretq_u8_p128(tmp1));

    tmp0 = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(xmm_crc1)), (poly64_t)vget_low_p64(crc_fold_k1k2));
    tmp1 = vmull_high_p64(vreinterpretq_p64_u8(xmm_crc1), crc_fold_k1k2);
    xmm_crc2 = veorq_u8(xmm_crc2, vreinterpretq_u8_p128(tmp0));
    xmm_crc2 = veorq_u8(xmm_crc2, vreinterpretq_u8_p128(tmp1));

    tmp0 = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(xmm_crc2)), (poly64_t)vget_low_p64(crc_fold_k1k2));
    tmp1 = vmull_high_p64(vreinterpretq_p64_u8(xmm_crc2), crc_fold_k1k2);
    xmm_crc3 = veorq_u8(xmm_crc3, vreinterpretq_u8_p128(tmp0));
    xmm_crc3 = veorq_u8(xmm_crc3, vreinterpretq_u8_p128(tmp1));


    /*
     * k5
     */
    xmm_crc0 = xmm_crc3;
    xmm_crc3 = vreinterpretq_u8_p128(vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(xmm_crc3)), (poly64_t)crc_fold_k5k8.val[0]));
    xmm_crc0 = vextq_u8(xmm_crc0, vdupq_n_u8(0U), 8);
    xmm_crc3 = veorq_u8(xmm_crc3, xmm_crc0);

    xmm_crc0 = xmm_crc3;
    xmm_crc3 = vextq_u8(vdupq_n_u8(0), xmm_crc3, 16 - 4);
    xmm_crc3   = vreinterpretq_u8_p128(vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(xmm_crc3)), (poly64_t)crc_fold_k5k8.val[1]));
    xmm_crc3 = veorq_u8(xmm_crc3, xmm_crc0);
    xmm_crc3 = vandq_u8(xmm_crc3, xmm_mask2);

    /*
     * k7
     */
    xmm_crc1 = xmm_crc3;
    xmm_crc2 = xmm_crc3;
    xmm_crc3 = vreinterpretq_u8_p128(vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(xmm_crc3)), (poly64_t)crc_fold_k5k8.val[2]));
    xmm_crc3 = veorq_u8(xmm_crc3, xmm_crc2);
    xmm_crc3 = vandq_u8(xmm_crc3, xmm_mask1);


    xmm_crc2 = xmm_crc3;
    xmm_crc3   = vreinterpretq_u8_p128(vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(xmm_crc3)), (poly64_t)crc_fold_k5k8.val[3]));
    xmm_crc3 = veorq_u8(xmm_crc3, xmm_crc2);
    xmm_crc3 = veorq_u8(xmm_crc3, xmm_crc1);

    crc = vgetq_lane_u32(vreinterpretq_u32_u8(xmm_crc3), 2);
    crc = ~crc;
    return (unsigned long)crc;
}

local uLong crc32_crypto_aligned128(crc, buf, len)
uLong crc;
const Bytef *buf;
uInt len;
{
    uint8x16_t xmm_crc0 = vld1q_u8(buf +   0);
    uint8x16_t xmm_crc1 = vld1q_u8(buf +  16);
    uint8x16_t xmm_crc2 = vld1q_u8(buf +  32);
    uint8x16_t xmm_crc3 = vld1q_u8(buf +  48);
    uint8x16_t xmm_crc4 = vld1q_u8(buf +  64);
    uint8x16_t xmm_crc5 = vld1q_u8(buf +  80);
    uint8x16_t xmm_crc6 = vld1q_u8(buf +  96);
    uint8x16_t xmm_crc7 = vld1q_u8(buf + 112);
    uint8x16_t c = vreinterpretq_u8_u32(vsetq_lane_u32((z_crc_t)crc, vdupq_n_u32(0U), 0));

    len -= 128U;
    buf += 128;

    xmm_crc0 = veorq_u8(xmm_crc0, c);

    while (len >= 128U) {
        fold_40(&xmm_crc0, &xmm_crc1, &xmm_crc2, &xmm_crc3, &xmm_crc4, &xmm_crc5, &xmm_crc6, &xmm_crc7);

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

    if (len >= 48U) {
        fold_30(&xmm_crc0, &xmm_crc1, &xmm_crc2, &xmm_crc3);

        xmm_crc1 = veorq_u8(xmm_crc1, vld1q_u8(buf +  0));
        xmm_crc2 = veorq_u8(xmm_crc2, vld1q_u8(buf + 16));
        xmm_crc3 = veorq_u8(xmm_crc3, vld1q_u8(buf + 32));

        len -= 48U;
        buf += 48;
    } else if (len >= 32U) {
        fold_20(&xmm_crc0, &xmm_crc1, &xmm_crc2, &xmm_crc3);

        xmm_crc2 = veorq_u8(xmm_crc2, vld1q_u8(buf +  0));
        xmm_crc3 = veorq_u8(xmm_crc3, vld1q_u8(buf + 16));

        len -= 32U;
        buf += 32;
    } else if (len >= 16U) {
        fold_10(&xmm_crc0, &xmm_crc1, &xmm_crc2, &xmm_crc3);
        
        xmm_crc3 = veorq_u8(xmm_crc3, vld1q_u8(buf +  0));
        
        len -= 16U;
        buf += 16;
    }
    xmm_crc0 = veorq_u8(xmm_crc0, xmm_crc4);
    xmm_crc1 = veorq_u8(xmm_crc1, xmm_crc5);
    xmm_crc2 = veorq_u8(xmm_crc2, xmm_crc6);
    xmm_crc3 = veorq_u8(xmm_crc3, xmm_crc7);
    return crc_fold_1024to32(xmm_crc0, xmm_crc1, xmm_crc2, xmm_crc3);
}

local void fold_1(uint8x16_t *xmm_crc0, uint8x16_t *xmm_crc1,
                  uint8x16_t *xmm_crc2, uint8x16_t *xmm_crc3)
{
    static z_const uint64_t xmm_fold128_u64[] = {
        0x0000000154442bd4U,
        0x00000001c6e41596U
    };
    z_const poly64x2_t xmm_fold4 = vld1q_p64(xmm_fold128_u64);

    poly128_t tmp0, tmp1;

    tmp0 = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(*xmm_crc0)), (poly64_t)vget_low_p64(xmm_fold4));
    tmp1 = vmull_high_p64(vreinterpretq_p64_u8(*xmm_crc0), xmm_fold4);

    *xmm_crc0 = *xmm_crc1;
    *xmm_crc1 = *xmm_crc2;
    *xmm_crc2 = *xmm_crc3;
    *xmm_crc3 = veorq_u8(vreinterpretq_u8_p128(tmp0), vreinterpretq_u8_p128(tmp1));
}

local void fold_2(uint8x16_t *xmm_crc0, uint8x16_t *xmm_crc1,
                  uint8x16_t *xmm_crc2, uint8x16_t *xmm_crc3)
{
    static z_const uint64_t xmm_fold128_u64[] = {
        0x0000000154442bd4U,
        0x00000001c6e41596U
    };
    z_const poly64x2_t xmm_fold4 = vld1q_p64(xmm_fold128_u64);

    poly128_t tmp0, tmp1, tmp2, tmp3;

    tmp0 = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(*xmm_crc0)), (poly64_t)vget_low_p64(xmm_fold4));
    tmp1 = vmull_high_p64(vreinterpretq_p64_u8(*xmm_crc0), xmm_fold4);

    tmp2 = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(*xmm_crc1)), (poly64_t)vget_low_p64(xmm_fold4));
    tmp3 = vmull_high_p64(vreinterpretq_p64_u8(*xmm_crc1), xmm_fold4);

    *xmm_crc0 = *xmm_crc2;
    *xmm_crc1 = *xmm_crc3;
    *xmm_crc2 = veorq_u8(vreinterpretq_u8_p128(tmp0), vreinterpretq_u8_p128(tmp1));
    *xmm_crc3 = veorq_u8(vreinterpretq_u8_p128(tmp2), vreinterpretq_u8_p128(tmp3));
}

local void fold_3(uint8x16_t *xmm_crc0, uint8x16_t *xmm_crc1,
                  uint8x16_t *xmm_crc2, uint8x16_t *xmm_crc3)
{
    static z_const uint64_t xmm_fold128_u64[] = {
        0x0000000154442bd4U,
        0x00000001c6e41596U
    };
    z_const poly64x2_t xmm_fold4 = vld1q_p64(xmm_fold128_u64);

    poly128_t tmp0, tmp1, tmp2, tmp3, tmp4, tmp5;

    tmp0 = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(*xmm_crc0)), (poly64_t)vget_low_p64(xmm_fold4));
    tmp1 = vmull_high_p64(vreinterpretq_p64_u8(*xmm_crc0), xmm_fold4);

    tmp2 = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(*xmm_crc1)), (poly64_t)vget_low_p64(xmm_fold4));
    tmp3 = vmull_high_p64(vreinterpretq_p64_u8(*xmm_crc1), xmm_fold4);

    tmp4 = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(*xmm_crc2)), (poly64_t)vget_low_p64(xmm_fold4));
    tmp5 = vmull_high_p64(vreinterpretq_p64_u8(*xmm_crc2), xmm_fold4);

    *xmm_crc0 = *xmm_crc3;
    *xmm_crc1 = veorq_u8(vreinterpretq_u8_p128(tmp0), vreinterpretq_u8_p128(tmp1));
    *xmm_crc2 = veorq_u8(vreinterpretq_u8_p128(tmp2), vreinterpretq_u8_p128(tmp3));
    *xmm_crc3 = veorq_u8(vreinterpretq_u8_p128(tmp4), vreinterpretq_u8_p128(tmp5));
}

local void fold_4(uint8x16_t *xmm_crc0, uint8x16_t *xmm_crc1,
                  uint8x16_t *xmm_crc2, uint8x16_t *xmm_crc3)
{
    static z_const uint64_t xmm_fold128_u64[] = {
        0x0000000154442bd4U,
        0x00000001c6e41596U
    };
    z_const poly64x2_t xmm_fold4 = vld1q_p64(xmm_fold128_u64);

    poly128_t tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;

    tmp0 = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(*xmm_crc0)), (poly64_t)vget_low_p64(xmm_fold4));
    tmp1 = vmull_high_p64(vreinterpretq_p64_u8(*xmm_crc0), xmm_fold4);

    tmp2 = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(*xmm_crc1)), (poly64_t)vget_low_p64(xmm_fold4));
    tmp3 = vmull_high_p64(vreinterpretq_p64_u8(*xmm_crc1), xmm_fold4);

    tmp4 = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(*xmm_crc2)), (poly64_t)vget_low_p64(xmm_fold4));
    tmp5 = vmull_high_p64(vreinterpretq_p64_u8(*xmm_crc2), xmm_fold4);

    tmp6 = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(*xmm_crc3)), (poly64_t)vget_low_p64(xmm_fold4));
    tmp7 = vmull_high_p64(vreinterpretq_p64_u8(*xmm_crc3), xmm_fold4);

    *xmm_crc0 = veorq_u8(vreinterpretq_u8_p128(tmp0), vreinterpretq_u8_p128(tmp1));
    *xmm_crc1 = veorq_u8(vreinterpretq_u8_p128(tmp2), vreinterpretq_u8_p128(tmp3));
    *xmm_crc2 = veorq_u8(vreinterpretq_u8_p128(tmp4), vreinterpretq_u8_p128(tmp5));
    *xmm_crc3 = veorq_u8(vreinterpretq_u8_p128(tmp6), vreinterpretq_u8_p128(tmp7));
}

local unsigned long crc_fold_512to32(uint8x16_t xmm_crc0, uint8x16_t xmm_crc1, uint8x16_t xmm_crc2, uint8x16_t xmm_crc3 )
{
    static z_const uint64_t crc_k[] = {
        0x00000001751997d0U, /* rk2 */
        0x00000000ccaa009eU, /* rk1 */
        0x00000000ccaa009eU, /* rk5 */
        0x0000000163cd6124U, /* rk6 */
        0x00000001f7011640U, /* rk7 */
        0x00000001db710640U  /* rk8 */
    };

    static z_const uint32_t crc_mask[] = {
        0xFFFFFFFFU, 0xFFFFFFFFU, 0x00000000U, 0x00000000U,
        0x00000000U, 0xFFFFFFFFU, 0xFFFFFFFFU, 0xFFFFFFFFU
    };

    z_const uint8x16_t xmm_mask1 = vld1q_u8((uint8_t *)(crc_mask + 0));
    z_const uint8x16_t xmm_mask2 = vld1q_u8((uint8_t *)(crc_mask + 4));

    z_const poly64x2_t   crc_fold_k1k2 = vld1q_p64(crc_k + 0);
    z_const poly64x1x4_t crc_fold_k5k8 = vld1_p64_x4(crc_k + 2);

    unsigned int crc;
    poly128_t tmp0, tmp1;

    /*
     * k1
     */
    tmp0 = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(xmm_crc0)), (poly64_t)vget_low_p64(crc_fold_k1k2));
    tmp1 = vmull_high_p64(vreinterpretq_p64_u8(xmm_crc0), crc_fold_k1k2);
    xmm_crc1 = veorq_u8(xmm_crc1, vreinterpretq_u8_p128(tmp0));
    xmm_crc1 = veorq_u8(xmm_crc1, vreinterpretq_u8_p128(tmp1));

    tmp0 = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(xmm_crc1)), (poly64_t)vget_low_p64(crc_fold_k1k2));
    tmp1 = vmull_high_p64(vreinterpretq_p64_u8(xmm_crc1), crc_fold_k1k2);
    xmm_crc2 = veorq_u8(xmm_crc2, vreinterpretq_u8_p128(tmp0));
    xmm_crc2 = veorq_u8(xmm_crc2, vreinterpretq_u8_p128(tmp1));

    tmp0 = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(xmm_crc2)), (poly64_t)vget_low_p64(crc_fold_k1k2));
    tmp1 = vmull_high_p64(vreinterpretq_p64_u8(xmm_crc2), crc_fold_k1k2);
    xmm_crc3 = veorq_u8(xmm_crc3, vreinterpretq_u8_p128(tmp0));
    xmm_crc3 = veorq_u8(xmm_crc3, vreinterpretq_u8_p128(tmp1));


    /*
     * k5
     */
    xmm_crc0 = xmm_crc3;
    xmm_crc3 = vreinterpretq_u8_p128(vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(xmm_crc3)), (poly64_t)crc_fold_k5k8.val[0]));
    xmm_crc0 = vextq_u8(xmm_crc0, vdupq_n_u8(0U), 8);
    xmm_crc3 = veorq_u8(xmm_crc3, xmm_crc0);

    xmm_crc0 = xmm_crc3;
    xmm_crc3 = vextq_u8(vdupq_n_u8(0), xmm_crc3, 16 - 4);
    xmm_crc3   = vreinterpretq_u8_p128(vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(xmm_crc3)), (poly64_t)crc_fold_k5k8.val[1]));
    xmm_crc3 = veorq_u8(xmm_crc3, xmm_crc0);
    xmm_crc3 = vandq_u8(xmm_crc3, xmm_mask2);

    /*
     * k7
     */
    xmm_crc1 = xmm_crc3;
    xmm_crc2 = xmm_crc3;
    xmm_crc3 = vreinterpretq_u8_p128(vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(xmm_crc3)), (poly64_t)crc_fold_k5k8.val[2]));
    xmm_crc3 = veorq_u8(xmm_crc3, xmm_crc2);
    xmm_crc3 = vandq_u8(xmm_crc3, xmm_mask1);


    xmm_crc2 = xmm_crc3;
    xmm_crc3   = vreinterpretq_u8_p128(vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(xmm_crc3)), (poly64_t)crc_fold_k5k8.val[3]));
    xmm_crc3 = veorq_u8(xmm_crc3, xmm_crc2);
    xmm_crc3 = veorq_u8(xmm_crc3, xmm_crc1);

    crc = vgetq_lane_u32(vreinterpretq_u32_u8(xmm_crc3), 2);
    crc = ~crc;
    return (unsigned long)crc;
}

local uLong crc32_crypto_aligned64(crc, buf, len)
uLong crc;
const Bytef *buf;
uInt len;
{
    uint8x16_t xmm_crc0 = vld1q_u8(buf +  0);
    uint8x16_t xmm_crc1 = vld1q_u8(buf + 16);
    uint8x16_t xmm_crc2 = vld1q_u8(buf + 32);
    uint8x16_t xmm_crc3 = vld1q_u8(buf + 48);
    uint8x16_t c = vreinterpretq_u8_u32(vsetq_lane_u32((z_crc_t)crc, vdupq_n_u32(0U), 0));

    len -= 64U;
    buf += 64;

    xmm_crc0 = veorq_u8(xmm_crc0, c);

    while (len >= 64U) {
        fold_4(&xmm_crc0, &xmm_crc1, &xmm_crc2, &xmm_crc3);

        xmm_crc0 = veorq_u8(xmm_crc0, vld1q_u8(buf +  0));
        xmm_crc1 = veorq_u8(xmm_crc1, vld1q_u8(buf + 16));
        xmm_crc2 = veorq_u8(xmm_crc2, vld1q_u8(buf + 32));
        xmm_crc3 = veorq_u8(xmm_crc3, vld1q_u8(buf + 48));

        len -= 64U;
        buf += 64;
    }

    if (len >= 48U) {
        fold_3(&xmm_crc0, &xmm_crc1, &xmm_crc2, &xmm_crc3);

        xmm_crc1 = veorq_u8(xmm_crc1, vld1q_u8(buf +  0));
        xmm_crc2 = veorq_u8(xmm_crc2, vld1q_u8(buf + 16));
        xmm_crc3 = veorq_u8(xmm_crc3, vld1q_u8(buf + 32));

        len -= 48U;
        buf += 48;
    } else if (len >= 32U) {
        fold_2(&xmm_crc0, &xmm_crc1, &xmm_crc2, &xmm_crc3);

        xmm_crc2 = veorq_u8(xmm_crc2, vld1q_u8(buf +  0));
        xmm_crc3 = veorq_u8(xmm_crc3, vld1q_u8(buf + 16));

        len -= 32U;
        buf += 32;
    } else if (len >= 16U) {
        fold_1(&xmm_crc0, &xmm_crc1, &xmm_crc2, &xmm_crc3);

        xmm_crc3 = veorq_u8(xmm_crc3, vld1q_u8(buf +  0));

        len -= 16U;
        buf += 16;
    }
    return crc_fold_512to32(xmm_crc0, xmm_crc1, xmm_crc2, xmm_crc3);
}

local void fold128(uint8x16_t *xmm_crc0)
{
    static z_const uint64_t xmm_fold128_u64[] = {
        0x00000001751997d0U,
        0x00000000ccaa009eU
    };
    z_const poly64x2_t xmm_fold128 = vld1q_p64(xmm_fold128_u64);

    poly128_t tmp0;
    poly128_t tmp1;

    tmp0 = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u8(*xmm_crc0)), (poly64_t)vget_low_p64(xmm_fold128));
    tmp1 = vmull_high_p64(vreinterpretq_p64_u8(*xmm_crc0), xmm_fold128);

    *xmm_crc0 = veorq_u8(vreinterpretq_u8_p128(tmp0), vreinterpretq_u8_p128(tmp1));
}

local unsigned long crc_fold_128to32(uint8x16_t xmm_crc0)
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

local uLong crc32_crypto_aligned16(crc, buf, len)
uLong crc;
const Bytef *buf;
uInt len;
{
    uint8x16_t xmm_crc0 = vld1q_u8(buf +  0);
    uint8x16_t c = vreinterpretq_u8_u32(vsetq_lane_u32((z_crc_t)crc, vdupq_n_u32(0U), 0));

    len -= 16U;
    buf += 16;

    xmm_crc0 = veorq_u8(xmm_crc0, c);

    if (len >= 16) {
        do
        {
            fold128(&xmm_crc0);

            xmm_crc0 = veorq_u8(xmm_crc0, vld1q_u8(buf +  0));

            len -= 16U;
            buf += 16;
        } while (len >= 16U);
    }
    return crc_fold_128to32(xmm_crc0);
}

ZLIB_INTERNAL uLong crc32_generic OF((uLong crc, const Bytef *buf, uInt len));

ZLIB_INTERNAL uLong crc32_crypto(crc, buf, len)
uLong crc;
const Bytef *buf;
uInt len;
{
    if (len < 16) {
        return crc32_generic(crc, buf, len);
    }
    /* TODO */
    /*
    if (len >= 128) {
        crc = crc32_crypto_aligned128(~crc, buf, len & ~(uInt)15U);
        buf += len & ~(uInt)15U;
        len &= 15U;
    }
    else */
    if (len >= 64) {
        crc = crc32_crypto_aligned64(~crc, buf, len & ~(uInt)15U);
        buf += len & ~(uInt)15U;
        len &= 15U;
    }
    else if (len >= 16U) {
        crc = crc32_crypto_aligned16(~crc, buf, len & ~(uInt)15U);
        buf += len & ~(uInt)15U;
        len &= 15U;
    }
    if (len > 0) {
        crc = crc32_generic(crc, buf, len);
    }
    return crc;
}

