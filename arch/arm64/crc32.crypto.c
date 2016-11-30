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
#include <arm_acle.h>

local void fold128(uint8x16_t *xmm_crc0)
{
    static z_const uint64_t xmm_fold128_u64[] = {
        0x00000001751997d0U,
        0x000000000ccaa009eU
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

    xmm_crc0 = vreinterpretq_u8_u64(vzip1q_u64(vreinterpretq_u64_u8(xmm_crc0), vdupq_n_u64(0U)));
    xmm_crc0 = veorq_u8(xmm_crc0, x_tmp0);

    x_tmp0 = vreinterpretq_u8_u32(vcopyq_lane_u32(vdupq_n_u32(0U), 0, vget_low_u32(vreinterpretq_u32_u8(xmm_crc0)), 0));

    xmm_crc0 = vreinterpretq_u8_u64(vshrq_n_u64(vreinterpretq_u64_u8(xmm_crc0), 32));
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
    if (len >= 16U) {
        crc = crc32_crypto_aligned16(~crc, buf, len & ~(uInt)15U);
        buf += len & ~(uInt)15U;
        if (len & 15U) {
            return crc32_generic(crc, buf, len & 15U);
        }
        return crc;
    }
    return crc32_generic(crc, buf, len);
}

