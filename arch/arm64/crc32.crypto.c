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
        crc = fold_128to32(xmm_crc0);
        if (len > 0) {
            crc = crc32_generic(crc, buf, len);
        }
    }
    return crc;
}

