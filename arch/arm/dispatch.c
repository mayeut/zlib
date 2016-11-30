/* arch/arm/dispatch.c -- dispatch ARM optimized functions
 * Copyright (C) 2016 Matthieu Darbois
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

/* @(#) $Id$ */

#include "zutil.h"
#include "deflate.h"

/* flags definitions */
#define ZL_INITIALIZED 0x0001U
#define ZL_NEON        0x0002U

local unsigned int zl_get_feature_flags_once OF(());
local unsigned int zl_get_feature_flags OF(());
local volatile unsigned int zl_feature_flags = 0U;

local uLong zl_adler32_dispatch_init     OF((uLong adler, const Bytef *buf, uInt len));
local uLong zl_crc32_dispatch_init       OF((uLong crc,   const Bytef *buf, uInt len));
local void  zl_fill_window_dispatch_init OF((deflate_state *s));

typedef uLong (*zl_adler32_func)     OF((uLong adler, const Bytef *buf, uInt len));
typedef uLong (*zl_crc32_func)       OF((uLong crc,   const Bytef *buf, uInt len));
typedef void  (*zl_fill_window_func) OF((deflate_state *s));

local volatile zl_adler32_func     zl_adler32_dispatch     = zl_adler32_dispatch_init;
local volatile zl_crc32_func       zl_crc32_dispatch       = zl_crc32_dispatch_init;
local volatile zl_fill_window_func zl_fill_window_dispatch = zl_fill_window_dispatch_init;

local unsigned int zl_get_feature_flags_once()
{
  unsigned int result = ZL_INITIALIZED;

	result |= ZL_NEON;

  return result;
}

local unsigned int zl_get_feature_flags()
{
  /* TODO atomic read */
  unsigned int result = zl_feature_flags;

  if (result == 0U) {
    result = zl_get_feature_flags_once();
    /* TODO atomic operation */
    zl_feature_flags = result;
  }

  return result;
}

ZLIB_INTERNAL uLong adler32_generic OF((uLong adler, const Bytef *buf, uInt len));
ZLIB_INTERNAL uLong adler32_neon    OF((uLong adler, const Bytef *buf, uInt len));

local uLong zl_adler32_dispatch_init(adler, buf, len)
	uLong adler;
	const Bytef *buf;
	uInt len;
{
  zl_adler32_func function = adler32_generic;
  unsigned int features = zl_get_feature_flags();

  if (features & ZL_NEON) {
    function = adler32_neon;
  }

  /* TODO atomic */
  zl_adler32_dispatch = function;
  return function(adler, buf, len);
}
ZLIB_INTERNAL uLong adler32_dispatch(adler, buf, len)
	uLong adler;
	const Bytef *buf;
	uInt len;
{
  /* TODO atomic read */
  return zl_adler32_dispatch(adler, buf, len);
}

ZLIB_INTERNAL uLong crc32_generic OF((uLong crc, const Bytef *buf, uInt len));

local uLong zl_crc32_dispatch_init(crc, buf, len)
	uLong crc;
	const Bytef *buf;
	uInt len;
{
  zl_crc32_func function = crc32_generic;
  //unsigned int features = zl_get_feature_flags();

  /* TODO atomic */
  zl_crc32_dispatch = function;
  return function(crc, buf, len);
}

ZLIB_INTERNAL uLong crc32_dispatch(adler, buf, len)
	uLong adler;
	const Bytef *buf;
	uInt len;
{
  /* TODO atomic read */
  return zl_crc32_dispatch(adler, buf, len);
}


ZLIB_INTERNAL void fill_window_generic OF((deflate_state *s));

local void zl_fill_window_dispatch_init(s)
deflate_state *s;
{
	zl_fill_window_func function = fill_window_generic;
	//unsigned int features = zl_get_feature_flags();

	/*if (features & ZL_NEON) {
		function = fill_window_neon;
	}*/

	/* TODO atomic */
	zl_fill_window_dispatch = function;
	return function(s);
}

ZLIB_INTERNAL void fill_window_dispatch(s)
	deflate_state *s;
{
	zl_fill_window_dispatch(s);
}
