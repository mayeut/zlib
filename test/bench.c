/* bench.c -- a few benchmarks
 * Copyright (C) 2016 Matthieu Darbois
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

/* @(#) $Id$ */

#include "zlib.h"
#include <stdio.h>
#include <float.h>
#include <dlfcn.h>

const size_t maxChunkSize = 32768U;
const size_t totalSize = maxChunkSize * 256U;
const size_t iterations = totalSize / maxChunkSize;

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
typedef LARGE_INTEGER z_timespec;
#elif defined(__MACH__)
#include <mach/mach_time.h>
typedef uint64_t z_timespec;
#else
#include <time.h>
typedef struct timespec z_timespec;
#endif

/* timing functions */
static void
z_gettime (z_timespec * o_time)
{
#if defined(_WIN32) || defined(_WIN64)
	QueryPerformanceCounter(o_time);
#elif defined(__MACH__)
	*o_time = mach_absolute_time();
#else
	clock_gettime(CLOCK_MONOTONIC, o_time);
#endif
}

static double
z_timediff_sec (z_timespec *start, z_timespec *end)
{
#if defined(_WIN32) || defined(_WIN64)
	LARGE_INTEGER frequency;
	QueryPerformanceFrequency(&frequency);
	return (end.QuadPart - start.QuadPart) / (double)frequency.QuadPart;
#elif defined(__MACH__)
	uint64_t diff = *end - *start;
	mach_timebase_info_data_t tb = { 0, 0 };
	mach_timebase_info(&tb);
	return (double)((diff * tb.numer) / tb.denom) / 1e9;
#else
	return (end->tv_sec - start->tv_sec) + ((double)(end->tv_nsec - start->tv_nsec)) / 1e9;
#endif
}

int main(int argc, char** argv)
{
	int result = 0;
	static Byte chunk[maxChunkSize];
	size_t i, j, chunkSize, chunkIter;
	z_timespec start, end;
	void* sysZlib = Z_NULL;
	uLong (*sysAdler32) OF((uLong adler, const Bytef *buf, uInt len)) = Z_NULL;
	uLong (*sysCrc32) OF((uLong adler, const Bytef *buf, uInt len)) = Z_NULL;

	uLong randomNumber = 0x27121978U;
	// let's fill chunks using a Linear congruential generator
	for (i = 0; i < maxChunkSize; i++)
	{
		chunk[i] = (Byte)randomNumber;
		randomNumber = (1664525U * randomNumber + 1013904223U) & 0xFFFFFFFFU;
	}

	sysZlib = dlopen("libz.1.dylib", RTLD_NOW | RTLD_LOCAL);
	if (sysZlib != NULL) {
		sysAdler32  = (uLong (*) OF((uLong adler, const Bytef *buf, uInt len)))dlsym(sysZlib, "adler32");
		sysCrc32  = (uLong (*) OF((uLong adler, const Bytef *buf, uInt len)))dlsym(sysZlib, "crc32");
	}

	/* non regression */
	/* adler32 */
	{
		chunkSize = (8192U <= (maxChunkSize - 64U)) ? 8192U : (maxChunkSize - 64U);

		while (chunkSize > 0U)
		{
			uInt  misalign;
			for (misalign = 0U; misalign < 32U; ++misalign)
			{
				uLong adler;
				uLong sysAdler;

				adler = adler32(0U, Z_NULL, 0U);
				sysAdler = sysAdler32(0U, Z_NULL, 0U);
				adler = adler32(adler, chunk + misalign, (uInt)chunkSize);
				sysAdler = sysAdler32(sysAdler, chunk + misalign, (uInt)chunkSize);

				if (adler != sysAdler) {
					printf("adler32   , %2u misalign, %5u bytes: %08lX\n", misalign, (uInt)chunkSize, adler);
					printf("sysAdler32, %2u misalign, %5u bytes: %08lX\n", misalign, (uInt)chunkSize, sysAdler);
					result = 1;
				}
			}
			--chunkSize;
		}
	}
	/* crc32 */
	{
		chunkSize = (8192U <= (maxChunkSize - 64U)) ? 8192U : (maxChunkSize - 64U);

		while (chunkSize > 0U)
		{
			uInt  misalign;
			for (misalign = 0U; misalign < 32U; ++misalign)
			{
				uLong crc;
				uLong sysCrc;

				crc = crc32(0U, Z_NULL, 0U);
				sysCrc = sysCrc32(0U, Z_NULL, 0U);
				crc = crc32(crc, chunk + misalign, (uInt)chunkSize);
				sysCrc = sysCrc32(sysCrc, chunk + misalign, (uInt)chunkSize);

				if (crc != sysCrc) {
					printf("crc32   , %2u misalign, %5u bytes: %08lX\n", misalign, (uInt)chunkSize, crc);
					printf("sysCrc32, %2u misalign, %5u bytes: %08lX\n", misalign, (uInt)chunkSize, sysCrc);
					result = 1;
				}
			}
			--chunkSize;
		}
	}

	/* bench */

	for (chunkSize = 2U, chunkIter = iterations; chunkSize != 2U; chunkSize >>= 1, chunkIter <<= 1) {
		double diff = DBL_MAX;
		double sysDiff = DBL_MAX;
		uLong adler = 0U, sysAdler = 0U;

		for (j = 0U;j < 256U; ++j) {
			double onediff;

			adler = adler32(0U, NULL, 0U);
			z_gettime(&start);
			for (i = chunkIter; i != 0U; --i) {
				adler = adler32(adler, chunk, (uInt)chunkSize);
			}
			z_gettime(&end);
			onediff = z_timediff_sec(&start, &end);
			if (onediff < diff) diff = onediff;

			sysAdler = sysAdler32(0U, NULL, 0U);
			z_gettime(&start);
			for (i = chunkIter; i != 0U; --i) {
				sysAdler = sysAdler32(sysAdler, chunk, (uInt)chunkSize);
			}
			z_gettime(&end);
			onediff = z_timediff_sec(&start, &end);
			if (onediff < sysDiff) sysDiff = onediff;
		}
		printf("adler32    %5u bytes: %08lX at %8.2f MB/s\n", (uInt)chunkSize,    adler, totalSize /    diff / 1024.0 /1024.0);
		printf("sysAdler32 %5u bytes: %08lX at %8.2f MB/s\n", (uInt)chunkSize, sysAdler, totalSize / sysDiff / 1024.0 /1024.0);
	}
	
	for (chunkSize = maxChunkSize, chunkIter = iterations; chunkSize != 2U; chunkSize >>= 1, chunkIter <<= 1) {
		double diff = DBL_MAX;
		double sysDiff = DBL_MAX;
		uLong crc = 0U, sysCrc = 0U;

		for (j = 0U;j < 256U; ++j) {
			double onediff;

			crc = crc32(0U, NULL, 0U);
			z_gettime(&start);
			for (i = chunkIter; i != 0U; --i) {
				crc = crc32(crc, chunk, (uInt)chunkSize);
			}
			z_gettime(&end);
			onediff = z_timediff_sec(&start, &end);
			if (onediff < diff) diff = onediff;

			sysCrc = sysCrc32(0U, NULL, 0U);
			z_gettime(&start);
			for (i = chunkIter; i != 0U; --i) {
				sysCrc = sysCrc32(sysCrc, chunk, (uInt)chunkSize);
			}
			z_gettime(&end);
			onediff = z_timediff_sec(&start, &end);
			if (onediff < sysDiff) sysDiff = onediff;
		}
		printf("crc32    %5u bytes: %08lX at %6.2f MB/s\n", (uInt)chunkSize,    crc, totalSize /    diff / 1024.0 /1024.0);
		printf("sysCrc32 %5u bytes: %08lX at %6.2f MB/s\n", (uInt)chunkSize, sysCrc, totalSize / sysDiff / 1024.0 /1024.0);
	}
	return result;
}
