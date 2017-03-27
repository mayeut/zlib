/* bench.c -- a few benchmarks
 * Copyright (C) 2016 Matthieu Darbois
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

/* @(#) $Id$ */

#include "zlib.h"
#include "zutil.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <float.h>
#include <dlfcn.h>

static const size_t maxChunkSize = 32768U;
static const size_t totalSize = maxChunkSize * 256U;
static const size_t iterations = totalSize / maxChunkSize;

static Byte chunk[maxChunkSize];
static Byte chunkDst[maxChunkSize];

static uLong (*sysAdler32) OF((uLong adler, const Bytef *buf, uInt len)) = Z_NULL;
static uLong (*sysCrc32) OF((uLong crc, const Bytef *buf, uInt len)) = Z_NULL;

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

static uLong sysCrc32WithCopy(uLong crc, const Bytef *buf, uInt len, Bytef *dst) {
	memcpy(dst, buf, len);
	return sysCrc32(crc, buf, len);
}

static uLong sysAdler32WithCopy(uLong crc, const Bytef *buf, uInt len, Bytef *dst) {
	memcpy(dst, buf, len);
	return sysAdler32(crc, buf, len);
}

static int unit_adler32()
{
	int result = 0;
	size_t chunkSize = (8192U <= (maxChunkSize - 64U)) ? 8192U : (maxChunkSize - 64U);

	while (chunkSize > 0U)
	{
		uInt  misalign;
		for (misalign = 0U; misalign < 32U; ++misalign)
		{
			uLong adler;
			uLong adlerWithCopy;
			uLong sysAdler;

			adler = adler32(0U, Z_NULL, 0U);
			adlerWithCopy = adler32(0U, Z_NULL, 0U);
			sysAdler = sysAdler32(0U, Z_NULL, 0U);
			adler = adler32(adler, chunk + misalign, (uInt)chunkSize);
			adlerWithCopy = adler32_copy(adlerWithCopy, chunk + misalign, (uInt)chunkSize, chunkDst);
			sysAdler = sysAdler32(sysAdler, chunk + misalign, (uInt)chunkSize);

			if ((adler != sysAdler) || (adlerWithCopy != sysAdler)) {
				printf("adler32,      %2u misalign, %5u bytes: %08lX\n", misalign, (uInt)chunkSize, adler);
				printf("adler32_copy, %2u misalign, %5u bytes: %08lX\n", misalign, (uInt)chunkSize, adlerWithCopy);
				printf("sysAdler32,   %2u misalign, %5u bytes: %08lX\n", misalign, (uInt)chunkSize, sysAdler);
				result = 1;
				chunkSize = 1U;
				break;
			}
			if (memcmp(chunk + misalign, chunkDst, chunkSize) != 0) {
				printf("adler32_copy, %2u misalign, %5u bytes: copy error\n", misalign, (uInt)chunkSize);
				result = 1;
				chunkSize = 1U;
				break;
			}
		}
		--chunkSize;
	}
	return result;
}

static int unit_crc32()
{
	int result = 0;
	size_t chunkSize = (8192U <= (maxChunkSize - 64U)) ? 8192U : (maxChunkSize - 64U);

	while (chunkSize > 0U)
	{
		uInt  misalign;
		for (misalign = 0U; misalign < 32U; ++misalign)
		{
			uLong crc;
			uLong crcWithCopy;
			uLong sysCrc;

			crc = crc32(0U, Z_NULL, 0U);
			crcWithCopy = crc32(0U, Z_NULL, 0U);
			sysCrc = sysCrc32(0U, Z_NULL, 0U);
			crc = crc32(crc, chunk + misalign, (uInt)chunkSize);
			crcWithCopy = crc32_copy(crcWithCopy, chunk + misalign, (uInt)chunkSize, chunkDst);
			sysCrc = sysCrc32(sysCrc, chunk + misalign, (uInt)chunkSize);

			if ((crc != sysCrc) || (crcWithCopy != sysCrc)) {
				printf("crc32,      %2u misalign, %5u bytes: %08lX\n", misalign, (uInt)chunkSize, crc);
				printf("crc32_copy, %2u misalign, %5u bytes: %08lX\n", misalign, (uInt)chunkSize, crcWithCopy);
				printf("sysCrc32,   %2u misalign, %5u bytes: %08lX\n", misalign, (uInt)chunkSize, sysCrc);
				result = 1;
				chunkSize = 1U;
				break;
			}
			if (memcmp(chunk + misalign, chunkDst, chunkSize) != 0) {
				printf("crc2_copy, %2u misalign, %5u bytes: copy error\n", misalign, (uInt)chunkSize);
				result = 1;
				chunkSize = 1U;
				break;
			}
		}
		--chunkSize;
	}
	return result;
}

static void bench_adler32_copy()
{
	size_t chunkSize, chunkIter;

	for (chunkSize = maxChunkSize, chunkIter = iterations; chunkSize != 2U; chunkSize >>= 1, chunkIter <<= 1) {
		double diff = DBL_MAX;
		double sysDiff = DBL_MAX;
		uLong adler = 0U, sysAdler = 0U;
		size_t j;

		for (j = 0U;j < 256U; ++j) {
			z_timespec start, end;
			double onediff;
			size_t i;

			adler = adler32(0U, NULL, 0U);
			z_gettime(&start);
			for (i = chunkIter; i != 0U; --i) {
				adler = adler32_copy(adler, chunk, (uInt)chunkSize - 1U, chunkDst);
			}
			z_gettime(&end);
			onediff = z_timediff_sec(&start, &end);
			if (onediff < diff) diff = onediff;

			if (sysAdler32 != Z_NULL) {
				sysAdler = sysAdler32(0U, NULL, 0U);
				z_gettime(&start);
				for (i = chunkIter; i != 0U; --i) {
					sysAdler = sysAdler32WithCopy(sysAdler, chunk, (uInt)chunkSize - 1U, chunkDst);
				}
				z_gettime(&end);
				onediff = z_timediff_sec(&start, &end);
				if (onediff < sysDiff) sysDiff = onediff;
			}
		}
		printf("adler32_copy       %5u bytes: %08lX at %8.2f MB/s\n", (uInt)chunkSize - 1U,    adler, totalSize /    diff / 1024.0 /1024.0);
		if (sysAdler32 != Z_NULL) {
			printf("sysAdler32WithCopy %5u bytes: %08lX at %8.2f MB/s\n", (uInt)chunkSize - 1U, sysAdler, totalSize / sysDiff / 1024.0 /1024.0);
		}
	}
}

static void bench_adler32()
{
	size_t chunkSize, chunkIter;

	for (chunkSize = maxChunkSize, chunkIter = iterations; chunkSize != 2U; chunkSize >>= 1, chunkIter <<= 1) {
		double diff = DBL_MAX;
		double sysDiff = DBL_MAX;
		uLong adler = 0U, sysAdler = 0U;
		size_t j;

		for (j = 0U;j < 256U; ++j) {
			z_timespec start, end;
			double onediff;
			size_t i;

			adler = adler32(0U, NULL, 0U);
			z_gettime(&start);
			for (i = chunkIter; i != 0U; --i) {
				adler = adler32(adler, chunk, (uInt)chunkSize - 1U);
			}
			z_gettime(&end);
			onediff = z_timediff_sec(&start, &end);
			if (onediff < diff) diff = onediff;

			if (sysAdler32 != Z_NULL) {
				sysAdler = sysAdler32(0U, NULL, 0U);
				z_gettime(&start);
				for (i = chunkIter; i != 0U; --i) {
					sysAdler = sysAdler32(sysAdler, chunk, (uInt)chunkSize - 1U);
				}
				z_gettime(&end);
				onediff = z_timediff_sec(&start, &end);
				if (onediff < sysDiff) sysDiff = onediff;
			}
		}
		printf("adler32    %5u bytes: %08lX at %8.2f MB/s\n", (uInt)chunkSize - 1U,    adler, totalSize /    diff / 1024.0 /1024.0);
		if (sysAdler32 != Z_NULL) {
			printf("sysAdler32 %5u bytes: %08lX at %8.2f MB/s\n", (uInt)chunkSize - 1U, sysAdler, totalSize / sysDiff / 1024.0 /1024.0);
		}
	}
}

static void bench_crc32_copy()
{
	size_t chunkSize, chunkIter;

	for (chunkSize = maxChunkSize, chunkIter = iterations; chunkSize != 2U; chunkSize >>= 1, chunkIter <<= 1) {
		double diff = DBL_MAX;
		double sysDiff = DBL_MAX;
		uLong crc = 0U, sysCrc = 0U;
		size_t j;

		for (j = 0U;j < 256U; ++j) {
			z_timespec start, end;
			double onediff;
			size_t i;

			crc = crc32(0U, NULL, 0U);
			z_gettime(&start);
			for (i = chunkIter; i != 0U; --i) {
				crc = crc32_copy(crc, chunk, (uInt)chunkSize - 1U, chunkDst);
			}
			z_gettime(&end);
			onediff = z_timediff_sec(&start, &end);
			if (onediff < diff) diff = onediff;

			if (sysCrc32 != Z_NULL) {
				sysCrc = sysCrc32(0U, NULL, 0U);
				z_gettime(&start);
				for (i = chunkIter; i != 0U; --i) {
					sysCrc = sysCrc32WithCopy(sysCrc, chunk, (uInt)chunkSize - 1U, chunkDst);
				}
				z_gettime(&end);
				onediff = z_timediff_sec(&start, &end);
				if (onediff < sysDiff) sysDiff = onediff;
			}
		}
		printf("crc32_copy       %5u bytes: %08lX at %8.2f MB/s\n", (uInt)chunkSize - 1U,    crc, totalSize /    diff / 1024.0 /1024.0);
		if (sysCrc32 != Z_NULL) {
			printf("sysCrc32WithCopy %5u bytes: %08lX at %8.2f MB/s\n", (uInt)chunkSize - 1U, sysCrc, totalSize / sysDiff / 1024.0 /1024.0);
		}
	}
}

static void bench_crc32()
{
	size_t chunkSize, chunkIter;

	for (chunkSize = maxChunkSize, chunkIter = iterations; chunkSize != 2U; chunkSize >>= 1, chunkIter <<= 1) {
		double diff = DBL_MAX;
		double sysDiff = DBL_MAX;
		uLong crc = 0U, sysCrc = 0U;
		size_t j;

		for (j = 0U;j < 256U; ++j) {
			z_timespec start, end;
			double onediff;
			size_t i;

			crc = crc32(0U, NULL, 0U);
			z_gettime(&start);
			for (i = chunkIter; i != 0U; --i) {
				crc = crc32(crc, chunk, (uInt)chunkSize - 1U);
			}
			z_gettime(&end);
			onediff = z_timediff_sec(&start, &end);
			if (onediff < diff) diff = onediff;

			if (sysCrc32 != Z_NULL) {
				sysCrc = sysCrc32(0U, NULL, 0U);
				z_gettime(&start);
				for (i = chunkIter; i != 0U; --i) {
					sysCrc = sysCrc32(sysCrc, chunk, (uInt)chunkSize - 1U);
				}
				z_gettime(&end);
				onediff = z_timediff_sec(&start, &end);
				if (onediff < sysDiff) sysDiff = onediff;
			}
		}
		printf("crc32    %5u bytes: %08lX at %6.2f MB/s\n", (uInt)chunkSize - 1U,    crc, totalSize /    diff / 1024.0 /1024.0);
		if (sysCrc32 != Z_NULL) {
			printf("sysCrc32 %5u bytes: %08lX at %6.2f MB/s\n", (uInt)chunkSize - 1U, sysCrc, totalSize / sysDiff / 1024.0 /1024.0);
		}
	}
}

static int bench_file(const char* filename)
{
	FILE  *file = Z_NULL;
	uLong input_reference_size = 0U;
	uLong output_size = 0U;
	uLong input_roundtrip_size = 0U;
	Bytef *input_reference = Z_NULL;
	Bytef *output = Z_NULL;
	Bytef *input_roundtrip = Z_NULL;
	double compressTime = DBL_MAX;
	double uncompressTime = DBL_MAX;
	size_t i;

	file = fopen(filename, "rb");
	if (file == Z_NULL) {
		fprintf(stderr, "Can't open file \"%s\"\n", filename);
		return 1;
	}
	if (fseek(file, 0, SEEK_END) == 0) {
		long size = ftell(file);
		rewind(file);
		if ((size > 0) && (size <= SIZE_MAX)) {
			input_reference_size = (uLong)size;
			input_reference = malloc((size_t)input_reference_size);
			if (input_reference != Z_NULL) {
				if (fread(input_reference, 1U, (size_t)input_reference_size, file) != (size_t)input_reference_size) {
					free(input_reference);
					input_reference = Z_NULL;
				}
			}
		}
	}
	fclose(file);

	if (input_reference == Z_NULL) {
		fprintf(stderr, "An error occured when reading file \"%s\"\n", filename);
		return 1;
	}
	if ((input_roundtrip = malloc((size_t)input_reference_size)) == Z_NULL) {
		fprintf(stderr, "Allocation error\n");
		free(input_reference);
		return 1;
	}
	output_size = compressBound(input_reference_size);
	if ((output = malloc((size_t)output_size)) == Z_NULL) {
		fprintf(stderr, "Allocation error\n");
		free(input_reference);
		free(input_roundtrip);
		return 1;
	}

	if (compress2(output, &output_size, input_reference, input_reference_size, 1) != Z_OK) {
		fprintf(stderr, "Compression error\n");
		free(input_reference);
		free(input_roundtrip);
		free(output);
		return 1;
	}

	input_roundtrip_size = input_reference_size;
	if (uncompress(input_roundtrip, &input_roundtrip_size, output, output_size) != Z_OK) {
		fprintf(stderr, "Decompression error\n");
		free(input_reference);
		free(input_roundtrip);
		free(output);
		return 1;
	}
	if ((input_roundtrip_size != input_reference_size) || memcmp(input_reference, input_roundtrip, input_reference_size) != 0) {
		fprintf(stderr, "Roundtrip error\n");
		free(input_reference);
		free(input_roundtrip);
		free(output);
		return 1;
	}

	for (i = 10U; i != 0U; --i) {
		z_timespec start, end;
		double onediff;

		input_roundtrip_size = input_reference_size;
		z_gettime(&start);
		uncompress(input_roundtrip, &input_roundtrip_size, output, output_size);
		z_gettime(&end);
		onediff = z_timediff_sec(&start, &end);
		if (onediff < uncompressTime) uncompressTime = onediff;
	}
	printf("uncompress %9lu bytes at %6.2f MB/s\n", output_size, input_roundtrip_size / uncompressTime / 1024.0 /1024.0);

	for (i = 5U; i != 0U; --i) {
		z_timespec start, end;
		double onediff;

		output_size = compressBound(input_reference_size);
		z_gettime(&start);
		compress2(output, &output_size, input_reference, input_reference_size, 1);
		z_gettime(&end);
		onediff = z_timediff_sec(&start, &end);
		if (onediff < compressTime) compressTime = onediff;
	}
	printf("compress   %9lu bytes at %6.2f MB/s\n", input_reference_size, input_reference_size / compressTime / 1024.0 /1024.0);


	free(input_reference);
	free(input_roundtrip);
	free(output);
	return 0;
}

int main(int argc, char** argv)
{
	int result = 0;
	size_t i;
	void* sysZlib = Z_NULL;

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

	/* bench */
	if (argc > 1) {
		if (bench_file(argv[1]) != 0) {
			result = 1;
		}
	} else {
		/* non regression */
		if (unit_adler32() != 0) {
			result = 1;
		}
		if (unit_crc32() != 0) {
			result = 1;
		}

		bench_adler32_copy();
		bench_adler32();
		bench_crc32_copy();
		bench_crc32();
	}
	return result;
}
