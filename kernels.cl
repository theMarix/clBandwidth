/*
 * This file is part of clBandwidth.
 *
 * clBandwidth is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * clBandwidth is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with clBandwidth.  If not, see <http://www.gnu.org/licenses/>.
 *
 * (c) 2011 Matthias Bach <bach@compeng.uni-frankfurt.de>
 */

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define DOUBLE_ENABLED
#else /* cl_khr_fp64 */
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#define DOUBLE_ENABLED
#endif /* cl_amd_fp64 */
#endif /* cl_khr_fp64 */

/*
 * float kernels
 */

__kernel void copyFloat(__global float * out, __global float * in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = in[i];
	}
}
__kernel void readFloat(__global float * out, __global float * in, const ulong elems)
{
	float tmp = 0.0f;
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp += in[i];
	}
	out[get_global_id(0)] = tmp;
}
__kernel void writeFloat(__global float * out, const float in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = in;
	}
}

__kernel void copyFloatRestricted(__global float * const restrict out, __global const float * const restrict in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = in[i];
	}
}
__kernel void readFloatRestricted(__global float * const restrict out, __global const float * const restrict in, const ulong elems)
{
	float tmp = 0.0f;
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp += in[i];
	}
	out[get_global_id(0)] = tmp;
}
__kernel void writeFloatRestricted(__global float * const restrict out, const float in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = in;
	}
}

__kernel void copyFloat2(__global float2 * out, __global float2 * in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = in[i];
	}
}
__kernel void readFloat2(__global float2 * out, __global float2 * in, const ulong elems)
{
	float2 tmp = 0.0f;
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp += in[i];
	}
	out[get_global_id(0)] = tmp;
}
__kernel void writeFloat2(__global float2 * out, const float in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = in;
	}
}

__kernel void copyFloat4(__global float4 * out, __global float4 * in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = in[i];
	}
}
__kernel void readFloat4(__global float4 * out, __global float4 * in, const ulong elems)
{
	float4 tmp = 0.0f;
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp += in[i];
	}
	out[get_global_id(0)] = tmp;
}
__kernel void writeFloat4(__global float4 * out, const float in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = in;
	}
}

/*
 * Single precision complex
 */

typedef struct { float re; float im; } spComplex;

spComplex make_spComplex(const float re, const float im) {
	return (spComplex) {re, im};
}

spComplex spComplexAdd(const spComplex left, const spComplex right) {
	return make_spComplex(left.re + right.re, left.im + right.im);
}

__kernel void copySpComplex(__global spComplex * out, __global spComplex * in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = in[i];
	}
}
__kernel void readSpComplex(__global spComplex * out, __global spComplex * in, const ulong elems)
{
	spComplex tmp = make_spComplex(0.0f, 0.0f);
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp = spComplexAdd(tmp, in[i]);
	}
	out[get_global_id(0)] = tmp;
}
__kernel void writeSpComplex(__global spComplex * out, const float in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = make_spComplex(in, in);
	}
}

__kernel void copySpComplexRestricted(__global spComplex * const restrict out, __global const spComplex * const restrict in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = in[i];
	}
}
__kernel void readSpComplexRestricted(__global spComplex * const restrict out, __global const spComplex * const restrict in, const ulong elems)
{
	spComplex tmp = make_spComplex(0.0f, 0.0f);
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp = spComplexAdd(tmp, in[i]);
	}
	out[get_global_id(0)] = tmp;
}
__kernel void writeSpComplexRestricted(__global spComplex * const restrict out, const float in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = make_spComplex(in, in);
	}
}

typedef struct { float re; float im; } __attribute__ ((aligned (8))) alignedSpComplex;

alignedSpComplex make_alignedSpComplex(const float re, const float im) {
	return (alignedSpComplex) {re, im};
}

alignedSpComplex alignedSpComplexAdd(const alignedSpComplex left, const alignedSpComplex right) {
	return make_alignedSpComplex(left.re + right.re, left.im + right.im);
}

__kernel void copyAlignedSpComplex(__global alignedSpComplex * out, __global alignedSpComplex * in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = in[i];
	}
}
__kernel void readAlignedSpComplex(__global alignedSpComplex * out, __global alignedSpComplex * in, const ulong elems)
{
	alignedSpComplex tmp = make_alignedSpComplex(0.0f, 0.0f);
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp = alignedSpComplexAdd(tmp, in[i]);
	}
	out[get_global_id(0)] = tmp;
}
__kernel void writeAlignedSpComplex(__global alignedSpComplex * out, const float in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = make_alignedSpComplex(in, in);
	}
}

__kernel void copyAlignedSpComplexRestricted(__global alignedSpComplex * const restrict out, __global const alignedSpComplex * const restrict in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = in[i];
	}
}
__kernel void readAlignedSpComplexRestricted(__global alignedSpComplex * const restrict out, __global const alignedSpComplex * const restrict in, const ulong elems)
{
	alignedSpComplex tmp = make_alignedSpComplex(0.0f, 0.0f);
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp = alignedSpComplexAdd(tmp, in[i]);
	}
	out[get_global_id(0)] = tmp;
}
__kernel void writeAlignedSpComplexRestricted(__global alignedSpComplex * const restrict out, const float in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = make_alignedSpComplex(in, in);
	}
}

/*
 * Single precisoin SU3 vectors
 */

typedef struct {
	spComplex e0;
	spComplex e1;
	spComplex e2;
} spSu3vec;

spSu3vec make_spSu3vec(const spComplex e0, const spComplex e1, const spComplex e2) {
	return (spSu3vec) {e0, e1, e2};
}

spSu3vec spSu3vecAdd(const spSu3vec left, const spSu3vec right) {
	return make_spSu3vec(
		spComplexAdd(left.e0, right.e0),
		spComplexAdd(left.e1, right.e1),
		spComplexAdd(left.e2, right.e2)
	);
}

__kernel void copySpSu3vec(__global spSu3vec * out, __global spSu3vec * in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = in[i];
	}
}
__kernel void readSpSu3vec(__global spSu3vec * out, __global spSu3vec * in, const ulong elems)
{
	spComplex bla = make_spComplex(0.0f, 0.0f);
	spSu3vec tmp = make_spSu3vec(bla, bla, bla);
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp = spSu3vecAdd(tmp, in[i]);
	}
	out[get_global_id(0)] = tmp;
}
__kernel void writeSpSu3vec(__global spSu3vec * out, const float in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		spComplex bla = make_spComplex(in, in);
		out[i] = make_spSu3vec(bla, bla, bla);
	}
}

__kernel void copySpSu3vecRestricted(__global spSu3vec * const restrict out, __global const spSu3vec * const restrict in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = in[i];
	}
}
__kernel void readSpSu3vecRestricted(__global spSu3vec * const restrict out, __global const spSu3vec * const restrict in, const ulong elems)
{
	spComplex bla = make_spComplex(0.0f, 0.0f);
	spSu3vec tmp = make_spSu3vec(bla, bla, bla);
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp = spSu3vecAdd(tmp, in[i]);
	}
	out[get_global_id(0)] = tmp;
}
__kernel void writeSpSu3vecRestricted(__global spSu3vec * const restrict out, const float in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		spComplex bla = make_spComplex(in, in);
		out[i] = make_spSu3vec(bla, bla, bla);
	}
}

typedef struct {
	spComplex e0;
	spComplex e1;
	spComplex e2;
} __attribute((aligned)) alignedSpSu3vec;

alignedSpSu3vec make_alignedSpSu3vec(const spComplex e0, const spComplex e1, const spComplex e2) {
	return (alignedSpSu3vec) {e0, e1, e2};
}

alignedSpSu3vec alignedSpSu3vecAdd(const alignedSpSu3vec left, const alignedSpSu3vec right) {
	return make_alignedSpSu3vec(
		spComplexAdd(left.e0, right.e0),
		spComplexAdd(left.e1, right.e1),
		spComplexAdd(left.e2, right.e2)
	);
}

__kernel void copyAlignedSpSu3vecRestricted(__global alignedSpSu3vec * const restrict out, __global const alignedSpSu3vec * const restrict in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = in[i];
	}
}
__kernel void readAlignedSpSu3vecRestricted(__global alignedSpSu3vec * const restrict out, __global const alignedSpSu3vec * const restrict in, const ulong elems)
{
	spComplex bla = make_spComplex(0.0f, 0.0f);
	alignedSpSu3vec tmp = make_alignedSpSu3vec(bla, bla, bla);
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp = alignedSpSu3vecAdd(tmp, in[i]);
	}
	out[get_global_id(0)] = tmp;
}
__kernel void writeAlignedSpSu3vecRestricted(__global alignedSpSu3vec * const restrict out, const float in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		spComplex bla = make_spComplex(in, in);
		out[i] = make_alignedSpSu3vec(bla, bla, bla);
	}
}

typedef struct {
	spComplex e0;
	spComplex e1;
	spComplex e2;
} __attribute((aligned(8))) aligned8SpSu3vec;

aligned8SpSu3vec make_aligned8SpSu3vec(const spComplex e0, const spComplex e1, const spComplex e2) {
	return (aligned8SpSu3vec) {e0, e1, e2};
}

aligned8SpSu3vec aligned8SpSu3vecAdd(const aligned8SpSu3vec left, const aligned8SpSu3vec right) {
	return make_aligned8SpSu3vec(
		spComplexAdd(left.e0, right.e0),
		spComplexAdd(left.e1, right.e1),
		spComplexAdd(left.e2, right.e2)
	);
}

__kernel void copyAligned8SpSu3vecRestricted(__global aligned8SpSu3vec * const restrict out, __global const aligned8SpSu3vec * const restrict in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = in[i];
	}
}
__kernel void readAligned8SpSu3vecRestricted(__global aligned8SpSu3vec * const restrict out, __global const aligned8SpSu3vec * const restrict in, const ulong elems)
{
	spComplex bla = make_spComplex(0.0f, 0.0f);
	aligned8SpSu3vec tmp = make_aligned8SpSu3vec(bla, bla, bla);
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp = aligned8SpSu3vecAdd(tmp, in[i]);
	}
	out[get_global_id(0)] = tmp;
}
__kernel void writeAligned8SpSu3vecRestricted(__global aligned8SpSu3vec * const restrict out, const float in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		spComplex bla = make_spComplex(in, in);
		out[i] = make_aligned8SpSu3vec(bla, bla, bla);
	}
}

typedef struct {
	spComplex e0;
	spComplex e1;
	spComplex e2;
} __attribute((aligned(16))) aligned16SpSu3vec;

aligned16SpSu3vec make_aligned16SpSu3vec(const spComplex e0, const spComplex e1, const spComplex e2) {
	return (aligned16SpSu3vec) {e0, e1, e2};
}

aligned16SpSu3vec aligned16SpSu3vecAdd(const aligned16SpSu3vec left, const aligned16SpSu3vec right) {
	return make_aligned16SpSu3vec(
		spComplexAdd(left.e0, right.e0),
		spComplexAdd(left.e1, right.e1),
		spComplexAdd(left.e2, right.e2)
	);
}

__kernel void copyAligned16SpSu3vecRestricted(__global aligned16SpSu3vec * const restrict out, __global const aligned16SpSu3vec * const restrict in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = in[i];
	}
}
__kernel void readAligned16SpSu3vecRestricted(__global aligned16SpSu3vec * const restrict out, __global const aligned16SpSu3vec * const restrict in, const ulong elems)
{
	spComplex bla = make_spComplex(0.0f, 0.0f);
	aligned16SpSu3vec tmp = make_aligned16SpSu3vec(bla, bla, bla);
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp = aligned16SpSu3vecAdd(tmp, in[i]);
	}
	out[get_global_id(0)] = tmp;
}
__kernel void writeAligned16SpSu3vecRestricted(__global aligned16SpSu3vec * const restrict out, const float in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		spComplex bla = make_spComplex(in, in);
		out[i] = make_aligned16SpSu3vec(bla, bla, bla);
	}
}

typedef struct {
	spComplex e0;
	spComplex e1;
	spComplex e2;
} __attribute((aligned(32))) aligned32SpSu3vec;

aligned32SpSu3vec make_aligned32SpSu3vec(const spComplex e0, const spComplex e1, const spComplex e2) {
	return (aligned32SpSu3vec) {e0, e1, e2};
}

aligned32SpSu3vec aligned32SpSu3vecAdd(const aligned32SpSu3vec left, const aligned32SpSu3vec right) {
	return make_aligned32SpSu3vec(
		spComplexAdd(left.e0, right.e0),
		spComplexAdd(left.e1, right.e1),
		spComplexAdd(left.e2, right.e2)
	);
}

__kernel void copyAligned32SpSu3vecRestricted(__global aligned32SpSu3vec * const restrict out, __global const aligned32SpSu3vec * const restrict in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = in[i];
	}
}
__kernel void readAligned32SpSu3vecRestricted(__global aligned32SpSu3vec * const restrict out, __global const aligned32SpSu3vec * const restrict in, const ulong elems)
{
	spComplex bla = make_spComplex(0.0f, 0.0f);
	aligned32SpSu3vec tmp = make_aligned32SpSu3vec(bla, bla, bla);
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp = aligned32SpSu3vecAdd(tmp, in[i]);
	}
	out[get_global_id(0)] = tmp;
}
__kernel void writeAligned32SpSu3vecRestricted(__global aligned32SpSu3vec * const restrict out, const float in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		spComplex bla = make_spComplex(in, in);
		out[i] = make_aligned32SpSu3vec(bla, bla, bla);
	}
}

typedef struct {
	alignedSpComplex e0;
	alignedSpComplex e1;
	alignedSpComplex e2;
} spSu3vecFromAligned;

spSu3vecFromAligned make_spSu3vecFromAligned(const alignedSpComplex e0, const alignedSpComplex e1, const alignedSpComplex e2) {
	return (spSu3vecFromAligned) {e0, e1, e2};
}

spSu3vecFromAligned spSu3vecFromAlignedAdd(const spSu3vecFromAligned left, const spSu3vecFromAligned right) {
	return make_spSu3vecFromAligned(
		alignedSpComplexAdd(left.e0, right.e0),
		alignedSpComplexAdd(left.e1, right.e1),
		alignedSpComplexAdd(left.e2, right.e2)
	);
}

__kernel void copySpSu3vecFromAlignedRestricted(__global spSu3vecFromAligned * const restrict out, __global const spSu3vecFromAligned * const restrict in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = in[i];
	}
}
__kernel void readSpSu3vecFromAlignedRestricted(__global spSu3vecFromAligned * const restrict out, __global const spSu3vecFromAligned * const restrict in, const ulong elems)
{
	alignedSpComplex bla = make_alignedSpComplex(0.0f, 0.0f);
	spSu3vecFromAligned tmp = make_spSu3vecFromAligned(bla, bla, bla);
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp = spSu3vecFromAlignedAdd(tmp, in[i]);
	}
	out[get_global_id(0)] = tmp;
}
__kernel void writeSpSu3vecFromAlignedRestricted(__global spSu3vecFromAligned * const restrict out, const float in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		alignedSpComplex bla = make_alignedSpComplex(in, in);
		out[i] = make_spSu3vecFromAligned(bla, bla, bla);
	}
}

/*
 * Single precisoin SU3
 */

typedef struct {
	spComplex e00, e01, e02;
	spComplex e10, e11, e12;
	spComplex e20, e21, e22;
} spSu3;

spSu3 make_spSu3(const spComplex e00, const spComplex e01, const spComplex e02,
                 const spComplex e10, const spComplex e11, const spComplex e12,
                 const spComplex e20, const spComplex e21, const spComplex e22) {
	return (spSu3) {e00, e01, e02,
	                e10, e11, e12,
	                e20, e21, e22};
}

spSu3 spSu3Add(const spSu3 left, const spSu3 right) {
	return make_spSu3(
		spComplexAdd(left.e00, right.e00),
		spComplexAdd(left.e01, right.e01),
		spComplexAdd(left.e02, right.e02),
		spComplexAdd(left.e10, right.e10),
		spComplexAdd(left.e11, right.e11),
		spComplexAdd(left.e12, right.e12),
		spComplexAdd(left.e20, right.e20),
		spComplexAdd(left.e21, right.e21),
		spComplexAdd(left.e22, right.e22)
	);
}

__kernel void copySpSu3(__global spSu3 * out, __global spSu3 * in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = in[i];
	}
}
__kernel void readSpSu3(__global spSu3 * out, __global spSu3 * in, const ulong elems)
{
	spComplex bla = make_spComplex(0.0f, 0.0f);
	spSu3 tmp = make_spSu3(bla, bla, bla, bla, bla, bla, bla, bla, bla);
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp = spSu3Add(tmp, in[i]);
	}
	out[get_global_id(0)] = tmp;
}
__kernel void writeSpSu3(__global spSu3 * out, const float in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		spComplex bla = make_spComplex(in, in);
		out[i] = make_spSu3(bla, bla, bla, bla, bla, bla, bla, bla, bla);
	}
}

__kernel void copySpSu3Restricted(__global spSu3 * const restrict out, __global const spSu3 * const restrict in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = in[i];
	}
}
__kernel void readSpSu3Restricted(__global spSu3 * const restrict out, __global const spSu3 * const restrict in, const ulong elems)
{
	spComplex bla = make_spComplex(0.0f, 0.0f);
	spSu3 tmp = make_spSu3(bla, bla, bla, bla, bla, bla, bla, bla, bla);
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp = spSu3Add(tmp, in[i]);
	}
	out[get_global_id(0)] = tmp;
}
__kernel void writeSpSu3Restricted(__global spSu3 * const restrict out, const float in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		spComplex bla = make_spComplex(in, in);
		out[i] = make_spSu3(bla, bla, bla, bla, bla, bla, bla, bla, bla);
	}
}

spSu3 getSpSu3SOA(__global const spComplex * const restrict in, const size_t i)
{
	const size_t stride = get_global_size(0);
	return make_spSu3(in[0 * stride + i], in[1 * stride + i], in[2 * stride + i],
	                  in[3 * stride + i], in[4 * stride + i], in[5 * stride + i],
	                  in[6 * stride + i], in[7 * stride + i], in[8 * stride + i]);
}

void putSpSu3SOA(__global spComplex * const restrict out, const size_t i, const spSu3 val)
{
	const size_t stride = get_global_size(0);
	out[0 * stride + i] = val.e00;
	out[1 * stride + i] = val.e01;
	out[2 * stride + i] = val.e02;
	out[3 * stride + i] = val.e10;
	out[4 * stride + i] = val.e11;
	out[5 * stride + i] = val.e12;
	out[6 * stride + i] = val.e20;
	out[7 * stride + i] = val.e21;
	out[8 * stride + i] = val.e22;
}

__kernel void copySpSu3SOARestricted(__global spComplex * const restrict out, __global const spComplex * const restrict in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		spSu3 tmp = getSpSu3SOA(in, i);
		putSpSu3SOA(out, i, tmp);
	}
}
__kernel void readSpSu3SOARestricted(__global spComplex * const restrict out, __global const spComplex * const restrict in, const ulong elems)
{
	spComplex bla = make_spComplex(0.0f, 0.0f);
	spSu3 tmp = make_spSu3(bla, bla, bla, bla, bla, bla, bla, bla, bla);
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp = spSu3Add(tmp, getSpSu3SOA(in, i));
	}
	putSpSu3SOA(out, get_global_id(0), tmp);
}
__kernel void writeSpSu3SOARestricted(__global spComplex * const restrict out, const float in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		spComplex bla = make_spComplex(in, in);
		spSu3 tmp = make_spSu3(bla, bla, bla, bla, bla, bla, bla, bla, bla);
		putSpSu3SOA(out, i, tmp);
	}
}

typedef struct {
	alignedSpComplex e00, e01, e02;
	alignedSpComplex e10, e11, e12;
	alignedSpComplex e20, e21, e22;
} spSu3FromAligned;

spSu3FromAligned make_spSu3FromAligned(const alignedSpComplex e00, const alignedSpComplex e01, const alignedSpComplex e02,
                                       const alignedSpComplex e10, const alignedSpComplex e11, const alignedSpComplex e12,
                                       const alignedSpComplex e20, const alignedSpComplex e21, const alignedSpComplex e22) {
	return (spSu3FromAligned) {e00, e01, e02,
	                e10, e11, e12,
	                e20, e21, e22};
}

spSu3FromAligned spSu3FromAlignedAdd(const spSu3FromAligned left, const spSu3FromAligned right) {
	return make_spSu3FromAligned(
		alignedSpComplexAdd(left.e00, right.e00),
		alignedSpComplexAdd(left.e01, right.e01),
		alignedSpComplexAdd(left.e02, right.e02),
		alignedSpComplexAdd(left.e10, right.e10),
		alignedSpComplexAdd(left.e11, right.e11),
		alignedSpComplexAdd(left.e12, right.e12),
		alignedSpComplexAdd(left.e20, right.e20),
		alignedSpComplexAdd(left.e21, right.e21),
		alignedSpComplexAdd(left.e22, right.e22)
	);
}

__kernel void copySpSu3FromAlignedRestricted(__global spSu3FromAligned * const restrict out, __global const spSu3FromAligned * const restrict in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = in[i];
	}
}
__kernel void readSpSu3FromAlignedRestricted(__global spSu3FromAligned * const restrict out, __global const spSu3FromAligned * const restrict in, const ulong elems)
{
	alignedSpComplex bla = make_alignedSpComplex(0.0f, 0.0f);
	spSu3FromAligned tmp = make_spSu3FromAligned(bla, bla, bla, bla, bla, bla, bla, bla, bla);
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp = spSu3FromAlignedAdd(tmp, in[i]);
	}
	out[get_global_id(0)] = tmp;
}
__kernel void writeSpSu3FromAlignedRestricted(__global spSu3FromAligned * const restrict out, const float in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		alignedSpComplex bla = make_alignedSpComplex(in, in);
		out[i] = make_spSu3FromAligned(bla, bla, bla, bla, bla, bla, bla, bla, bla);
	}
}

spSu3FromAligned getSpSu3FromAlignedSOA(__global const alignedSpComplex * const restrict in, const size_t i)
{
	const size_t stride = get_global_size(0);
	return make_spSu3FromAligned(in[0 * stride + i], in[1 * stride + i], in[2 * stride + i],
	                  in[3 * stride + i], in[4 * stride + i], in[5 * stride + i],
	                  in[6 * stride + i], in[7 * stride + i], in[8 * stride + i]);
}

void putSpSu3FromAlignedSOA(__global alignedSpComplex * const restrict out, const size_t i, const spSu3FromAligned val)
{
	const size_t stride = get_global_size(0);
	out[0 * stride + i] = val.e00;
	out[1 * stride + i] = val.e01;
	out[2 * stride + i] = val.e02;
	out[3 * stride + i] = val.e10;
	out[4 * stride + i] = val.e11;
	out[5 * stride + i] = val.e12;
	out[6 * stride + i] = val.e20;
	out[7 * stride + i] = val.e21;
	out[8 * stride + i] = val.e22;
}

__kernel void copySpSu3FromAlignedSOARestricted(__global alignedSpComplex * const restrict out, __global const alignedSpComplex * const restrict in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		spSu3FromAligned tmp = getSpSu3FromAlignedSOA(in, i);
		putSpSu3FromAlignedSOA(out, i, tmp);
	}
}
__kernel void readSpSu3FromAlignedSOARestricted(__global alignedSpComplex * const restrict out, __global const alignedSpComplex * const restrict in, const ulong elems)
{
	alignedSpComplex bla = make_alignedSpComplex(0.0f, 0.0f);
	spSu3FromAligned tmp = make_spSu3FromAligned(bla, bla, bla, bla, bla, bla, bla, bla, bla);
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp = spSu3FromAlignedAdd(tmp, getSpSu3FromAlignedSOA(in, i));
	}
	putSpSu3FromAlignedSOA(out, get_global_id(0), tmp);
}
__kernel void writeSpSu3FromAlignedSOARestricted(__global alignedSpComplex * const restrict out, const float in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		alignedSpComplex bla = make_alignedSpComplex(in, in);
		spSu3FromAligned tmp = make_spSu3FromAligned(bla, bla, bla, bla, bla, bla, bla, bla, bla);
		putSpSu3FromAlignedSOA(out, i, tmp);
	}
}

typedef struct {
	spComplex e00, e01, e02;
	spComplex e10, e11, e12;
	spComplex e20, e21, e22;
} __attribute__((aligned(8))) alignedSpSu3;

alignedSpSu3 make_alignedSpSu3(const spComplex e00, const spComplex e01, const spComplex e02,
                               const spComplex e10, const spComplex e11, const spComplex e12,
                               const spComplex e20, const spComplex e21, const spComplex e22) {
	return (alignedSpSu3) {e00, e01, e02,
	                e10, e11, e12,
	                e20, e21, e22};
}

alignedSpSu3 alignedSpSu3Add(const alignedSpSu3 left, const alignedSpSu3 right) {
	return make_alignedSpSu3(
		spComplexAdd(left.e00, right.e00),
		spComplexAdd(left.e01, right.e01),
		spComplexAdd(left.e02, right.e02),
		spComplexAdd(left.e10, right.e10),
		spComplexAdd(left.e11, right.e11),
		spComplexAdd(left.e12, right.e12),
		spComplexAdd(left.e20, right.e20),
		spComplexAdd(left.e21, right.e21),
		spComplexAdd(left.e22, right.e22)
	);
}

__kernel void copyAligned8SpSu3Restricted(__global alignedSpSu3 * const restrict out, __global const alignedSpSu3 * const restrict in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = in[i];
	}
}
__kernel void readAligned8SpSu3Restricted(__global alignedSpSu3 * const restrict out, __global const alignedSpSu3 * const restrict in, const ulong elems)
{
	spComplex bla = make_spComplex(0.0f, 0.0f);
	alignedSpSu3 tmp = make_alignedSpSu3(bla, bla, bla, bla, bla, bla, bla, bla, bla);
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp = alignedSpSu3Add(tmp, in[i]);
	}
	out[get_global_id(0)] = tmp;
}
__kernel void writeAligned8SpSu3Restricted(__global alignedSpSu3 * const restrict out, const float in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		spComplex bla = make_spComplex(in, in);
		out[i] = make_alignedSpSu3(bla, bla, bla, bla, bla, bla, bla, bla, bla);
	}
}

/*
 * double kernels
 */

#ifdef DOUBLE_ENABLED

__kernel void copyDouble(__global double * out, __global double * in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = in[i];
	}
}
__kernel void readDouble(__global double * out, __global double * in, const ulong elems)
{
	double tmp = 0.0f;
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp += in[i];
	}
	out[get_global_id(0)] = tmp;
}
__kernel void writeDouble(__global double * out, const double in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = in;
	}
}

__kernel void copyDoubleRestricted(__global double * const restrict out, __global const double * const restrict in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = in[i];
	}
}
__kernel void readDoubleRestricted(__global double * const restrict out, __global const double * const restrict in, const ulong elems)
{
	double tmp = 0.0f;
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp += in[i];
	}
	out[get_global_id(0)] = tmp;
}
__kernel void writeDoubleRestricted(__global double * const restrict out, const double in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = in;
	}
}

__kernel void copyDouble2(__global double2 * out, __global double2 * in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = in[i];
	}
}
__kernel void readDouble2(__global double2 * out, __global double2 * in, const ulong elems)
{
	double2 tmp = 0.0f;
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp += in[i];
	}
	out[get_global_id(0)] = tmp;
}
__kernel void writeDouble2(__global double2 * out, const double in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = in;
	}
}

__kernel void copyDouble4(__global double4 * out, __global double4 * in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = in[i];
	}
}
__kernel void readDouble4(__global double4 * out, __global double4 * in, const ulong elems)
{
	double4 tmp = 0.0f;
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp += in[i];
	}
	out[get_global_id(0)] = tmp;
}
__kernel void writeDouble4(__global double4 * out, const double in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = in;
	}
}

/*
 * Double precision complex
 */

typedef struct { double re; double im; } dpComplex;

dpComplex make_dpComplex(const double re, const double im) {
	return (dpComplex) {re, im};
}

dpComplex dpComplexAdd(const dpComplex left, const dpComplex right) {
	return make_dpComplex(left.re + right.re, left.im + right.im);
}

__kernel void copyDpComplex(__global dpComplex * out, __global dpComplex * in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = in[i];
	}
}
__kernel void readDpComplex(__global dpComplex * out, __global dpComplex * in, const ulong elems)
{
	dpComplex tmp = make_dpComplex(0.0f, 0.0f);
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp = dpComplexAdd(tmp, in[i]);
	}
	out[get_global_id(0)] = tmp;
}
__kernel void writeDpComplex(__global dpComplex * out, const double in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = make_dpComplex(in, in);
	}
}

__kernel void copyDpComplexRestricted(__global dpComplex * const restrict out, __global const dpComplex * const restrict in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = in[i];
	}
}
__kernel void readDpComplexRestricted(__global dpComplex * const restrict out, __global const dpComplex * const restrict in, const ulong elems)
{
	dpComplex tmp = make_dpComplex(0.0f, 0.0f);
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp = dpComplexAdd(tmp, in[i]);
	}
	out[get_global_id(0)] = tmp;
}
__kernel void writeDpComplexRestricted(__global dpComplex * const restrict out, const double in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = make_dpComplex(in, in);
	}
}

typedef struct { double re; double im; } __attribute__((aligned (16))) alignedDpComplex;

alignedDpComplex make_alignedDpComplex(const double re, const double im) {
	return (alignedDpComplex) {re, im};
}

alignedDpComplex alignedDpComplexAdd(const alignedDpComplex left, const alignedDpComplex right) {
	return make_alignedDpComplex(left.re + right.re, left.im + right.im);
}

__kernel void copyAlignedDpComplex(__global alignedDpComplex * out, __global alignedDpComplex * in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = in[i];
	}
}
__kernel void readAlignedDpComplex(__global alignedDpComplex * out, __global alignedDpComplex * in, const ulong elems)
{
	alignedDpComplex tmp = make_alignedDpComplex(0.0f, 0.0f);
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp = alignedDpComplexAdd(tmp, in[i]);
	}
	out[get_global_id(0)] = tmp;
}
__kernel void writeAlignedDpComplex(__global alignedDpComplex * out, const double in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = make_alignedDpComplex(in, in);
	}
}

__kernel void copyAlignedDpComplexRestricted(__global alignedDpComplex * const restrict out, __global const alignedDpComplex * const restrict in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = in[i];
	}
}
__kernel void readAlignedDpComplexRestricted(__global alignedDpComplex * const restrict out, __global const alignedDpComplex * const restrict in, const ulong elems)
{
	alignedDpComplex tmp = make_alignedDpComplex(0.0f, 0.0f);
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp = alignedDpComplexAdd(tmp, in[i]);
	}
	out[get_global_id(0)] = tmp;
}
__kernel void writeAlignedDpComplexRestricted(__global alignedDpComplex * const restrict out, const double in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = make_alignedDpComplex(in, in);
	}
}

#endif /* DOUBLE_ENABLED */
