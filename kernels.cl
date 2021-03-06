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
 * Single precision SU3
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

spSu3 getSpSu3SOA(__global const spComplex * const restrict in, const size_t i, const size_t stride)
{
	return make_spSu3(in[0 * stride + i], in[1 * stride + i], in[2 * stride + i],
	                  in[3 * stride + i], in[4 * stride + i], in[5 * stride + i],
	                  in[6 * stride + i], in[7 * stride + i], in[8 * stride + i]);
}

void putSpSu3SOA(__global spComplex * const restrict out, const size_t i, const spSu3 val, const size_t stride)
{
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

__kernel void copySpSu3SOARestricted(__global spComplex * const restrict out, __global const spComplex * const restrict in, const ulong elems, ulong stride)
{
	stride = (stride == 0) ? elems : stride;
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		spSu3 tmp = getSpSu3SOA(in, i, stride);
		putSpSu3SOA(out, i, tmp, stride);
	}
}
__kernel void readSpSu3SOARestricted(__global spComplex * const restrict out, __global const spComplex * const restrict in, const ulong elems, ulong stride)
{
	stride = (stride == 0) ? elems : stride;
	spComplex bla = make_spComplex(0.0f, 0.0f);
	spSu3 tmp = make_spSu3(bla, bla, bla, bla, bla, bla, bla, bla, bla);
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp = spSu3Add(tmp, getSpSu3SOA(in, i, stride));
	}
	putSpSu3SOA(out, get_global_id(0), tmp, stride);
}
__kernel void writeSpSu3SOARestricted(__global spComplex * const restrict out, const float in, const ulong elems, ulong stride)
{
	stride = (stride == 0) ? elems : stride;
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		spComplex bla = make_spComplex(in, in);
		spSu3 tmp = make_spSu3(bla, bla, bla, bla, bla, bla, bla, bla, bla);
		putSpSu3SOA(out, i, tmp, stride);
	}
}

spSu3 getSpSu3ViaLocal(__global const spSu3 * const restrict in, const size_t block, __local spSu3 * const restrict scratch)
{
	event_t event = async_work_group_copy((__local float2 *) scratch, (__global float2 *) &in[block * get_local_size(0)], get_local_size(0) * 9, 0);
	wait_group_events(1, &event);
	return scratch[get_local_id(0)];
}


void putSpSu3ViaLocal(__global spSu3 * const restrict out, const size_t block, const spSu3 val, __local spSu3 * const restrict scratch)
{
	scratch[get_local_id(0)] = val;
	barrier(CLK_LOCAL_MEM_FENCE);
	event_t event = async_work_group_copy((__global float2 *) &out[block * get_local_size(0)], (__local float2 *) scratch, get_local_size(0) * 9, 0);
	wait_group_events(1, &event);
}

__kernel void copySpSu3ViaLocalRestricted(__global spSu3 * const restrict out, __global const spSu3 * const restrict in, const ulong elems, __local spSu3 * const restrict scratch)
{
	for(size_t i = get_group_id(0); i < elems / get_num_groups(0); i += get_num_groups(0)) {
		spSu3 tmp = getSpSu3ViaLocal(in, i, scratch);
		putSpSu3ViaLocal(out, i, tmp, scratch);
	}
}
__kernel void readSpSu3ViaLocalRestricted(__global spSu3 * const restrict out, __global const spSu3 * const restrict in, const ulong elems, __local spSu3 * const restrict scratch)
{
	spComplex bla = make_spComplex(0.0f, 0.0f);
	spSu3 tmp = make_spSu3(bla, bla, bla, bla, bla, bla, bla, bla, bla);
	for(size_t i = get_group_id(0); i < elems / get_num_groups(0); i += get_num_groups(0)) {
		tmp = spSu3Add(tmp, getSpSu3ViaLocal(in, i, scratch));
	}
	putSpSu3ViaLocal(out, get_global_id(0), tmp, scratch);
}
__kernel void writeSpSu3ViaLocalRestricted(__global spSu3 * const restrict out, const float in, const ulong elems, __local spSu3 * const restrict scratch)
{
	for(size_t i = get_group_id(0); i < elems / get_num_groups(0); i += get_num_groups(0)) {
		spComplex bla = make_spComplex(in, in);
		spSu3 tmp = make_spSu3(bla, bla, bla, bla, bla, bla, bla, bla, bla);
		putSpSu3ViaLocal(out, i, tmp, scratch);
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

spSu3FromAligned getSpSu3FromAlignedSOA(__global const alignedSpComplex * const restrict in, const size_t i, const size_t stride)
{
	return make_spSu3FromAligned(in[0 * stride + i], in[1 * stride + i], in[2 * stride + i],
	                  in[3 * stride + i], in[4 * stride + i], in[5 * stride + i],
	                  in[6 * stride + i], in[7 * stride + i], in[8 * stride + i]);
}

void putSpSu3FromAlignedSOA(__global alignedSpComplex * const restrict out, const size_t i, const spSu3FromAligned val, const size_t stride)
{
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

__kernel void copySpSu3FromAlignedSOARestricted(__global alignedSpComplex * const restrict out, __global const alignedSpComplex * const restrict in, const ulong elems, ulong stride)
{
	stride = (stride == 0) ? elems : stride;
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		spSu3FromAligned tmp = getSpSu3FromAlignedSOA(in, i, stride);
		putSpSu3FromAlignedSOA(out, i, tmp, stride);
	}
}
__kernel void readSpSu3FromAlignedSOARestricted(__global alignedSpComplex * const restrict out, __global const alignedSpComplex * const restrict in, const ulong elems, ulong stride)
{
	stride = (stride == 0) ? elems : stride;
	alignedSpComplex bla = make_alignedSpComplex(0.0f, 0.0f);
	spSu3FromAligned tmp = make_spSu3FromAligned(bla, bla, bla, bla, bla, bla, bla, bla, bla);
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp = spSu3FromAlignedAdd(tmp, getSpSu3FromAlignedSOA(in, i, stride));
	}
	putSpSu3FromAlignedSOA(out, get_global_id(0), tmp, stride);
}
__kernel void writeSpSu3FromAlignedSOARestricted(__global alignedSpComplex * const restrict out, const float in, const ulong elems, ulong stride)
{
	stride = (stride == 0) ? elems : stride;
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		alignedSpComplex bla = make_alignedSpComplex(in, in);
		spSu3FromAligned tmp = make_spSu3FromAligned(bla, bla, bla, bla, bla, bla, bla, bla, bla);
		putSpSu3FromAlignedSOA(out, i, tmp, stride);
	}
}

spSu3FromAligned getSpSu3FromAlignedViaLocal(__global const spSu3FromAligned * const restrict in, const size_t block, __local spSu3FromAligned * const restrict scratch)
{
	event_t event = async_work_group_copy((__local float2 *) scratch, (__global float2 *) &in[block * get_local_size(0)], get_local_size(0) * 9, 0);
	wait_group_events(1, &event);
	return scratch[get_local_id(0)];
}
void putSpSu3FromAlignedViaLocal(__global spSu3FromAligned * const restrict out, const size_t block, const spSu3FromAligned val, __local spSu3FromAligned * const restrict scratch)
{
	scratch[get_local_id(0)] = val;
	barrier(CLK_LOCAL_MEM_FENCE);
	event_t event = async_work_group_copy((__global float2 *) &out[block * get_local_size(0)], (__local float2 *) scratch, get_local_size(0) * 9, 0);
	wait_group_events(1, &event);
}

__kernel void copySpSu3FromAlignedViaLocalRestricted(__global spSu3FromAligned * const restrict out, __global const spSu3FromAligned * const restrict in, const ulong elems, __local spSu3FromAligned * const restrict scratch)
{
	for(size_t i = get_group_id(0); i < elems / get_num_groups(0); i += get_num_groups(0)) {
		spSu3FromAligned tmp = getSpSu3FromAlignedViaLocal(in, i, scratch);
		putSpSu3FromAlignedViaLocal(out, i, tmp, scratch);
	}
}
__kernel void readSpSu3FromAlignedViaLocalRestricted(__global spSu3FromAligned * const restrict out, __global const spSu3FromAligned * const restrict in, const ulong elems, __local spSu3FromAligned * const restrict scratch)
{
	alignedSpComplex bla = make_alignedSpComplex(0.0f, 0.0f);
	spSu3FromAligned tmp = make_spSu3FromAligned(bla, bla, bla, bla, bla, bla, bla, bla, bla);
	for(size_t i = get_group_id(0); i < elems / get_num_groups(0); i += get_num_groups(0)) {
		tmp = spSu3FromAlignedAdd(tmp, getSpSu3FromAlignedViaLocal(in, i, scratch));
	}
	putSpSu3FromAlignedViaLocal(out, get_global_id(0), tmp, scratch);
}
__kernel void writeSpSu3FromAlignedViaLocalRestricted(__global spSu3FromAligned * const restrict out, const float in, const ulong elems, __local spSu3FromAligned * const restrict scratch)
{
	for(size_t i = get_group_id(0); i < elems / get_num_groups(0); i += get_num_groups(0)) {
		alignedSpComplex bla = make_alignedSpComplex(in, in);
		spSu3FromAligned tmp = make_spSu3FromAligned(bla, bla, bla, bla, bla, bla, bla, bla, bla);
		putSpSu3FromAlignedViaLocal(out, i, tmp, scratch);
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
 * Single precision spinors
 */

typedef struct {
	spSu3vec e0;
	spSu3vec e1;
	spSu3vec e2;
	spSu3vec e3;
} spSpinor;

spSpinor make_spSpinor(const spSu3vec e0, const spSu3vec e1, const spSu3vec e2, const spSu3vec e3) {
	return (spSpinor) {e0, e1, e2, e3};
}
spSpinor spSpinorAdd(const spSpinor left, const spSpinor right) {
	return make_spSpinor(
		spSu3vecAdd(left.e0, right.e0),
		spSu3vecAdd(left.e1, right.e1),
		spSu3vecAdd(left.e2, right.e2),
		spSu3vecAdd(left.e3, right.e3)
	);
}

__kernel void copySpSpinor(__global spSpinor * out, __global spSpinor * in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = in[i];
	}
}
__kernel void readSpSpinor(__global spSpinor * out, __global spSpinor * in, const ulong elems)
{
	spComplex bla = make_spComplex(0.0f, 0.0f);
	spSu3vec foo = make_spSu3vec(bla, bla, bla);
	spSpinor tmp = make_spSpinor(foo, foo, foo, foo);
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp = spSpinorAdd(tmp, in[i]);
	}
	out[get_global_id(0)] = tmp;
}
__kernel void writeSpSpinor(__global spSpinor * out, const float in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		spComplex bla = make_spComplex(in, in);
		spSu3vec foo = make_spSu3vec(bla, bla, bla);
		out[i] = make_spSpinor(foo, foo, foo, foo);
	}
}

__kernel void copySpSpinorRestricted(__global spSpinor * const restrict out, __global const spSpinor * const restrict in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = in[i];
	}
}
__kernel void readSpSpinorRestricted(__global spSpinor * const restrict out, __global const spSpinor * const restrict in, const ulong elems)
{
	spComplex bla = make_spComplex(0.0f, 0.0f);
	spSu3vec foo = make_spSu3vec(bla, bla, bla);
	spSpinor tmp = make_spSpinor(foo, foo, foo, foo);
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp = spSpinorAdd(tmp, in[i]);
	}
	out[get_global_id(0)] = tmp;
}
__kernel void writeSpSpinorRestricted(__global spSpinor * const restrict out, const float in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		spComplex bla = make_spComplex(in, in);
		spSu3vec foo = make_spSu3vec(bla, bla, bla);
		out[i] = make_spSpinor(foo, foo, foo, foo);
	}
}

spSpinor getSpSpinorSOA(__global const spSu3vec * const restrict in, const size_t i, const size_t stride)
{
	return make_spSpinor(in[0 * stride + i], in[1 * stride + i], in[2 * stride + i], in[3 * stride + i]);
}

void putSpSpinorSOA(__global spSu3vec * const restrict out, const size_t i, const spSpinor val, const size_t stride)
{
	out[0 * stride + i] = val.e0;
	out[1 * stride + i] = val.e1;
	out[2 * stride + i] = val.e2;
	out[3 * stride + i] = val.e3;
}

__kernel void copySpSpinorSOARestricted(__global spSu3vec * const restrict out, __global const spSu3vec * const restrict in, const ulong elems, ulong stride)
{
	stride = (stride == 0) ? elems : stride;
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		spSpinor tmp = getSpSpinorSOA(in, i, stride);
		putSpSpinorSOA(out, i, tmp, stride);
	}
}
__kernel void readSpSpinorSOARestricted(__global spSu3vec * const restrict out, __global const spSu3vec * const restrict in, const ulong elems, ulong stride)
{
	stride = (stride == 0) ? elems : stride;
	spComplex bla = make_spComplex(0.0f, 0.0f);
	spSu3vec foo = make_spSu3vec(bla, bla, bla);
	spSpinor tmp = make_spSpinor(foo, foo, foo, foo);
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp = spSpinorAdd(tmp, getSpSpinorSOA(in, i, stride));
	}
	putSpSpinorSOA(out, get_global_id(0), tmp, stride);
}
__kernel void writeSpSpinorSOARestricted(__global spSu3vec * const restrict out, const float in, const ulong elems, ulong stride)
{
	stride = (stride == 0) ? elems : stride;
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		spComplex bla = make_spComplex(in, in);
		spSu3vec foo = make_spSu3vec(bla, bla, bla);
		spSpinor tmp = make_spSpinor(foo, foo, foo, foo);
		putSpSpinorSOA(out, i, tmp, stride);
	}
}

spSpinor getSpSpinorFullSOA(__global const spComplex * const restrict in, const size_t i, const size_t stride)
{
	return make_spSpinor(make_spSu3vec(in[0 * stride + i], in[ 1 * stride + i], in[ 2 * stride + i]),
	                     make_spSu3vec(in[3 * stride + i], in[ 4 * stride + i], in[ 5 * stride + i]),
	                     make_spSu3vec(in[6 * stride + i], in[ 7 * stride + i], in[ 8 * stride + i]),
	                     make_spSu3vec(in[9 * stride + i], in[10 * stride + i], in[11 * stride + i]));
}

void putSpSpinorFullSOA(__global spComplex * const restrict out, const size_t i, const spSpinor val, const size_t stride)
{
	out[ 0 * stride + i] = val.e0.e0;
	out[ 1 * stride + i] = val.e0.e1;
	out[ 2 * stride + i] = val.e0.e2;
	out[ 3 * stride + i] = val.e1.e0;
	out[ 4 * stride + i] = val.e1.e1;
	out[ 5 * stride + i] = val.e1.e2;
	out[ 6 * stride + i] = val.e2.e0;
	out[ 7 * stride + i] = val.e2.e1;
	out[ 8 * stride + i] = val.e2.e2;
	out[ 9 * stride + i] = val.e3.e0;
	out[10 * stride + i] = val.e3.e1;
	out[11 * stride + i] = val.e3.e2;
}

__kernel void copySpSpinorFullSOARestricted(__global spComplex * const restrict out, __global const spComplex * const restrict in, const ulong elems, ulong stride)
{
	stride = (stride == 0) ? elems : stride;
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		spSpinor tmp = getSpSpinorFullSOA(in, i, stride);
		putSpSpinorFullSOA(out, i, tmp, stride);
	}
}
__kernel void readSpSpinorFullSOARestricted(__global spComplex * const restrict out, __global const spComplex * const restrict in, const ulong elems, ulong stride)
{
	stride = (stride == 0) ? elems : stride;
	spComplex bla = make_spComplex(0.0f, 0.0f);
	spSu3vec foo = make_spSu3vec(bla, bla, bla);
	spSpinor tmp = make_spSpinor(foo, foo, foo, foo);
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp = spSpinorAdd(tmp, getSpSpinorFullSOA(in, i, stride));
	}
	putSpSpinorFullSOA(out, get_global_id(0), tmp, stride);
}
__kernel void writeSpSpinorFullSOARestricted(__global spComplex * const restrict out, const float in, const ulong elems, ulong stride)
{
	stride = (stride == 0) ? elems : stride;
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		spComplex bla = make_spComplex(in, in);
		spSu3vec foo = make_spSu3vec(bla, bla, bla);
		spSpinor tmp = make_spSpinor(foo, foo, foo, foo);
		putSpSpinorFullSOA(out, i, tmp, stride);
	}
}

typedef struct {
	spSu3vecFromAligned e0;
	spSu3vecFromAligned e1;
	spSu3vecFromAligned e2;
	spSu3vecFromAligned e3;
} spSpinorFromFromAligned;

spSpinorFromFromAligned make_spSpinorFromFromAligned(const spSu3vecFromAligned e0, const spSu3vecFromAligned e1, const spSu3vecFromAligned e2, const spSu3vecFromAligned e3)
{
	return (spSpinorFromFromAligned) {e0, e1, e2, e3};
}

spSpinorFromFromAligned spSpinorFromFromAlignedAdd(const spSpinorFromFromAligned left, const spSpinorFromFromAligned right)
{
	return make_spSpinorFromFromAligned(spSu3vecFromAlignedAdd(left.e0, right.e0),
	                                    spSu3vecFromAlignedAdd(left.e1, right.e1),
	                                    spSu3vecFromAlignedAdd(left.e2, right.e2),
	                                    spSu3vecFromAlignedAdd(left.e3, right.e3));
}

spSpinorFromFromAligned getSpSpinorFullAlignedSOA(__global const alignedSpComplex * const restrict in, const size_t i, const size_t stride)
{
	return make_spSpinorFromFromAligned(make_spSu3vecFromAligned(in[0 * stride + i], in[ 1 * stride + i], in[ 2 * stride + i]),
	                                    make_spSu3vecFromAligned(in[3 * stride + i], in[ 4 * stride + i], in[ 5 * stride + i]),
	                                    make_spSu3vecFromAligned(in[6 * stride + i], in[ 7 * stride + i], in[ 8 * stride + i]),
	                                    make_spSu3vecFromAligned(in[9 * stride + i], in[10 * stride + i], in[11 * stride + i]));
}

void putSpSpinorFullAlignedSOA(__global alignedSpComplex * const restrict out, const size_t i, const spSpinorFromFromAligned val, const size_t stride)
{
	out[ 0 * stride + i] = val.e0.e0;
	out[ 1 * stride + i] = val.e0.e1;
	out[ 2 * stride + i] = val.e0.e2;
	out[ 3 * stride + i] = val.e1.e0;
	out[ 4 * stride + i] = val.e1.e1;
	out[ 5 * stride + i] = val.e1.e2;
	out[ 6 * stride + i] = val.e2.e0;
	out[ 7 * stride + i] = val.e2.e1;
	out[ 8 * stride + i] = val.e2.e2;
	out[ 9 * stride + i] = val.e3.e0;
	out[10 * stride + i] = val.e3.e1;
	out[11 * stride + i] = val.e3.e2;
}

__kernel void copySpSpinorFullAlignedSOARestricted(__global alignedSpComplex * const restrict out, __global const alignedSpComplex * const restrict in, const ulong elems, ulong stride)
{
	stride = (stride == 0) ? elems : stride;
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		spSpinorFromFromAligned tmp = getSpSpinorFullAlignedSOA(in, i, stride);
		putSpSpinorFullAlignedSOA(out, i, tmp, stride);
	}
}
__kernel void readSpSpinorFullAlignedSOARestricted(__global alignedSpComplex * const restrict out, __global const alignedSpComplex * const restrict in, const ulong elems, ulong stride)
{
	stride = (stride == 0) ? elems : stride;
	alignedSpComplex bla = make_alignedSpComplex(0.0f, 0.0f);
	spSu3vecFromAligned foo = make_spSu3vecFromAligned(bla, bla, bla);
	spSpinorFromFromAligned tmp = make_spSpinorFromFromAligned(foo, foo, foo, foo);
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp = spSpinorFromFromAlignedAdd(tmp, getSpSpinorFullAlignedSOA(in, i, stride));
	}
	putSpSpinorFullAlignedSOA(out, get_global_id(0), tmp, stride);
}
__kernel void writeSpSpinorFullAlignedSOARestricted(__global alignedSpComplex * const restrict out, const float in, const ulong elems, ulong stride)
{
	stride = (stride == 0) ? elems : stride;
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		alignedSpComplex bla = make_alignedSpComplex(in, in);
		spSu3vecFromAligned foo = make_spSu3vecFromAligned(bla, bla, bla);
		spSpinorFromFromAligned tmp = make_spSpinorFromFromAligned(foo, foo, foo, foo);
		putSpSpinorFullAlignedSOA(out, i, tmp, stride);
	}
}


spSpinor getSpSpinorViaLocal(__global const spSpinor * const restrict in, const size_t block, __local spSpinor * const restrict scratch)
{
	event_t event = async_work_group_copy((__local float2 *) scratch, (__global float2 *) &in[block * get_local_size(0)], get_local_size(0) * 12, 0);
	wait_group_events(1, &event);
	return scratch[get_local_id(0)];
}
void putSpSpinorViaLocal(__global spSpinor * const restrict out, const size_t block, const spSpinor val, __local spSpinor * const restrict scratch)
{
	scratch[get_local_id(0)] = val;
	barrier(CLK_LOCAL_MEM_FENCE);
	event_t event = async_work_group_copy((__global float2 *) &out[block * get_local_size(0)], (__local float2 *) scratch, get_local_size(0) * 12, 0);
	wait_group_events(1, &event);
}

__kernel void copySpSpinorViaLocalRestricted(__global spSpinor * const restrict out, __global const spSpinor * const restrict in, const ulong elems, __local spSpinor * const restrict scratch)
{
	for(size_t i = get_group_id(0); i < elems / get_num_groups(0); i += get_num_groups(0)) {
		spSpinor tmp = getSpSpinorViaLocal(in, i, scratch);
		putSpSpinorViaLocal(out, i, tmp, scratch);
	}
}
__kernel void readSpSpinorViaLocalRestricted(__global spSpinor * const restrict out, __global const spSpinor * const restrict in, const ulong elems, __local spSpinor * const restrict scratch)
{
	spComplex bla = make_spComplex(0.0f, 0.0f);
	spSu3vec foo = make_spSu3vec(bla, bla, bla);
	spSpinor tmp = make_spSpinor(foo, foo, foo, foo);
	for(size_t i = get_group_id(0); i < elems / get_num_groups(0); i += get_num_groups(0)) {
		tmp = spSpinorAdd(tmp, getSpSpinorViaLocal(in, i, scratch));
	}
	putSpSpinorViaLocal(out, get_global_id(0), tmp, scratch);
}
__kernel void writeSpSpinorViaLocalRestricted(__global spSpinor * const restrict out, const float in, const ulong elems, __local spSpinor * const restrict scratch)
{
	for(size_t i = get_group_id(0); i < elems / get_num_groups(0); i += get_num_groups(0)) {
		spComplex bla = make_spComplex(in, in);
		spSu3vec foo = make_spSu3vec(bla, bla, bla);
		spSpinor tmp = make_spSpinor(foo, foo, foo, foo);
		putSpSpinorViaLocal(out, i, tmp, scratch);
	}
}

typedef struct {
	aligned8SpSu3vec e0;
	aligned8SpSu3vec e1;
	aligned8SpSu3vec e2;
	aligned8SpSu3vec e3;
} spSpinorFromAligned;

spSpinorFromAligned make_spSpinorFromAligned(const aligned8SpSu3vec e0, const aligned8SpSu3vec e1, const aligned8SpSu3vec e2, const aligned8SpSu3vec e3)
{
	return (spSpinorFromAligned) {e0, e1, e2, e3};
}
spSpinorFromAligned spSpinorFromAlignedAdd(const spSpinorFromAligned left, const spSpinorFromAligned right) {
	return make_spSpinorFromAligned(
		aligned8SpSu3vecAdd(left.e0, right.e0),
		aligned8SpSu3vecAdd(left.e1, right.e1),
		aligned8SpSu3vecAdd(left.e2, right.e2),
		aligned8SpSu3vecAdd(left.e3, right.e3)
	);
}

__kernel void copySpSpinorFromAlignedRestricted(__global spSpinorFromAligned * const restrict out, __global const spSpinorFromAligned * const restrict in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = in[i];
	}
}
__kernel void readSpSpinorFromAlignedRestricted(__global spSpinorFromAligned * const restrict out, __global const spSpinorFromAligned * const restrict in, const ulong elems)
{
	spComplex bla = make_spComplex(0.0f, 0.0f);
	aligned8SpSu3vec foo = make_aligned8SpSu3vec(bla, bla, bla);
	spSpinorFromAligned tmp = make_spSpinorFromAligned(foo, foo, foo, foo);
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp = spSpinorFromAlignedAdd(tmp, in[i]);
	}
	out[get_global_id(0)] = tmp;
}
__kernel void writeSpSpinorFromAlignedRestricted(__global spSpinorFromAligned * const restrict out, const float in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		spComplex bla = make_spComplex(in, in);
		aligned8SpSu3vec foo = make_aligned8SpSu3vec(bla, bla, bla);
		out[i] = make_spSpinorFromAligned(foo, foo, foo, foo);
	}
}

spSpinorFromAligned getSpSpinorFromAlignedSOA(__global const aligned8SpSu3vec * const restrict in, const size_t i, const size_t stride)
{
	return make_spSpinorFromAligned(in[0 * stride + i], in[1 * stride + i], in[2 * stride + i], in[3 * stride + i]);
}

void putSpSpinorFromAlignedSOA(__global aligned8SpSu3vec * const restrict out, const size_t i, const spSpinorFromAligned val, const size_t stride)
{
	out[0 * stride + i] = val.e0;
	out[1 * stride + i] = val.e1;
	out[2 * stride + i] = val.e2;
	out[3 * stride + i] = val.e3;
}

__kernel void copySpSpinorFromAlignedSOARestricted(__global aligned8SpSu3vec * const restrict out, __global const aligned8SpSu3vec * const restrict in, const ulong elems, ulong stride)
{
	stride = (stride == 0) ? elems : stride;
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		spSpinorFromAligned tmp = getSpSpinorFromAlignedSOA(in, i, stride);
		putSpSpinorFromAlignedSOA(out, i, tmp, stride);
	}
}
__kernel void readSpSpinorFromAlignedSOARestricted(__global aligned8SpSu3vec * const restrict out, __global const aligned8SpSu3vec * const restrict in, const ulong elems, ulong stride)
{
	stride = (stride == 0) ? elems : stride;
	spComplex bla = make_spComplex(0.0f, 0.0f);
	aligned8SpSu3vec foo = make_aligned8SpSu3vec(bla, bla, bla);
	spSpinorFromAligned tmp = make_spSpinorFromAligned(foo, foo, foo, foo);
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp = spSpinorFromAlignedAdd(tmp, getSpSpinorFromAlignedSOA(in, i, stride));
	}
	putSpSpinorFromAlignedSOA(out, get_global_id(0), tmp, stride);
}
__kernel void writeSpSpinorFromAlignedSOARestricted(__global aligned8SpSu3vec * const restrict out, const float in, const ulong elems, ulong stride)
{
	stride = (stride == 0) ? elems : stride;
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		spComplex bla = make_spComplex(in, in);
		aligned8SpSu3vec foo = make_aligned8SpSu3vec(bla, bla, bla);
		spSpinorFromAligned tmp = make_spSpinorFromAligned(foo, foo, foo, foo);
		putSpSpinorFromAlignedSOA(out, i, tmp, stride);
	}
}

spSpinorFromAligned getSpSpinorFromAlignedViaLocal(__global const spSpinorFromAligned * const restrict in, const size_t block, __local spSpinorFromAligned * const restrict scratch)
{
	event_t event = async_work_group_copy((__local float2 *) scratch, (__global float2 *) &in[block * get_local_size(0)], get_local_size(0) * 12, 0);
	wait_group_events(1, &event);
	return scratch[get_local_id(0)];
}
void putSpSpinorFromAlignedViaLocal(__global spSpinorFromAligned * const restrict out, const size_t block, const spSpinorFromAligned val, __local spSpinorFromAligned * const restrict scratch)
{
	scratch[get_local_id(0)] = val;
	barrier(CLK_LOCAL_MEM_FENCE);
	event_t event = async_work_group_copy((__global float2 *) &out[block * get_local_size(0)], (__local float2 *) scratch, get_local_size(0) * 12, 0);
	wait_group_events(1, &event);
}

__kernel void copySpSpinorFromAlignedViaLocalRestricted(__global spSpinorFromAligned * const restrict out, __global const spSpinorFromAligned * const restrict in, const ulong elems, __local spSpinorFromAligned * const restrict scratch)
{
	for(size_t i = get_group_id(0); i < elems / get_num_groups(0); i += get_num_groups(0)) {
		spSpinorFromAligned tmp = getSpSpinorFromAlignedViaLocal(in, i, scratch);
		putSpSpinorFromAlignedViaLocal(out, i, tmp, scratch);
	}
}
__kernel void readSpSpinorFromAlignedViaLocalRestricted(__global spSpinorFromAligned * const restrict out, __global const spSpinorFromAligned * const restrict in, const ulong elems, __local spSpinorFromAligned * const restrict scratch)
{
	spComplex bla = make_spComplex(0.0f, 0.0f);
	aligned8SpSu3vec foo = make_aligned8SpSu3vec(bla, bla, bla);
	spSpinorFromAligned tmp = make_spSpinorFromAligned(foo, foo, foo, foo);
	for(size_t i = get_group_id(0); i < elems / get_num_groups(0); i += get_num_groups(0)) {
		tmp = spSpinorFromAlignedAdd(tmp, getSpSpinorFromAlignedViaLocal(in, i, scratch));
	}
	putSpSpinorFromAlignedViaLocal(out, get_global_id(0), tmp, scratch);
}
__kernel void writeSpSpinorFromAlignedViaLocalRestricted(__global spSpinorFromAligned * const restrict out, const float in, const ulong elems, __local spSpinorFromAligned * const restrict scratch)
{
	for(size_t i = get_group_id(0); i < elems / get_num_groups(0); i += get_num_groups(0)) {
		spComplex bla = make_spComplex(in, in);
		aligned8SpSu3vec foo = make_aligned8SpSu3vec(bla, bla, bla);
		spSpinorFromAligned tmp = make_spSpinorFromAligned(foo, foo, foo, foo);
		putSpSpinorFromAlignedViaLocal(out, i, tmp, scratch);
	}
}


typedef struct {
	aligned8SpSu3vec e0;
	aligned8SpSu3vec e1;
	aligned8SpSu3vec e2;
	aligned8SpSu3vec e3;
} __attribute__((aligned(8))) aligned8SpSpinor;

aligned8SpSpinor make_aligned8SpSpinor(const aligned8SpSu3vec e0, const aligned8SpSu3vec e1, const aligned8SpSu3vec e2, const aligned8SpSu3vec e3) {
	return (aligned8SpSpinor) {e0, e1, e2, e3};
}

aligned8SpSpinor aligned8SpSpinorAdd(const aligned8SpSpinor left, const aligned8SpSpinor right) {
	return make_aligned8SpSpinor(
		aligned8SpSu3vecAdd(left.e0, right.e0),
		aligned8SpSu3vecAdd(left.e1, right.e1),
		aligned8SpSu3vecAdd(left.e2, right.e2),
		aligned8SpSu3vecAdd(left.e3, right.e3)
	);
}

__kernel void copyAligned8SpSpinorRestricted(__global aligned8SpSpinor * const restrict out, __global const aligned8SpSpinor * const restrict in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = in[i];
	}
}
__kernel void readAligned8SpSpinorRestricted(__global aligned8SpSpinor * const restrict out, __global const aligned8SpSpinor * const restrict in, const ulong elems)
{
	spComplex bla = make_spComplex(0.0f, 0.0f);
	aligned8SpSu3vec foo = make_aligned8SpSu3vec(bla, bla, bla);
	aligned8SpSpinor tmp = make_aligned8SpSpinor(foo, foo, foo, foo);
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp = aligned8SpSpinorAdd(tmp, in[i]);
	}
	out[get_global_id(0)] = tmp;
}
__kernel void writeAligned8SpSpinorRestricted(__global aligned8SpSpinor * const restrict out, const float in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		spComplex bla = make_spComplex(in, in);
		aligned8SpSu3vec foo = make_aligned8SpSu3vec(bla, bla, bla);
		out[i] = make_aligned8SpSpinor(foo, foo, foo, foo);
	}
}


typedef struct {
	aligned8SpSu3vec e0;
	aligned8SpSu3vec e1;
	aligned8SpSu3vec e2;
	aligned8SpSu3vec e3;
} __attribute__((aligned(16))) aligned16SpSpinor;

aligned16SpSpinor make_aligned16SpSpinor(const aligned8SpSu3vec e0, const aligned8SpSu3vec e1, const aligned8SpSu3vec e2, const aligned8SpSu3vec e3) {
	return (aligned16SpSpinor) {e0, e1, e2, e3};
}

aligned16SpSpinor aligned16SpSpinorAdd(const aligned16SpSpinor left, const aligned16SpSpinor right) {
	return make_aligned16SpSpinor(
		aligned8SpSu3vecAdd(left.e0, right.e0),
		aligned8SpSu3vecAdd(left.e1, right.e1),
		aligned8SpSu3vecAdd(left.e2, right.e2),
		aligned8SpSu3vecAdd(left.e3, right.e3)
	);
}

__kernel void copyAligned16SpSpinorRestricted(__global aligned16SpSpinor * const restrict out, __global const aligned16SpSpinor * const restrict in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = in[i];
	}
}
__kernel void readAligned16SpSpinorRestricted(__global aligned16SpSpinor * const restrict out, __global const aligned16SpSpinor * const restrict in, const ulong elems)
{
	spComplex bla = make_spComplex(0.0f, 0.0f);
	aligned8SpSu3vec foo = make_aligned8SpSu3vec(bla, bla, bla);
	aligned16SpSpinor tmp = make_aligned16SpSpinor(foo, foo, foo, foo);
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp = aligned16SpSpinorAdd(tmp, in[i]);
	}
	out[get_global_id(0)] = tmp;
}
__kernel void writeAligned16SpSpinorRestricted(__global aligned16SpSpinor * const restrict out, const float in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		spComplex bla = make_spComplex(in, in);
		aligned8SpSu3vec foo = make_aligned8SpSu3vec(bla, bla, bla);
		out[i] = make_aligned16SpSpinor(foo, foo, foo, foo);
	}
}


typedef struct {
	aligned8SpSu3vec e0;
	aligned8SpSu3vec e1;
	aligned8SpSu3vec e2;
	aligned8SpSu3vec e3;
} __attribute__((aligned(32))) aligned32SpSpinor;

aligned32SpSpinor make_aligned32SpSpinor(const aligned8SpSu3vec e0, const aligned8SpSu3vec e1, const aligned8SpSu3vec e2, const aligned8SpSu3vec e3) {
	return (aligned32SpSpinor) {e0, e1, e2, e3};
}

aligned32SpSpinor aligned32SpSpinorAdd(const aligned32SpSpinor left, const aligned32SpSpinor right) {
	return make_aligned32SpSpinor(
		aligned8SpSu3vecAdd(left.e0, right.e0),
		aligned8SpSu3vecAdd(left.e1, right.e1),
		aligned8SpSu3vecAdd(left.e2, right.e2),
		aligned8SpSu3vecAdd(left.e3, right.e3)
	);
}

__kernel void copyAligned32SpSpinorRestricted(__global aligned32SpSpinor * const restrict out, __global const aligned32SpSpinor * const restrict in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = in[i];
	}
}
__kernel void readAligned32SpSpinorRestricted(__global aligned32SpSpinor * const restrict out, __global const aligned32SpSpinor * const restrict in, const ulong elems)
{
	spComplex bla = make_spComplex(0.0f, 0.0f);
	aligned8SpSu3vec foo = make_aligned8SpSu3vec(bla, bla, bla);
	aligned32SpSpinor tmp = make_aligned32SpSpinor(foo, foo, foo, foo);
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp = aligned32SpSpinorAdd(tmp, in[i]);
	}
	out[get_global_id(0)] = tmp;
}
__kernel void writeAligned32SpSpinorRestricted(__global aligned32SpSpinor * const restrict out, const float in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		spComplex bla = make_spComplex(in, in);
		aligned8SpSu3vec foo = make_aligned8SpSu3vec(bla, bla, bla);
		out[i] = make_aligned32SpSpinor(foo, foo, foo, foo);
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


/*
 * Double precisoin SU3 vectors
 * We always base on aligned types, even if we don't explicitly align the struct
 */

typedef struct {
	alignedDpComplex e0;
	alignedDpComplex e1;
	alignedDpComplex e2;
} dpSu3vec;

dpSu3vec make_dpSu3vec(const alignedDpComplex e0, const alignedDpComplex e1, const alignedDpComplex e2) {
	return (dpSu3vec) {e0, e1, e2};
}

dpSu3vec dpSu3vecAdd(const dpSu3vec left, const dpSu3vec right) {
	return make_dpSu3vec(
		alignedDpComplexAdd(left.e0, right.e0),
		alignedDpComplexAdd(left.e1, right.e1),
		alignedDpComplexAdd(left.e2, right.e2)
	);
}

__kernel void copyDpSu3vec(__global dpSu3vec * out, __global dpSu3vec * in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = in[i];
	}
}
__kernel void readDpSu3vec(__global dpSu3vec * out, __global dpSu3vec * in, const ulong elems)
{
	alignedDpComplex bla = make_alignedDpComplex(0.0f, 0.0f);
	dpSu3vec tmp = make_dpSu3vec(bla, bla, bla);
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp = dpSu3vecAdd(tmp, in[i]);
	}
	out[get_global_id(0)] = tmp;
}
__kernel void writeDpSu3vec(__global dpSu3vec * out, const float in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		alignedDpComplex bla = make_alignedDpComplex(in, in);
		out[i] = make_dpSu3vec(bla, bla, bla);
	}
}

__kernel void copyDpSu3vecRestricted(__global dpSu3vec * const restrict out, __global const dpSu3vec * const restrict in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = in[i];
	}
}
__kernel void readDpSu3vecRestricted(__global dpSu3vec * const restrict out, __global const dpSu3vec * const restrict in, const ulong elems)
{
	alignedDpComplex bla = make_alignedDpComplex(0.0f, 0.0f);
	dpSu3vec tmp = make_dpSu3vec(bla, bla, bla);
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp = dpSu3vecAdd(tmp, in[i]);
	}
	out[get_global_id(0)] = tmp;
}
__kernel void writeDpSu3vecRestricted(__global dpSu3vec * const restrict out, const float in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		alignedDpComplex bla = make_alignedDpComplex(in, in);
		out[i] = make_dpSu3vec(bla, bla, bla);
	}
}

typedef struct {
	alignedDpComplex e0;
	alignedDpComplex e1;
	alignedDpComplex e2;
} __attribute((aligned(16))) aligned16DpSu3vec;

aligned16DpSu3vec make_aligned16DpSu3vec(const alignedDpComplex e0, const alignedDpComplex e1, const alignedDpComplex e2) {
	return (aligned16DpSu3vec) {e0, e1, e2};
}

aligned16DpSu3vec aligned16DpSu3vecAdd(const aligned16DpSu3vec left, const aligned16DpSu3vec right) {
	return make_aligned16DpSu3vec(
		alignedDpComplexAdd(left.e0, right.e0),
		alignedDpComplexAdd(left.e1, right.e1),
		alignedDpComplexAdd(left.e2, right.e2)
	);
}

__kernel void copyAligned16DpSu3vecRestricted(__global aligned16DpSu3vec * const restrict out, __global const aligned16DpSu3vec * const restrict in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = in[i];
	}
}
__kernel void readAligned16DpSu3vecRestricted(__global aligned16DpSu3vec * const restrict out, __global const aligned16DpSu3vec * const restrict in, const ulong elems)
{
	alignedDpComplex bla = make_alignedDpComplex(0.0f, 0.0f);
	aligned16DpSu3vec tmp = make_aligned16DpSu3vec(bla, bla, bla);
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp = aligned16DpSu3vecAdd(tmp, in[i]);
	}
	out[get_global_id(0)] = tmp;
}
__kernel void writeAligned16DpSu3vecRestricted(__global aligned16DpSu3vec * const restrict out, const float in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		alignedDpComplex bla = make_alignedDpComplex(in, in);
		out[i] = make_aligned16DpSu3vec(bla, bla, bla);
	}
}

typedef struct {
	alignedDpComplex e0;
	alignedDpComplex e1;
	alignedDpComplex e2;
} __attribute((aligned(32))) aligned32DpSu3vec;

aligned32DpSu3vec make_aligned32DpSu3vec(const alignedDpComplex e0, const alignedDpComplex e1, const alignedDpComplex e2) {
	return (aligned32DpSu3vec) {e0, e1, e2};
}

aligned32DpSu3vec aligned32DpSu3vecAdd(const aligned32DpSu3vec left, const aligned32DpSu3vec right) {
	return make_aligned32DpSu3vec(
		alignedDpComplexAdd(left.e0, right.e0),
		alignedDpComplexAdd(left.e1, right.e1),
		alignedDpComplexAdd(left.e2, right.e2)
	);
}

__kernel void copyAligned32DpSu3vecRestricted(__global aligned32DpSu3vec * const restrict out, __global const aligned32DpSu3vec * const restrict in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = in[i];
	}
}
__kernel void readAligned32DpSu3vecRestricted(__global aligned32DpSu3vec * const restrict out, __global const aligned32DpSu3vec * const restrict in, const ulong elems)
{
	alignedDpComplex bla = make_alignedDpComplex(0.0f, 0.0f);
	aligned32DpSu3vec tmp = make_aligned32DpSu3vec(bla, bla, bla);
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp = aligned32DpSu3vecAdd(tmp, in[i]);
	}
	out[get_global_id(0)] = tmp;
}
__kernel void writeAligned32DpSu3vecRestricted(__global aligned32DpSu3vec * const restrict out, const float in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		alignedDpComplex bla = make_alignedDpComplex(in, in);
		out[i] = make_aligned32DpSu3vec(bla, bla, bla);
	}
}

dpSu3vec getDpSu3vecSOA(__global const alignedDpComplex * const restrict in, const size_t i, const size_t stride)
{
	return make_dpSu3vec(in[0 * stride + i], in[ 1 * stride + i], in[ 2 * stride + i]);
}

void putDpSu3vecSOA(__global alignedDpComplex * const restrict out, const size_t i, const dpSu3vec val, const size_t stride)
{
	out[ 0 * stride + i] = val.e0;
	out[ 1 * stride + i] = val.e1;
	out[ 2 * stride + i] = val.e2;
}

__kernel void copyDpSu3vecSOARestricted(__global alignedDpComplex * const restrict out, __global const alignedDpComplex * const restrict in, const ulong elems, ulong stride)
{
	stride = (stride == 0) ? elems : stride;
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		dpSu3vec tmp = getDpSu3vecSOA(in, i, stride);
		putDpSu3vecSOA(out, i, tmp, stride);
	}
}
__kernel void readDpSu3vecSOARestricted(__global alignedDpComplex * const restrict out, __global const alignedDpComplex * const restrict in, const ulong elems, ulong stride)
{
	stride = (stride == 0) ? elems : stride;
	alignedDpComplex bla = make_alignedDpComplex(0.0f, 0.0f);
	dpSu3vec tmp = make_dpSu3vec(bla, bla, bla);
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp = dpSu3vecAdd(tmp, getDpSu3vecSOA(in, i, stride));
	}
	putDpSu3vecSOA(out, get_global_id(0), tmp, stride);
}
__kernel void writeDpSu3vecSOARestricted(__global alignedDpComplex * const restrict out, const float in, const ulong elems, ulong stride)
{
	stride = (stride == 0) ? elems : stride;
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		alignedDpComplex bla = make_alignedDpComplex(in, in);
		dpSu3vec tmp = make_dpSu3vec(bla, bla, bla);
		putDpSu3vecSOA(out, i, tmp, stride);
	}
}


dpSu3vec getDpSu3vecFullSOA(__global const double * const restrict in, const size_t i, const size_t stride)
{
	return make_dpSu3vec(make_alignedDpComplex(in[0 * stride + i], in[1 * stride + i]),
	                     make_alignedDpComplex(in[2 * stride + i], in[3 * stride + i]),
	                     make_alignedDpComplex(in[4 * stride + i], in[5 * stride + i]));
}
void putDpSu3vecFullSOA(__global double * const restrict out, const size_t i, const dpSu3vec val, const size_t stride)
{
	out[ 0 * stride + i] = val.e0.re;
	out[ 1 * stride + i] = val.e0.im;
	out[ 2 * stride + i] = val.e1.re;
	out[ 3 * stride + i] = val.e1.im;
	out[ 4 * stride + i] = val.e2.re;
	out[ 5 * stride + i] = val.e2.im;
}

__kernel void copyDpSu3vecFullSOARestricted(__global double * const restrict out, __global const double * const restrict in, const ulong elems, ulong stride)
{
	stride = (stride == 0) ? elems : stride;
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		dpSu3vec tmp = getDpSu3vecFullSOA(in, i, stride);
		putDpSu3vecFullSOA(out, i, tmp, stride);
	}
}
__kernel void readDpSu3vecFullSOARestricted(__global double * const restrict out, __global const double * const restrict in, const ulong elems, ulong stride)
{
	stride = (stride == 0) ? elems : stride;
	alignedDpComplex bla = make_alignedDpComplex(0.0f, 0.0f);
	dpSu3vec tmp = make_dpSu3vec(bla, bla, bla);
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp = dpSu3vecAdd(tmp, getDpSu3vecFullSOA(in, i, stride));
	}
	putDpSu3vecFullSOA(out, get_global_id(0), tmp, stride);
}
__kernel void writeDpSu3vecFullSOARestricted(__global double * const restrict out, const float in, const ulong elems, ulong stride)
{
	stride = (stride == 0) ? elems : stride;
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		alignedDpComplex bla = make_alignedDpComplex(in, in);
		dpSu3vec tmp = make_dpSu3vec(bla, bla, bla);
		putDpSu3vecFullSOA(out, i, tmp, stride);
	}
}


/*
 * Double precisoin SU3 matrices
 * We always base on aligned types, even if we don't explicitly align the struct
 */

typedef struct {
	alignedDpComplex e00, e01, e02;
	alignedDpComplex e10, e11, e12;
	alignedDpComplex e20, e21, e22;
} dpSu3;

dpSu3 make_dpSu3(const alignedDpComplex e00, const alignedDpComplex e01, const alignedDpComplex e02,
                 const alignedDpComplex e10, const alignedDpComplex e11, const alignedDpComplex e12,
                 const alignedDpComplex e20, const alignedDpComplex e21, const alignedDpComplex e22) {
	return (dpSu3) {e00, e01, e02,
	                e10, e11, e12,
	                e20, e21, e22};
}

dpSu3 dpSu3Add(const dpSu3 left, const dpSu3 right) {
	return make_dpSu3(
		alignedDpComplexAdd(left.e00, right.e00),
		alignedDpComplexAdd(left.e01, right.e01),
		alignedDpComplexAdd(left.e02, right.e02),
		alignedDpComplexAdd(left.e10, right.e10),
		alignedDpComplexAdd(left.e11, right.e11),
		alignedDpComplexAdd(left.e12, right.e12),
		alignedDpComplexAdd(left.e20, right.e20),
		alignedDpComplexAdd(left.e21, right.e21),
		alignedDpComplexAdd(left.e22, right.e22)
	);
}

__kernel void copyDpSu3(__global dpSu3 * out, __global dpSu3 * in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = in[i];
	}
}
__kernel void readDpSu3(__global dpSu3 * out, __global dpSu3 * in, const ulong elems)
{
	alignedDpComplex bla = make_alignedDpComplex(0.0f, 0.0f);
	dpSu3 tmp = make_dpSu3(bla, bla, bla, bla, bla, bla, bla, bla, bla);
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp = dpSu3Add(tmp, in[i]);
	}
	out[get_global_id(0)] = tmp;
}
__kernel void writeDpSu3(__global dpSu3 * out, const float in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		alignedDpComplex bla = make_alignedDpComplex(in, in);
		out[i] = make_dpSu3(bla, bla, bla, bla, bla, bla, bla, bla, bla);
	}
}

__kernel void copyDpSu3Restricted(__global dpSu3 * const restrict out, __global const dpSu3 * const restrict in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = in[i];
	}
}
__kernel void readDpSu3Restricted(__global dpSu3 * const restrict out, __global const dpSu3 * const restrict in, const ulong elems)
{
	alignedDpComplex bla = make_alignedDpComplex(0.0f, 0.0f);
	dpSu3 tmp = make_dpSu3(bla, bla, bla, bla, bla, bla, bla, bla, bla);
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp = dpSu3Add(tmp, in[i]);
	}
	out[get_global_id(0)] = tmp;
}
__kernel void writeDpSu3Restricted(__global dpSu3 * const restrict out, const float in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		alignedDpComplex bla = make_alignedDpComplex(in, in);
		out[i] = make_dpSu3(bla, bla, bla, bla, bla, bla, bla, bla, bla);
	}
}

typedef struct {
	alignedDpComplex e00, e01, e02;
	alignedDpComplex e10, e11, e12;
	alignedDpComplex e20, e21, e22;
} __attribute((aligned(16))) aligned16DpSu3;

aligned16DpSu3 make_aligned16DpSu3(const alignedDpComplex e00, const alignedDpComplex e01, const alignedDpComplex e02,
                                   const alignedDpComplex e10, const alignedDpComplex e11, const alignedDpComplex e12,
                                   const alignedDpComplex e20, const alignedDpComplex e21, const alignedDpComplex e22) {
	return (aligned16DpSu3) {e00, e01, e02,
	                         e10, e11, e12,
	                         e20, e21, e22};
}

aligned16DpSu3 aligned16DpSu3Add(const aligned16DpSu3 left, const aligned16DpSu3 right) {
	return make_aligned16DpSu3(
		alignedDpComplexAdd(left.e00, right.e00),
		alignedDpComplexAdd(left.e01, right.e01),
		alignedDpComplexAdd(left.e02, right.e02),
		alignedDpComplexAdd(left.e10, right.e10),
		alignedDpComplexAdd(left.e11, right.e11),
		alignedDpComplexAdd(left.e12, right.e12),
		alignedDpComplexAdd(left.e20, right.e20),
		alignedDpComplexAdd(left.e21, right.e21),
		alignedDpComplexAdd(left.e22, right.e22)
	);
}


__kernel void copyAligned16DpSu3Restricted(__global aligned16DpSu3 * const restrict out, __global const aligned16DpSu3 * const restrict in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = in[i];
	}
}
__kernel void readAligned16DpSu3Restricted(__global aligned16DpSu3 * const restrict out, __global const aligned16DpSu3 * const restrict in, const ulong elems)
{
	alignedDpComplex bla = make_alignedDpComplex(0.0f, 0.0f);
	aligned16DpSu3 tmp = make_aligned16DpSu3(bla, bla, bla, bla, bla, bla, bla, bla, bla);
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp = aligned16DpSu3Add(tmp, in[i]);
	}
	out[get_global_id(0)] = tmp;
}
__kernel void writeAligned16DpSu3Restricted(__global aligned16DpSu3 * const restrict out, const float in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		alignedDpComplex bla = make_alignedDpComplex(in, in);
		out[i] = make_aligned16DpSu3(bla, bla, bla, bla, bla, bla, bla, bla, bla);
	}
}

typedef struct {
	alignedDpComplex e00, e01, e02;
	alignedDpComplex e10, e11, e12;
	alignedDpComplex e20, e21, e22;
} __attribute((aligned(32))) aligned32DpSu3;

aligned32DpSu3 make_aligned32DpSu3(const alignedDpComplex e00, const alignedDpComplex e01, const alignedDpComplex e02,
                                   const alignedDpComplex e10, const alignedDpComplex e11, const alignedDpComplex e12,
                                   const alignedDpComplex e20, const alignedDpComplex e21, const alignedDpComplex e22) {
	return (aligned32DpSu3) {e00, e01, e02,
	                         e10, e11, e12,
	                         e20, e21, e22};
}

aligned32DpSu3 aligned32DpSu3Add(const aligned32DpSu3 left, const aligned32DpSu3 right) {
	return make_aligned32DpSu3(
		alignedDpComplexAdd(left.e00, right.e00),
		alignedDpComplexAdd(left.e01, right.e01),
		alignedDpComplexAdd(left.e02, right.e02),
		alignedDpComplexAdd(left.e10, right.e10),
		alignedDpComplexAdd(left.e11, right.e11),
		alignedDpComplexAdd(left.e12, right.e12),
		alignedDpComplexAdd(left.e20, right.e20),
		alignedDpComplexAdd(left.e21, right.e21),
		alignedDpComplexAdd(left.e22, right.e22)
	);
}

__kernel void copyAligned32DpSu3Restricted(__global aligned32DpSu3 * const restrict out, __global const aligned32DpSu3 * const restrict in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = in[i];
	}
}
__kernel void readAligned32DpSu3Restricted(__global aligned32DpSu3 * const restrict out, __global const aligned32DpSu3 * const restrict in, const ulong elems)
{
	alignedDpComplex bla = make_alignedDpComplex(0.0f, 0.0f);
	aligned32DpSu3 tmp = make_aligned32DpSu3(bla, bla, bla, bla, bla, bla, bla, bla, bla);
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp = aligned32DpSu3Add(tmp, in[i]);
	}
	out[get_global_id(0)] = tmp;
}
__kernel void writeAligned32DpSu3Restricted(__global aligned32DpSu3 * const restrict out, const float in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		alignedDpComplex bla = make_alignedDpComplex(in, in);
		out[i] = make_aligned32DpSu3(bla, bla, bla, bla, bla, bla, bla, bla, bla);
	}
}

dpSu3 getDpSu3SOA(__global const alignedDpComplex * const restrict in, const size_t i, const size_t stride)
{
	return make_dpSu3(in[0 * stride + i], in[1 * stride + i], in[2 * stride + i],
                      in[3 * stride + i], in[4 * stride + i], in[5 * stride + i],
                      in[6 * stride + i], in[7 * stride + i], in[8 * stride + i]);
}

void putDpSu3SOA(__global alignedDpComplex * const restrict out, const size_t i, const dpSu3 val, const size_t stride)
{
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

__kernel void copyDpSu3SOARestricted(__global alignedDpComplex * const restrict out, __global const alignedDpComplex * const restrict in, const ulong elems, ulong stride)
{
	stride = (stride == 0) ? elems : stride;
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		dpSu3 tmp = getDpSu3SOA(in, i, stride);
		putDpSu3SOA(out, i, tmp, stride);
	}
}
__kernel void readDpSu3SOARestricted(__global alignedDpComplex * const restrict out, __global const alignedDpComplex * const restrict in, const ulong elems, ulong stride)
{
	stride = (stride == 0) ? elems : stride;
	alignedDpComplex bla = make_alignedDpComplex(0.0f, 0.0f);
	dpSu3 tmp = make_dpSu3(bla, bla, bla, bla, bla, bla, bla, bla, bla);
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp = dpSu3Add(tmp, getDpSu3SOA(in, i, stride));
	}
	putDpSu3SOA(out, get_global_id(0), tmp, stride);
}
__kernel void writeDpSu3SOARestricted(__global alignedDpComplex * const restrict out, const float in, const ulong elems, ulong stride)
{
	stride = (stride == 0) ? elems : stride;
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		alignedDpComplex bla = make_alignedDpComplex(in, in);
		dpSu3 tmp = make_dpSu3(bla, bla, bla, bla, bla, bla, bla, bla, bla);
		putDpSu3SOA(out, i, tmp, stride);
	}
}


dpSu3 getDpSu3FullSOA(__global const double * const restrict in, const size_t i, const size_t stride)
{
	return make_dpSu3(make_alignedDpComplex(in[ 0 * stride + i], in[ 1 * stride + i]),
                      make_alignedDpComplex(in[ 2 * stride + i], in[ 3 * stride + i]),
                      make_alignedDpComplex(in[ 4 * stride + i], in[ 5 * stride + i]),
                      make_alignedDpComplex(in[ 6 * stride + i], in[ 7 * stride + i]),
                      make_alignedDpComplex(in[ 8 * stride + i], in[ 9 * stride + i]),
                      make_alignedDpComplex(in[10 * stride + i], in[11 * stride + i]),
                      make_alignedDpComplex(in[12 * stride + i], in[13 * stride + i]),
                      make_alignedDpComplex(in[14 * stride + i], in[15 * stride + i]),
                      make_alignedDpComplex(in[16 * stride + i], in[17 * stride + i]));
}
void putDpSu3FullSOA(__global double * const restrict out, const size_t i, const dpSu3 val, const size_t stride)
{
	out[ 0 * stride + i] = val.e00.re;
	out[ 1 * stride + i] = val.e00.im;
	out[ 2 * stride + i] = val.e01.re;
	out[ 3 * stride + i] = val.e01.im;
	out[ 4 * stride + i] = val.e02.re;
	out[ 5 * stride + i] = val.e02.im;
	out[ 6 * stride + i] = val.e10.re;
	out[ 7 * stride + i] = val.e10.im;
	out[ 8 * stride + i] = val.e11.re;
	out[ 9 * stride + i] = val.e11.im;
	out[10 * stride + i] = val.e12.re;
	out[11 * stride + i] = val.e12.im;
	out[12 * stride + i] = val.e20.re;
	out[13 * stride + i] = val.e20.im;
	out[14 * stride + i] = val.e21.re;
	out[15 * stride + i] = val.e21.im;
	out[16 * stride + i] = val.e22.re;
	out[17 * stride + i] = val.e22.im;
}

__kernel void copyDpSu3FullSOARestricted(__global double * const restrict out, __global const double * const restrict in, const ulong elems, ulong stride)
{
	stride = (stride == 0) ? elems : stride;
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		dpSu3 tmp = getDpSu3FullSOA(in, i, elems);
		putDpSu3FullSOA(out, i, tmp, stride);
	}
}
__kernel void readDpSu3FullSOARestricted(__global double * const restrict out, __global const double * const restrict in, const ulong elems, ulong stride)
{
	stride = (stride == 0) ? elems : stride;
	alignedDpComplex bla = make_alignedDpComplex(0.0f, 0.0f);
	dpSu3 tmp = make_dpSu3(bla, bla, bla, bla, bla, bla, bla, bla, bla);
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp = dpSu3Add(tmp, getDpSu3FullSOA(in, i, stride));
	}
	putDpSu3FullSOA(out, get_global_id(0), tmp, stride);
}
__kernel void writeDpSu3FullSOARestricted(__global double * const restrict out, const float in, const ulong elems, ulong stride)
{
	stride = (stride == 0) ? elems : stride;
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		alignedDpComplex bla = make_alignedDpComplex(in, in);
		dpSu3 tmp = make_dpSu3(bla, bla, bla, bla, bla, bla, bla, bla, bla);
		putDpSu3FullSOA(out, i, tmp, stride);
	}
}


/*
 * Double precisoin Spinors
 * We always base on aligned types, even if we don't explicitly align the struct
 */

typedef struct {
	aligned16DpSu3vec e0;
	aligned16DpSu3vec e1;
	aligned16DpSu3vec e2;
	aligned16DpSu3vec e3;
} dpSpinor;

dpSpinor make_dpSpinor(const aligned16DpSu3vec e0, const aligned16DpSu3vec e1, const aligned16DpSu3vec e2, const aligned16DpSu3vec e3)
{
	return (dpSpinor) {e0, e1, e2, e3};
}

dpSpinor dpSpinorAdd(const dpSpinor left, const dpSpinor right) {
	return make_dpSpinor(
		aligned16DpSu3vecAdd(left.e0, right.e0),
		aligned16DpSu3vecAdd(left.e1, right.e1),
		aligned16DpSu3vecAdd(left.e2, right.e2),
		aligned16DpSu3vecAdd(left.e3, right.e3)
	);
}

__kernel void copyDpSpinor(__global dpSpinor * out, __global dpSpinor * in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = in[i];
	}
}
__kernel void readDpSpinor(__global dpSpinor * out, __global dpSpinor * in, const ulong elems)
{
	alignedDpComplex bla = make_alignedDpComplex(0.0f, 0.0f);
	aligned16DpSu3vec foo = make_aligned16DpSu3vec(bla, bla, bla);
	dpSpinor tmp = make_dpSpinor(foo, foo, foo, foo);
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp = dpSpinorAdd(tmp, in[i]);
	}
	out[get_global_id(0)] = tmp;
}
__kernel void writeDpSpinor(__global dpSpinor * out, const float in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		alignedDpComplex bla = make_alignedDpComplex(in, in);
		aligned16DpSu3vec foo = make_aligned16DpSu3vec(bla, bla, bla);
		out[i] = make_dpSpinor(foo, foo, foo, foo);
	}
}

__kernel void copyDpSpinorRestricted(__global dpSpinor * const restrict out, __global const dpSpinor * const restrict in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = in[i];
	}
}
__kernel void readDpSpinorRestricted(__global dpSpinor * const restrict out, __global const dpSpinor * const restrict in, const ulong elems)
{
	alignedDpComplex bla = make_alignedDpComplex(0.0f, 0.0f);
	aligned16DpSu3vec foo = make_aligned16DpSu3vec(bla, bla, bla);
	dpSpinor tmp = make_dpSpinor(foo, foo, foo, foo);
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp = dpSpinorAdd(tmp, in[i]);
	}
	out[get_global_id(0)] = tmp;
}
__kernel void writeDpSpinorRestricted(__global dpSpinor * const restrict out, const float in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		alignedDpComplex bla = make_alignedDpComplex(in, in);
		aligned16DpSu3vec foo = make_aligned16DpSu3vec(bla, bla, bla);
		out[i] = make_dpSpinor(foo, foo, foo, foo);
	}
}

typedef struct {
	aligned16DpSu3vec e0;
	aligned16DpSu3vec e1;
	aligned16DpSu3vec e2;
	aligned16DpSu3vec e3;
} __attribute((aligned(16))) aligned16DpSpinor;

aligned16DpSpinor make_aligned16DpSpinor(const aligned16DpSu3vec e0, const aligned16DpSu3vec e1, const aligned16DpSu3vec e2, const aligned16DpSu3vec e3)
{
	return (aligned16DpSpinor) {e0, e1, e2, e3};
}

aligned16DpSpinor aligned16DpSpinorAdd(const aligned16DpSpinor left, const aligned16DpSpinor right) {
	return make_aligned16DpSpinor(
		aligned16DpSu3vecAdd(left.e0, right.e0),
		aligned16DpSu3vecAdd(left.e1, right.e1),
		aligned16DpSu3vecAdd(left.e2, right.e2),
		aligned16DpSu3vecAdd(left.e3, right.e3)
	);
}

__kernel void copyAligned16DpSpinorRestricted(__global aligned16DpSpinor * const restrict out, __global const aligned16DpSpinor * const restrict in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = in[i];
	}
}
__kernel void readAligned16DpSpinorRestricted(__global aligned16DpSpinor * const restrict out, __global const aligned16DpSpinor * const restrict in, const ulong elems)
{
	alignedDpComplex bla = make_alignedDpComplex(0.0f, 0.0f);
	aligned16DpSu3vec foo = make_aligned16DpSu3vec(bla, bla, bla);
	aligned16DpSpinor tmp = make_aligned16DpSpinor(foo, foo, foo, foo);
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp = aligned16DpSpinorAdd(tmp, in[i]);
	}
	out[get_global_id(0)] = tmp;
}
__kernel void writeAligned16DpSpinorRestricted(__global aligned16DpSpinor * const restrict out, const float in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		alignedDpComplex bla = make_alignedDpComplex(in, in);
		aligned16DpSu3vec foo = make_aligned16DpSu3vec(bla, bla, bla);
		out[i] = make_aligned16DpSpinor(foo, foo, foo, foo);
	}
}

typedef struct {
	aligned16DpSu3vec e0;
	aligned16DpSu3vec e1;
	aligned16DpSu3vec e2;
	aligned16DpSu3vec e3;
} __attribute((aligned(32))) aligned32DpSpinor;

aligned32DpSpinor make_aligned32DpSpinor(const aligned16DpSu3vec e0, const aligned16DpSu3vec e1, const aligned16DpSu3vec e2, const aligned16DpSu3vec e3)
{
	return (aligned32DpSpinor) {e0, e1, e2, e3};
}

aligned32DpSpinor aligned32DpSpinorAdd(const aligned32DpSpinor left, const aligned32DpSpinor right) {
	return make_aligned32DpSpinor(
		aligned16DpSu3vecAdd(left.e0, right.e0),
		aligned16DpSu3vecAdd(left.e1, right.e1),
		aligned16DpSu3vecAdd(left.e2, right.e2),
		aligned16DpSu3vecAdd(left.e3, right.e3)
	);
}

__kernel void copyAligned32DpSpinorRestricted(__global aligned32DpSpinor * const restrict out, __global const aligned32DpSpinor * const restrict in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		out[i] = in[i];
	}
}
__kernel void readAligned32DpSpinorRestricted(__global aligned32DpSpinor * const restrict out, __global const aligned32DpSpinor * const restrict in, const ulong elems)
{
	alignedDpComplex bla = make_alignedDpComplex(0.0f, 0.0f);
	aligned16DpSu3vec foo = make_aligned16DpSu3vec(bla, bla, bla);
	aligned32DpSpinor tmp = make_aligned32DpSpinor(foo, foo, foo, foo);
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp = aligned32DpSpinorAdd(tmp, in[i]);
	}
	out[get_global_id(0)] = tmp;
}
__kernel void writeAligned32DpSpinorRestricted(__global aligned32DpSpinor * const restrict out, const float in, const ulong elems)
{
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		alignedDpComplex bla = make_alignedDpComplex(in, in);
		aligned16DpSu3vec foo = make_aligned16DpSu3vec(bla, bla, bla);
		out[i] = make_aligned32DpSpinor(foo, foo, foo, foo);
	}
}


dpSpinor getDpSpinorSOA(__global const aligned16DpSu3vec * const restrict in, const size_t i, const size_t stride)
{
	return make_dpSpinor(in[0 * stride + i], in[1 * stride + i], in[2 * stride + i], in[3 * stride + i]);
}

void putDpSpinorSOA(__global aligned16DpSu3vec * const restrict out, const size_t i, const dpSpinor val, const size_t stride)
{
	out[0 * stride + i] = val.e0;
	out[1 * stride + i] = val.e1;
	out[2 * stride + i] = val.e2;
	out[3 * stride + i] = val.e3;
}

__kernel void copyDpSpinorSOARestricted(__global aligned16DpSu3vec * const restrict out, __global const aligned16DpSu3vec * const restrict in, const ulong elems, ulong stride)
{
	stride = (stride == 0) ? elems : stride;
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		dpSpinor tmp = getDpSpinorSOA(in, i, stride);
		putDpSpinorSOA(out, i, tmp, stride);
	}
}
__kernel void readDpSpinorSOARestricted(__global aligned16DpSu3vec * const restrict out, __global const aligned16DpSu3vec * const restrict in, const ulong elems, ulong stride)
{
	stride = (stride == 0) ? elems : stride;
	alignedDpComplex bla = make_alignedDpComplex(0.0f, 0.0f);
	aligned16DpSu3vec foo = make_aligned16DpSu3vec(bla, bla, bla);
	dpSpinor tmp = make_dpSpinor(foo, foo, foo, foo);
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp = dpSpinorAdd(tmp, getDpSpinorSOA(in, i, stride));
	}
	putDpSpinorSOA(out, get_global_id(0), tmp, stride);
}
__kernel void writeDpSpinorSOARestricted(__global aligned16DpSu3vec * const restrict out, const float in, const ulong elems, ulong stride)
{
	stride = (stride == 0) ? elems : stride;
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		alignedDpComplex bla = make_alignedDpComplex(in, in);
		aligned16DpSu3vec foo = make_aligned16DpSu3vec(bla, bla, bla);
		dpSpinor tmp = make_dpSpinor(foo, foo, foo, foo);
		putDpSpinorSOA(out, i, tmp, stride);
	}
}

dpSpinor getDpSpinorFullSOA(__global const alignedDpComplex * const restrict in, const size_t i, const size_t stride)
{
	return make_dpSpinor(make_aligned16DpSu3vec(in[0 * stride + i], in[ 1 * stride + i], in[ 2 * stride + i]),
                         make_aligned16DpSu3vec(in[3 * stride + i], in[ 4 * stride + i], in[ 5 * stride + i]),
                         make_aligned16DpSu3vec(in[6 * stride + i], in[ 7 * stride + i], in[ 8 * stride + i]),
                         make_aligned16DpSu3vec(in[9 * stride + i], in[10 * stride + i], in[11 * stride + i]));
}
void putDpSpinorFullSOA(__global alignedDpComplex * const restrict out, const size_t i, const dpSpinor val, const size_t stride)
{
	out[ 0 * stride + i] = val.e0.e0;
	out[ 1 * stride + i] = val.e0.e1;
	out[ 2 * stride + i] = val.e0.e2;
	out[ 3 * stride + i] = val.e1.e0;
	out[ 4 * stride + i] = val.e1.e1;
	out[ 5 * stride + i] = val.e1.e2;
	out[ 6 * stride + i] = val.e2.e0;
	out[ 7 * stride + i] = val.e2.e1;
	out[ 8 * stride + i] = val.e2.e2;
	out[ 9 * stride + i] = val.e3.e0;
	out[10 * stride + i] = val.e3.e1;
	out[11 * stride + i] = val.e3.e2;
}

__kernel void copyDpSpinorFullSOARestricted(__global alignedDpComplex * const restrict out, __global const alignedDpComplex * const restrict in, const ulong elems, ulong stride)
{
	stride = (stride == 0) ? elems : stride;
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		dpSpinor tmp = getDpSpinorFullSOA(in, i, stride);
		putDpSpinorFullSOA(out, i, tmp, stride);
	}
}
__kernel void readDpSpinorFullSOARestricted(__global alignedDpComplex * const restrict out, __global const alignedDpComplex * const restrict in, const ulong elems, ulong stride)
{
	stride = (stride == 0) ? elems : stride;
	alignedDpComplex bla = make_alignedDpComplex(0.0f, 0.0f);
	aligned16DpSu3vec foo = make_aligned16DpSu3vec(bla, bla, bla);
	dpSpinor tmp = make_dpSpinor(foo, foo, foo, foo);
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp = dpSpinorAdd(tmp, getDpSpinorFullSOA(in, i, stride));
	}
	putDpSpinorFullSOA(out, get_global_id(0), tmp, stride);
}
__kernel void writeDpSpinorFullSOARestricted(__global alignedDpComplex * const restrict out, const float in, const ulong elems, ulong stride)
{
	stride = (stride == 0) ? elems : stride;
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		alignedDpComplex bla = make_alignedDpComplex(in, in);
		aligned16DpSu3vec foo = make_aligned16DpSu3vec(bla, bla, bla);
		dpSpinor tmp = make_dpSpinor(foo, foo, foo, foo);
		putDpSpinorFullSOA(out, i, tmp, stride);
	}
}

dpSpinor getDpSpinorFullestSOA(__global const double * const restrict in, const size_t i, const size_t stride)
{
	return make_dpSpinor(make_aligned16DpSu3vec(make_alignedDpComplex(in[ 0 * stride + i], in[ 1 * stride + i]),
	                                            make_alignedDpComplex(in[ 2 * stride + i], in[ 3 * stride + i]),
	                                            make_alignedDpComplex(in[ 4 * stride + i], in[ 5 * stride + i])),
	                     make_aligned16DpSu3vec(make_alignedDpComplex(in[ 6 * stride + i], in[ 7 * stride + i]),
	                                            make_alignedDpComplex(in[ 8 * stride + i], in[ 9 * stride + i]),
	                                            make_alignedDpComplex(in[10 * stride + i], in[11 * stride + i])),
	                     make_aligned16DpSu3vec(make_alignedDpComplex(in[12 * stride + i], in[13 * stride + i]),
	                                            make_alignedDpComplex(in[14 * stride + i], in[15 * stride + i]),
	                                            make_alignedDpComplex(in[16 * stride + i], in[17 * stride + i])),
	                     make_aligned16DpSu3vec(make_alignedDpComplex(in[18 * stride + i], in[19 * stride + i]),
	                                            make_alignedDpComplex(in[20 * stride + i], in[21 * stride + i]),
	                                            make_alignedDpComplex(in[22 * stride + i], in[23 * stride + i])));
}
void putDpSpinorFullestSOA(__global double * const restrict out, const size_t i, const dpSpinor val, const size_t stride)
{
	out[ 0 * stride + i] = val.e0.e0.re;
	out[ 1 * stride + i] = val.e0.e0.im;
	out[ 2 * stride + i] = val.e0.e1.re;
	out[ 3 * stride + i] = val.e0.e1.im;
	out[ 4 * stride + i] = val.e0.e2.re;
	out[ 5 * stride + i] = val.e0.e2.im;
	out[ 6 * stride + i] = val.e1.e0.re;
	out[ 7 * stride + i] = val.e1.e0.im;
	out[ 8 * stride + i] = val.e1.e1.re;
	out[ 9 * stride + i] = val.e1.e1.im;
	out[10 * stride + i] = val.e1.e2.re;
	out[11 * stride + i] = val.e1.e2.im;
	out[12 * stride + i] = val.e2.e0.re;
	out[13 * stride + i] = val.e2.e0.im;
	out[14 * stride + i] = val.e2.e1.re;
	out[15 * stride + i] = val.e2.e1.im;
	out[16 * stride + i] = val.e2.e2.re;
	out[17 * stride + i] = val.e2.e2.im;
	out[18 * stride + i] = val.e3.e0.re;
	out[19 * stride + i] = val.e3.e0.im;
	out[20 * stride + i] = val.e3.e1.re;
	out[21 * stride + i] = val.e3.e1.im;
	out[22 * stride + i] = val.e3.e2.re;
	out[23 * stride + i] = val.e3.e2.im;
}

__kernel void copyDpSpinorFullestSOARestricted(__global double * const restrict out, __global const double * const restrict in, const ulong elems, ulong stride)
{
	stride = (stride == 0) ? elems : stride;
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		dpSpinor tmp = getDpSpinorFullestSOA(in, i, stride);
		putDpSpinorFullestSOA(out, i, tmp, stride);
	}
}
__kernel void readDpSpinorFullestSOARestricted(__global double * const restrict out, __global const double * const restrict in, const ulong elems, ulong stride)
{
	stride = (stride == 0) ? elems : stride;
	alignedDpComplex bla = make_alignedDpComplex(0.0f, 0.0f);
	aligned16DpSu3vec foo = make_aligned16DpSu3vec(bla, bla, bla);
	dpSpinor tmp = make_dpSpinor(foo, foo, foo, foo);
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		tmp = dpSpinorAdd(tmp, getDpSpinorFullestSOA(in, i, stride));
	}
	putDpSpinorFullestSOA(out, get_global_id(0), tmp, stride);
}
__kernel void writeDpSpinorFullestSOARestricted(__global double * const restrict out, const float in, const ulong elems, ulong stride)
{
	stride = (stride == 0) ? elems : stride;
	for(size_t i = get_global_id(0); i < elems; i += get_global_size(0)) {
		alignedDpComplex bla = make_alignedDpComplex(in, in);
		aligned16DpSu3vec foo = make_aligned16DpSu3vec(bla, bla, bla);
		dpSpinor tmp = make_dpSpinor(foo, foo, foo, foo);
		putDpSpinorFullestSOA(out, i, tmp, stride);
	}
}

#endif /* DOUBLE_ENABLED */
