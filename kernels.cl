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
 * (c) 2011-2012 Matthias Bach <bach@compeng.uni-frankfurt.de>
 */

/*
 * The below kernel fragments are used to generate bandwidth measurement kernels
 * for a wide variety of types. The following preprocessor directives are used
 * to control the produces set of kernels. Any boolean variable are true if defined
 * and false if undefined.
 *
 * ENABLE_DOUBLE    Enables double precision support
 * SCALAR           The scalar type to use
 * PLAIN_POINTERS   Don't augment pointers to memory with restricts and consts, potentially
 *                  helping the compiler to use caches (AMD does this on the Cypress and Cayman)
 * ENABLE_STRUCT    Build kernels for struct performance measurements.
 * STRUCT2SCALAR    A code fragment reducing the struct to something that can be assigned to the
 *                  scalar type.
 * SOA_STRIDE       The stride to use for SOA kernels.
 * STRUCT_ELEMS     The number of elements per struct.
 * BLOCKED_LOOP     Have each thread process blocks of indices instead of neighbouring threads
 *                  working neighboring indices.
 * NUM_ELEMS        The number of elements in one array.
 *
 * Obviously the struct directives only have to be specified is ENABLE_STRUCT has been defined.
 */

/*
 * Define all used macros
 */

#ifdef BLOCKED_LOOP
#define PARALLEL_FOR(i) \
size_t i_blocking = (NUM_ELEMS + get_global_size(0) - 1) / get_global_size(0); \
size_t i_start = get_global_id(0) * i_blocking; \
size_t i_end = (get_global_id(0) + 1) * i_blocking; \
for(size_t i = i_start; i < i_end && i < NUM_ELEMS; ++i)
#else
#define PARALLEL_FOR(i) \
for(size_t i = get_global_id(0); i < NUM_ELEMS; i += get_global_size(0))
#endif

#ifndef ENABLE_STRUCT
/*
 * Scalar kernels
 */
__kernel void copyScalar(__global WRITEABLE(SCALAR, out), __global READONLY(SCALAR, in))
{
	PARALLEL_FOR(i) {
		out[OFFSET + i] = in[OFFSET + i];
	}
}

#else /* ENABLE_STRUCT */

/*
 * Struct kernels
 */

#ifndef N_SOA_BUFFERS

__kernel void copySOA(__global WRITEABLE(SCALAR, out), __global READONLY(SCALAR, in))
{
	PARALLEL_FOR(i) {
		Struct_t tmp = peekStruct(in, i);
		pokeStruct(out, i, tmp);
	}
}

#endif /* N_SOA_BUFFERS */

#endif /* ENABLE_STRUCT */
