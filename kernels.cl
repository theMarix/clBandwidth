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
