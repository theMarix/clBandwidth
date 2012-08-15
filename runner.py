#!/usr/bin/env python
# coding=utf8

# This file is part of clBandwidth.
#
# clBandwidth is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# clBandwidth is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with clBandwidth.  If not, see <http://www.gnu.org/licenses/>.
#
# (c) 2011-2012 Matthias Bach <bach@compeng.uni-frankfurt.de>

import pyopencl as cl
import numpy as np
from collections import namedtuple
import os.path as path

from datatypes import *
from data import *

DEFAULT_MEM_SIZE = 10 * 1024 * 1024 # 10 MiB
LOCAL_THREADS = 128

class Runner:

	def __init__(self, device = None, local_threads = None, global_threads = None, default_mem_size = DEFAULT_MEM_SIZE, alternate_buffers = True, optimizer = None):
		if device != None:
			platforms = cl.get_platforms()
			if len(platforms) > 1:
				raise Exception('Found more then one platform, giving up.')
			platform = platforms[0]
			properties = [(cl.context_properties.PLATFORM, platform)]
			devices = [platform.get_devices()[device]]
			self.ctx = cl.Context(devices, properties)
		else:
			self.ctx = cl.create_some_context()
		self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

		self.device = self.queue.device
		print '#Device: {0}'.format(self.device.name)
		print '#Memory size: {0} KiB'.format(self.device.global_mem_size / 1024)
		print '#Maximum buffer size: {0} KiB'.format(self.device.max_mem_alloc_size / 1024)

		if local_threads == None:
			if self.device.type == cl.device_type.CPU:
				local_threads = 1
			else:
				local_threads = LOCAL_THREADS
		if global_threads == None:
			if self.device.type == cl.device_type.CPU:
				global_threads = self.device.max_compute_units * local_threads
			else:
				global_threads = self.device.max_compute_units * local_threads * 8

		self.local_threads = local_threads
		self.global_threads = global_threads
		self.default_mem_size = default_mem_size

		self.alternate_buffers = alternate_buffers

		self.optimizer = optimizer

	def hasDoublePrecisionSupport(self):
		extensions = self.device.extensions
		return 'cl_khr_fp64' in extensions or 'cl_amd_fp64' in extensions

	def createKernel(self, datatype, num_elems, plain_pointers = False, SOA_stride = 0, offset = 0):
		generated_source = '''
#ifdef PLAIN_POINTERS
#define WRITEABLE(type, p) type * p
#define READONLY(type, p) type * p
#else
#define WRITEABLE(type, p) type * const restrict p
#define READONLY(type, p) const type * const restrict p
#endif
'''

		# check if double support ist required
		if datatype.name.startswith('double') or (isinstance(datatype, Struct) and datatype.scalar.name.startswith('double')):
			generated_source += '#define ENABLE_DOUBLE\n'

			generated_source += '''
/*
 * Enable double if requested.
 * There might theoretically be implementation specific issues caused by an enabled
 * double precision support, making it optional allows to easily check.
 */
#ifdef ENABLE_DOUBLE
#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define DOUBLE_ENABLED
#else /* cl_khr_fp64 */
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#define DOUBLE_ENABLED
#endif /* cl_amd_fp64 */
#endif /* cl_khr_fp64 */
#endif
'''

		generated_source += '#define OFFSET {0}\n'.format(offset)

		if isinstance(datatype, Struct):
			scalar_name = datatype.scalar.name
			# TODO allign type
			generated_source += '''
			typedef struct Struct_s {
			'''
			for i in range(datatype.elems):
				generated_source += '{0} e{1};\n'.format(scalar_name, i)
			generated_source += '} Struct_t;\n'

			if SOA_stride:
				generated_source += '#define SCALAR {0}\n'.format(scalar_name)
				generated_source += '#define ENABLE_STRUCT\n'
				generated_source += '#define SOA_STRIDE {0}\n'.format(SOA_stride)
				generated_source += 'Struct_t peekStruct(__global READONLY(SCALAR, in), const size_t idx);\n'
				generated_source += 'void pokeStruct(__global WRITEABLE(SCALAR, out), const size_t idx, const Struct_t val);\n'
				generated_source += 'Struct_t peekStruct(__global READONLY(SCALAR, in), const size_t idx) { return (Struct_t) {'
				for i in range(datatype.elems - 1):
					generated_source += 'in[idx + OFFSET + SOA_STRIDE * {0}], '.format(i)
				generated_source += 'in[idx + OFFSET + SOA_STRIDE * {0}]'.format(datatype.elems - 1)
				generated_source += '}; };\n'
				generated_source += 'void pokeStruct(__global WRITEABLE(SCALAR, out), const size_t idx, const Struct_t val) {'
				for i in range(datatype.elems):
					generated_source += 'out[idx + OFFSET + SOA_STRIDE * {0}] = val.e{0};'.format(i)
				generated_source += '};\n'
			else:
				scalar_name = 'Struct_t'
				generated_source += '#define SCALAR {0}\n'.format(scalar_name)
		else:
			scalar_name = datatype.name
			generated_source += '#define SCALAR {0}\n'.format(scalar_name)

		# not the strange way to send definitions to the kernel
		# this is due to some weired error on OSX
		generated_source += '''
		#define NUM_ELEMS {0}
		'''.format(num_elems);
		if plain_pointers:
			generated_source += '#define PLAIN_POINTERS\n';

		# TODO make choosable
		if self.device.type == cl.device_type.CPU:
			generated_source += '#define BLOCKED_LOOP\n'


		base_folder = path.dirname(__file__)
		f = open(path.join(base_folder, 'kernels.cl'), 'r')
		fstr = generated_source + "".join(f.readlines())
		prg = cl.Program(self.ctx, fstr).build()
		if SOA_stride:
			return prg.copySOA
		else:
			return prg.copyScalar;

	def benchmark(self, datatype, mem_size = None, global_threads = None, local_threads = None, stride = 0, offset = 0):
		BENCH_RUNS = 10
		WARMUP_RUNS = 2

		if not global_threads:
			global_threads = self.global_threads
		if not local_threads:
			local_threads = self.local_threads
		if not mem_size:
			mem_size = self.default_mem_size

		elems = mem_size / datatype.size;
		bytes_transferred = elems * datatype.size * 2

		if stride < 0:
			stride = self._guessStride(datatype, elems)
		if stride:
			required_buf_size = (offset + stride) * datatype.size
		else:
			required_buf_size = mem_size + datatype.size * offset

		if self.alternate_buffers:
			in_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, required_buf_size)
			out_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, required_buf_size)
		else:
			in_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY, required_buf_size)
			out_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, required_buf_size)

		kernel = self.createKernel(datatype, elems, SOA_stride = stride, offset = offset)

		events = []
		for i in range(BENCH_RUNS + WARMUP_RUNS):
			# Kernel launching logic
			# This was supposed to be in an own function, but when you return the events
			# you cannot call the function more than once for some unkown reason
			event = kernel(self.queue, (global_threads,), (local_threads,), out_buf, in_buf)
			events.append(event)
			if self.alternate_buffers:
				tmp = out_buf
				out_buf = in_buf
				in_buf = tmp

		cl.wait_for_events(events)

		# clean up memory
		in_buf.release()
		out_buf.release()

		# throw away warmup runs
		events = events[WARMUP_RUNS:]
		event_times = map(lambda event: (event.profile.end - event.profile.start), events)
		elapsed = np.mean(event_times)
		elapsed_std = np.std(event_times)

		if isinstance(datatype, Struct):
			stride_bytes = stride * datatype.scalar.size
			offset_bytes = offset * datatype.scalar.size
		else:
			stride_bytes = stride * datatype.size
			offset_bytes = offset * datatype.size

		return DataPoint(datatype.name, global_threads, local_threads, stride, stride_bytes, offset, offset_bytes, elems * datatype.size, elems * datatype.size, bytes_transferred, elapsed, elapsed_std, bytes_transferred / elapsed)

	def _guessStride(self, datatype, elems):
		if self.optimizer:
			return self.optimizer.recommendStride(datatype, elems)
		else:
			return elems
