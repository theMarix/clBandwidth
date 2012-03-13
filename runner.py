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

from datatypes import *

MAX_MEM_SIZE = 10 * 1024 * 1024 # 10 MiB
LOCAL_THREADS = 128

# Result of kernel invocation for a given set of parameters. time is in nanos, bandwidth in GB/s
DataPoint = namedtuple('DataPoint', 'kernel global_threads local_threads bytes_transferred time time_std bandwidth')

class Runner:

	def __init__(self, device = None, local_threads = None, global_threads = None, max_mem_size = MAX_MEM_SIZE, alternate_buffers = True):
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
		self.max_mem_size = max_mem_size

		self.alternate_buffers = alternate_buffers
		if self.alternate_buffers:
			self.in_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, max_mem_size)
			self.out_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, max_mem_size)
		else:
			self.in_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY, max_mem_size)
			self.out_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, max_mem_size)

	def hasDoublePrecisionSupport(self):
		extensions = self.device.extensions
		return 'cl_khr_fp64' in extensions or 'cl_amd_fp64' in extensions

	def createKernel(self, datatype, num_elems, plain_pointers = False, SOA_stride = 0):
		generated_source = '''
#ifdef PLAIN_POINTERS
#define WRITEABLE(type, p) type * p
#define READONLY(type, p) type * p
#else
#define WRITEABLE(type, p) type * const restrict p
#define READONLY(type, p) const type * const restrict p
#endif
'''

		if isinstance(datatype, Struct):
			scalar_name = datatype.scalar.name
			# TODO allign type
			generated_source += '''
			typedef struct Struct_s {
			'''
			for i in range(datatype.elems - 1):
				generated_source += '{0} e{1};\n'.format(scalar_name, i)
				i += 1
			generated_source += '{0} e{1}'.format(scalar_name, datatype.elems - 1)
			generated_source += '} Struct_t;\n'

			if SOA_stride:
				generated_source += '#define SCALAR {0}\n'.format(scalar_name)
				if SOA_stride < 0: # guess a stride
					SOA_stride = num_elems
				generated_source += '#define ENABLE_STRUCT\n'
				generated_source += '#define SOA_STRIDE {0}\n'.format(SOA_stride)
				generated_source += 'Struct_t peekStruct(__global READONLY(SCALAR, in), const size_t idx);\n'
				generated_source += 'void pokeStruct(__global WRITEABLE(SCALAR, out), const size_t idx, const Struct_t val);\n'
				generated_source += 'Struct_t peekStruct(__global READONLY(SCALAR, in), const size_t idx) { return (Struct_t) {'
				for i in range(datatype.elems - 1):
					generated_source += 'in[idx + SOA_STRIDE * {0}], '.format(i)
				generated_source += 'in[idx + SOA_STRIDE * {0}]'.format(datatype.elems - 1)
				generated_source += '}; };\n'
				generated_source += 'void pokeStruct(__global WRITEABLE(SCALAR, out), const size_t idx, const Struct_t val) {'
				for i in range(datatype.elems):
					generated_source += 'out[idx + SOA_STRIDE * {0}] = val.e{0};'.format(i)
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

		f = open('kernels.cl', 'r')
		fstr = generated_source + "".join(f.readlines())
		prg = cl.Program(self.ctx, fstr).build()
		if SOA_stride:
			return prg.copySOA
		else:
			return prg.copyScalar;

	def benchmark(self, datatype, mem_size = None, global_threads = None, local_threads = None, stride = None):
		BENCH_RUNS = 10
		WARMUP_RUNS = 2

		if not global_threads:
			global_threads = self.global_threads
		if not local_threads:
			local_threads = self.local_threads
		if not mem_size:
			mem_size = self.max_mem_size
		if not stride:
			stride = 0

		elems = mem_size / datatype.size;
		bytes_transferred = elems * datatype.size * 2

		kernel = self.createKernel(datatype, elems, SOA_stride = stride)

		events = []
		for i in range(BENCH_RUNS + WARMUP_RUNS):
			# Kernel launching logic
			# This was supposed to be in an own function, but when you return the events
			# you cannot call the function more than once for some unkown reason
			event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf)
			events.append(event)
			if self.alternate_buffers:
				tmp = self.out_buf
				self.out_buf = self.in_buf
				self.in_buf = tmp

		cl.wait_for_events(events)

		# throw away warmup runs
		events = events[WARMUP_RUNS:]
		event_times = map(lambda event: (event.profile.end - event.profile.start), events)
		elapsed = np.mean(event_times)
		elapsed_std = np.std(event_times)

		return DataPoint(datatype.name, global_threads, local_threads, bytes_transferred, elapsed, elapsed_std, bytes_transferred / elapsed)
