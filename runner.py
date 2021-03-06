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
# (c) 2011 Matthias Bach <bach@compeng.uni-frankfurt.de>

import pyopencl as cl
import numpy as np
from collections import namedtuple

MAX_MEM_SIZE = 10 * 1024 * 1024 # 10 MiB
LOCAL_THREADS = 128
GLOBAL_THREADS = 20 * 8 * LOCAL_THREADS

# Result of kernel invocation for a given set of parameters. time is in nanos, bandwidth in GB/s
DataPoint = namedtuple('DataPoint', 'kernel global_threads local_threads bytes_transferred time time_std bandwidth')

class Runner:

	def __init__(self, device = None, local_threads = LOCAL_THREADS, global_threads = GLOBAL_THREADS, max_mem_size = MAX_MEM_SIZE):
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

		f = open('kernels.cl', 'r')
		fstr = "".join(f.readlines())
		self.prg = cl.Program(self.ctx, fstr).build()

		self.in_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY, max_mem_size)
		self.out_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, max_mem_size)

		self.local_threads = local_threads
		self.global_threads = global_threads
		self.max_mem_size = max_mem_size

	def hasDoublePrecisionSupport(self):
		extensions = self.device.extensions
		return 'cl_khr_fp64' in extensions or 'cl_amd_fp64' in extensions

	def get_kernel_names(self):
		# all_kernels() is broken in pyopencl 2011.* :(
		# return map(lambda kernel: kernel.info.function_name, self.prg.all_kernels())
		kernels = [
		 'copyFloat',
		 'readFloat',
		 'writeFloat',
		 'copyFloatRestricted',
		 'readFloatRestricted',
		 'writeFloatRestricted',
		 'copyFloat2',
		 'readFloat2',
		 'writeFloat2',
		 'copyFloat4',
		 'readFloat4',
		 'writeFloat4',
		 'copySpComplex',
		 'readSpComplex',
		 'writeSpComplex',
		 'copySpComplexRestricted',
		 'readSpComplexRestricted',
		 'writeSpComplexRestricted',
		 'copyAlignedSpComplex',
		 'readAlignedSpComplex',
		 'writeAlignedSpComplex',
		 'copyAlignedSpComplexRestricted',
		 'readAlignedSpComplexRestricted',
		 'writeAlignedSpComplexRestricted',
		 'copySpSu3vec',
		 'readSpSu3vec',
		 'writeSpSu3vec',
		 'copySpSu3vecRestricted',
		 'readSpSu3vecRestricted',
		 'writeSpSu3vecRestricted',
		 'copyAlignedSpSu3vecRestricted',
		 'readAlignedSpSu3vecRestricted',
		 'writeAlignedSpSu3vecRestricted',
		 'copyAligned8SpSu3vecRestricted',
		 'readAligned8SpSu3vecRestricted',
		 'writeAligned8SpSu3vecRestricted',
		 'copyAligned16SpSu3vecRestricted',
		 'readAligned16SpSu3vecRestricted',
		 'writeAligned16SpSu3vecRestricted',
		 'copyAligned32SpSu3vecRestricted',
		 'readAligned32SpSu3vecRestricted',
		 'writeAligned32SpSu3vecRestricted',
		 'copySpSu3vecFromAlignedRestricted',
		 'readSpSu3vecFromAlignedRestricted',
		 'writeSpSu3vecFromAlignedRestricted',
		 'copySpSu3',
		 'readSpSu3',
		 'writeSpSu3',
		 'copySpSu3Restricted',
		 'readSpSu3Restricted',
		 'writeSpSu3Restricted',
		 'copySpSu3FromAlignedRestricted',
		 'readSpSu3FromAlignedRestricted',
		 'writeSpSu3FromAlignedRestricted',
		 'copyAligned8SpSu3Restricted',
		 'readAligned8SpSu3Restricted',
		 'writeAligned8SpSu3Restricted',
		 'copySpSu3SOARestricted',
		 'readSpSu3SOARestricted',
		 'writeSpSu3SOARestricted',
		 'copySpSu3FromAlignedSOARestricted',
		 'readSpSu3FromAlignedSOARestricted',
		 'writeSpSu3FromAlignedSOARestricted',
		 'copySpSu3ViaLocalRestricted',
		 'readSpSu3ViaLocalRestricted',
		 'writeSpSu3ViaLocalRestricted',
		 'copySpSu3FromAlignedViaLocalRestricted',
		 'readSpSu3FromAlignedViaLocalRestricted',
		 'writeSpSu3FromAlignedViaLocalRestricted',
		 'copySpSpinor',
		 'readSpSpinor',
		 'writeSpSpinor',
		 'copySpSpinorRestricted',
		 'readSpSpinorRestricted',
		 'writeSpSpinorRestricted',
		 'copySpSpinorFromAlignedRestricted',
		 'readSpSpinorFromAlignedRestricted',
		 'writeSpSpinorFromAlignedRestricted',
		 'copyAligned8SpSpinorRestricted',
		 'readAligned8SpSpinorRestricted',
		 'writeAligned8SpSpinorRestricted',
		 'copyAligned16SpSpinorRestricted',
		 'readAligned16SpSpinorRestricted',
		 'writeAligned16SpSpinorRestricted',
		 'copyAligned32SpSpinorRestricted',
		 'readAligned32SpSpinorRestricted',
		 'writeAligned32SpSpinorRestricted',
		 'copySpSpinorSOARestricted',
		 'readSpSpinorSOARestricted',
		 'writeSpSpinorSOARestricted',
		 'copySpSpinorFullSOARestricted',
		 'readSpSpinorFullSOARestricted',
		 'writeSpSpinorFullSOARestricted',
		 'copySpSpinorFullAlignedSOARestricted',
		 'readSpSpinorFullAlignedSOARestricted',
		 'writeSpSpinorFullAlignedSOARestricted',
		 'copySpSpinorFromAlignedSOARestricted',
		 'readSpSpinorFromAlignedSOARestricted',
		 'writeSpSpinorFromAlignedSOARestricted',
		 'copySpSpinorViaLocalRestricted',
		 'readSpSpinorViaLocalRestricted',
		 'writeSpSpinorViaLocalRestricted',
		]

		if self.hasDoublePrecisionSupport():
			kernels.extend([
			 'copyDouble',
			 'readDouble',
			 'writeDouble',
			 'copyDoubleRestricted',
			 'readDoubleRestricted',
			 'writeDoubleRestricted',
			 'copyDouble2',
			 'readDouble2',
			 'writeDouble2',
			 'copyDouble4',
			 'readDouble4',
			 'writeDouble4',
			 'copyDpComplex',
			 'readDpComplex',
			 'writeDpComplex',
			 'copyDpComplexRestricted',
			 'readDpComplexRestricted',
			 'writeDpComplexRestricted',
			 'copyAlignedDpComplex',
			 'readAlignedDpComplex',
			 'writeAlignedDpComplex',
			 'copyAlignedDpComplexRestricted',
			 'readAlignedDpComplexRestricted',
			 'writeAlignedDpComplexRestricted',
			 'copyDpSu3vec',
			 'readDpSu3vec',
			 'writeDpSu3vec',
			 'copyDpSu3vecRestricted',
			 'readDpSu3vecRestricted',
			 'writeDpSu3vecRestricted',
			 'copyAligned16DpSu3vecRestricted',
			 'readAligned16DpSu3vecRestricted',
			 'writeAligned16DpSu3vecRestricted',
			 'copyAligned32DpSu3vecRestricted',
			 'readAligned32DpSu3vecRestricted',
			 'writeAligned32DpSu3vecRestricted',
			 'copyDpSu3vecSOARestricted',
			 'readDpSu3vecSOARestricted',
			 'writeDpSu3vecSOARestricted',
			 'copyDpSu3vecFullSOARestricted',
			 'readDpSu3vecFullSOARestricted',
			 'writeDpSu3vecFullSOARestricted',
			 'copyDpSu3',
			 'readDpSu3',
			 'writeDpSu3',
			 'copyDpSu3Restricted',
			 'readDpSu3Restricted',
			 'writeDpSu3Restricted',
			 'copyAligned16DpSu3Restricted',
			 'readAligned16DpSu3Restricted',
			 'writeAligned16DpSu3Restricted',
			 'copyAligned32DpSu3Restricted',
			 'readAligned32DpSu3Restricted',
			 'writeAligned32DpSu3Restricted',
			 'copyDpSu3SOARestricted',
			 'readDpSu3SOARestricted',
			 'writeDpSu3SOARestricted',
			 'copyDpSu3FullSOARestricted',
			 'readDpSu3FullSOARestricted',
			 'writeDpSu3FullSOARestricted',
			 'copyDpSpinor',
			 'readDpSpinor',
			 'writeDpSpinor',
			 'copyDpSpinorRestricted',
			 'readDpSpinorRestricted',
			 'writeDpSpinorRestricted',
			 'copyAligned16DpSpinorRestricted',
			 'readAligned16DpSpinorRestricted',
			 'writeAligned16DpSpinorRestricted',
			 'copyAligned32DpSpinorRestricted',
			 'readAligned32DpSpinorRestricted',
			 'writeAligned32DpSpinorRestricted',
			 'copyDpSpinorSOARestricted',
			 'readDpSpinorSOARestricted',
			 'writeDpSpinorSOARestricted',
			 'copyDpSpinorFullSOARestricted',
			 'readDpSpinorFullSOARestricted',
			 'writeDpSpinorFullSOARestricted',
			 'copyDpSpinorFullestSOARestricted',
			 'readDpSpinorFullestSOARestricted',
			 'writeDpSpinorFullestSOARestricted',
			])

		return kernels

	def benchmark(self, kernelname, mem_size = None, global_threads = None, local_threads = None, stride = None):
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

		events = []
		for i in range(BENCH_RUNS + WARMUP_RUNS):

			# Kernel launching logic
			# This was supposed to be in an own function, but when you return the events
			# you cannot call the function more than once for some unkown reason
			if kernelname == 'copyFloat':
				kernel = self.prg.copyFloat;
				elems = mem_size / 4
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = elems * 4 * 2
			elif kernelname == 'readFloat':
				kernel = self.prg.readFloat;
				elems = mem_size / 4
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = (elems + global_threads) * 4
			elif kernelname == 'writeFloat':
				kernel = self.prg.writeFloat;
				elems = mem_size / 4
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float32(1.), np.uint64(elems))
				bytes_transferred = elems * 4
			elif kernelname == 'copyFloatRestricted':
				kernel = self.prg.copyFloatRestricted;
				elems = mem_size / 4
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = elems * 4 * 2
			elif kernelname == 'readFloatRestricted':
				kernel = self.prg.readFloatRestricted;
				elems = mem_size / 4
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = (elems + global_threads) * 4
			elif kernelname == 'writeFloatRestricted':
				kernel = self.prg.writeFloatRestricted;
				elems = mem_size / 4
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float32(1.), np.uint64(elems))
				bytes_transferred = elems * 4
			elif kernelname == 'copyFloat2':
				kernel = self.prg.copyFloat2;
				elems = mem_size / 8
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = elems * 8 * 2
			elif kernelname == 'readFloat2':
				kernel = self.prg.readFloat2;
				elems = mem_size / 8
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = (elems + global_threads) * 8
			elif kernelname == 'writeFloat2':
				kernel = self.prg.writeFloat2;
				elems = mem_size / 8
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float32(1.), np.uint64(elems))
				bytes_transferred = elems * 8
			elif kernelname == 'copyFloat4':
				kernel = self.prg.copyFloat4;
				elems = mem_size / 16
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = elems * 16 * 2
			elif kernelname == 'readFloat4':
				kernel = self.prg.readFloat4;
				elems = mem_size / 16
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = (elems + global_threads) * 16
			elif kernelname == 'writeFloat4':
				kernel = self.prg.writeFloat4;
				elems = mem_size / 16
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float32(1.), np.uint64(elems))
				bytes_transferred = elems * 16

			elif kernelname == 'copySpComplex':
				kernel = self.prg.copySpComplex;
				elems = mem_size / 8
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = elems * 8 * 2
			elif kernelname == 'readSpComplex':
				kernel = self.prg.readSpComplex;
				elems = mem_size / 8
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = (elems + global_threads) * 8
			elif kernelname == 'writeSpComplex':
				kernel = self.prg.writeSpComplex;
				elems = mem_size / 8
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float32(1.), np.uint64(elems))
				bytes_transferred = elems * 8
			elif kernelname == 'copySpComplexRestricted':
				kernel = self.prg.copySpComplexRestricted;
				elems = mem_size / 8
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = elems * 8 * 2
			elif kernelname == 'readSpComplexRestricted':
				kernel = self.prg.readSpComplexRestricted;
				elems = mem_size / 8
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = (elems + global_threads) * 8
			elif kernelname == 'writeSpComplexRestricted':
				kernel = self.prg.writeSpComplexRestricted;
				elems = mem_size / 8
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float32(1.), np.uint64(elems))
				bytes_transferred = elems * 8

			elif kernelname == 'copyAlignedSpComplex':
				kernel = self.prg.copyAlignedSpComplex;
				elems = mem_size / 8
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = elems * 8 * 2
			elif kernelname == 'readAlignedSpComplex':
				kernel = self.prg.readAlignedSpComplex;
				elems = mem_size / 8
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = (elems + global_threads) * 8
			elif kernelname == 'writeAlignedSpComplex':
				kernel = self.prg.writeAlignedSpComplex;
				elems = mem_size / 8
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float32(1.), np.uint64(elems))
				bytes_transferred = elems * 8
			elif kernelname == 'copyAlignedSpComplexRestricted':
				kernel = self.prg.copyAlignedSpComplexRestricted;
				elems = mem_size / 8
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = elems * 8 * 2
			elif kernelname == 'readAlignedSpComplexRestricted':
				kernel = self.prg.readAlignedSpComplexRestricted;
				elems = mem_size / 8
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = (elems + global_threads) * 8
			elif kernelname == 'writeAlignedSpComplexRestricted':
				kernel = self.prg.writeAlignedSpComplexRestricted;
				elems = mem_size / 8
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float32(1.), np.uint64(elems))
				bytes_transferred = elems * 8

			elif kernelname == 'copySpSu3vec':
				kernel = self.prg.copySpSu3vec;
				elems = mem_size / 24
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = elems * 24 * 2
			elif kernelname == 'readSpSu3vec':
				kernel = self.prg.readSpSu3vec;
				elems = mem_size / 24
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = (elems + global_threads) * 24
			elif kernelname == 'writeSpSu3vec':
				kernel = self.prg.writeSpSu3vec;
				elems = mem_size / 24
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float32(1.), np.uint64(elems))
				bytes_transferred = elems * 24
			elif kernelname == 'copySpSu3vecRestricted':
				kernel = self.prg.copySpSu3vecRestricted;
				elems = mem_size / 24
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = elems * 24 * 2
			elif kernelname == 'readSpSu3vecRestricted':
				kernel = self.prg.readSpSu3vecRestricted;
				elems = mem_size / 24
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = (elems + global_threads) * 24
			elif kernelname == 'writeSpSu3vecRestricted':
				kernel = self.prg.writeSpSu3vecRestricted;
				elems = mem_size / 24
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float32(1.), np.uint64(elems))
				bytes_transferred = elems * 24

			elif kernelname == 'copySpSu3vecFromAlignedRestricted':
				kernel = self.prg.copySpSu3vecFromAlignedRestricted;
				elems = mem_size / 24
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = elems * 24 * 2
			elif kernelname == 'readSpSu3vecFromAlignedRestricted':
				kernel = self.prg.readSpSu3vecFromAlignedRestricted;
				elems = mem_size / 24
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = (elems + global_threads) * 24
			elif kernelname == 'writeSpSu3vecFromAlignedRestricted':
				kernel = self.prg.writeSpSu3vecFromAlignedRestricted;
				elems = mem_size / 24
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float32(1.), np.uint64(elems))
				bytes_transferred = elems * 24

			elif kernelname == 'copyAlignedSpSu3vecRestricted':
				kernel = self.prg.copyAlignedSpSu3vecRestricted;
				elems = mem_size / 32
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = elems * 24 * 2
			elif kernelname == 'readAlignedSpSu3vecRestricted':
				kernel = self.prg.readAlignedSpSu3vecRestricted;
				elems = mem_size / 32
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = (elems + global_threads) * 24
			elif kernelname == 'writeAlignedSpSu3vecRestricted':
				kernel = self.prg.writeAlignedSpSu3vecRestricted;
				elems = mem_size / 32
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float32(1.), np.uint64(elems))
				bytes_transferred = elems * 24
			elif kernelname == 'copyAligned8SpSu3vecRestricted':
				kernel = self.prg.copyAligned8SpSu3vecRestricted;
				elems = mem_size / 24
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = elems * 24 * 2
			elif kernelname == 'readAligned8SpSu3vecRestricted':
				kernel = self.prg.readAligned8SpSu3vecRestricted;
				elems = mem_size / 24
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = (elems + global_threads) * 24
			elif kernelname == 'writeAligned8SpSu3vecRestricted':
				kernel = self.prg.writeAligned8SpSu3vecRestricted;
				elems = mem_size / 24
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float32(1.), np.uint64(elems))
				bytes_transferred = elems * 24
			elif kernelname == 'copyAligned16SpSu3vecRestricted':
				kernel = self.prg.copyAligned16SpSu3vecRestricted;
				elems = mem_size / 32
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = elems * 24 * 2
			elif kernelname == 'readAligned16SpSu3vecRestricted':
				kernel = self.prg.readAligned16SpSu3vecRestricted;
				elems = mem_size / 32
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = (elems + global_threads) * 24
			elif kernelname == 'writeAligned16SpSu3vecRestricted':
				kernel = self.prg.writeAligned16SpSu3vecRestricted;
				elems = mem_size / 32
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float32(1.), np.uint64(elems))
				bytes_transferred = elems * 24
			elif kernelname == 'copyAligned32SpSu3vecRestricted':
				kernel = self.prg.copyAligned32SpSu3vecRestricted;
				elems = mem_size / 32
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = elems * 24 * 2
			elif kernelname == 'readAligned32SpSu3vecRestricted':
				kernel = self.prg.readAligned32SpSu3vecRestricted;
				elems = mem_size / 32
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = (elems + global_threads) * 24
			elif kernelname == 'writeAligned32SpSu3vecRestricted':
				kernel = self.prg.writeAligned32SpSu3vecRestricted;
				elems = mem_size / 32
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float32(1.), np.uint64(elems))
				bytes_transferred = elems * 24

			elif kernelname == 'copySpSu3':
				kernel = self.prg.copySpSu3;
				elems = mem_size / 72
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = elems * 72 * 2
			elif kernelname == 'readSpSu3':
				kernel = self.prg.readSpSu3;
				elems = mem_size / 72
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = (elems + global_threads) * 72
			elif kernelname == 'writeSpSu3':
				kernel = self.prg.writeSpSu3;
				elems = mem_size / 72
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float32(1.), np.uint64(elems))
				bytes_transferred = elems * 72
			elif kernelname == 'copySpSu3Restricted':
				kernel = self.prg.copySpSu3Restricted;
				elems = mem_size / 72
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = elems * 72 * 2
			elif kernelname == 'readSpSu3Restricted':
				kernel = self.prg.readSpSu3Restricted;
				elems = mem_size / 72
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = (elems + global_threads) * 72
			elif kernelname == 'writeSpSu3Restricted':
				kernel = self.prg.writeSpSu3Restricted;
				elems = mem_size / 72
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float32(1.), np.uint64(elems))
				bytes_transferred = elems * 72
			elif kernelname == 'copySpSu3SOARestricted':
				kernel = self.prg.copySpSu3SOARestricted;
				elems = mem_size / 72
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems), np.uint64(stride))
				bytes_transferred = elems * 72 * 2
			elif kernelname == 'readSpSu3SOARestricted':
				kernel = self.prg.readSpSu3SOARestricted;
				elems = mem_size / 72
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems), np.uint64(stride))
				bytes_transferred = (elems + global_threads) * 72
			elif kernelname == 'writeSpSu3SOARestricted':
				kernel = self.prg.writeSpSu3SOARestricted;
				elems = mem_size / 72
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float32(1.), np.uint64(elems), np.uint64(stride))
				bytes_transferred = elems * 72
			elif kernelname == 'copySpSu3ViaLocalRestricted':
				kernel = self.prg.copySpSu3ViaLocalRestricted;
				elems = mem_size / 72
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems), cl.LocalMemory(local_threads * 72))
				bytes_transferred = elems * 72 * 2
			elif kernelname == 'readSpSu3ViaLocalRestricted':
				kernel = self.prg.readSpSu3ViaLocalRestricted;
				elems = mem_size / 72
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems), cl.LocalMemory(local_threads * 72))
				bytes_transferred = (elems + global_threads) * 72
			elif kernelname == 'writeSpSu3ViaLocalRestricted':
				kernel = self.prg.writeSpSu3ViaLocalRestricted;
				elems = mem_size / 72
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float32(1.), np.uint64(elems), cl.LocalMemory(local_threads * 72))
				bytes_transferred = elems * 72
			elif kernelname == 'copySpSu3FromAlignedViaLocalRestricted':
				kernel = self.prg.copySpSu3FromAlignedViaLocalRestricted;
				elems = mem_size / 72
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems), cl.LocalMemory(local_threads * 72))
				bytes_transferred = elems * 72 * 2
			elif kernelname == 'readSpSu3FromAlignedViaLocalRestricted':
				kernel = self.prg.readSpSu3FromAlignedViaLocalRestricted;
				elems = mem_size / 72
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems), cl.LocalMemory(local_threads * 72))
				bytes_transferred = (elems + global_threads) * 72
			elif kernelname == 'writeSpSu3FromAlignedViaLocalRestricted':
				kernel = self.prg.writeSpSu3FromAlignedViaLocalRestricted;
				elems = mem_size / 72
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float32(1.), np.uint64(elems), cl.LocalMemory(local_threads * 72))
				bytes_transferred = elems * 72

			elif kernelname == 'copySpSu3FromAlignedRestricted':
				kernel = self.prg.copySpSu3FromAlignedRestricted;
				elems = mem_size / 72
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = elems * 72 * 2
			elif kernelname == 'readSpSu3FromAlignedRestricted':
				kernel = self.prg.readSpSu3FromAlignedRestricted;
				elems = mem_size / 72
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = (elems + global_threads) * 72
			elif kernelname == 'writeSpSu3FromAlignedRestricted':
				kernel = self.prg.writeSpSu3FromAlignedRestricted;
				elems = mem_size / 72
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float32(1.), np.uint64(elems))
				bytes_transferred = elems * 72
			elif kernelname == 'copySpSu3FromAlignedSOARestricted':
				kernel = self.prg.copySpSu3FromAlignedSOARestricted;
				elems = mem_size / 72
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems), np.uint64(stride))
				bytes_transferred = elems * 72 * 2
			elif kernelname == 'readSpSu3FromAlignedSOARestricted':
				kernel = self.prg.readSpSu3FromAlignedSOARestricted;
				elems = mem_size / 72
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems), np.uint64(stride))
				bytes_transferred = (elems + global_threads) * 72
			elif kernelname == 'writeSpSu3FromAlignedSOARestricted':
				kernel = self.prg.writeSpSu3FromAlignedSOARestricted;
				elems = mem_size / 72
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float32(1.), np.uint64(elems), np.uint64(stride))
				bytes_transferred = elems * 72
			elif kernelname == 'copyAligned8SpSu3Restricted':
				kernel = self.prg.copyAligned8SpSu3Restricted;
				elems = mem_size / 72
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = elems * 72 * 2
			elif kernelname == 'readAligned8SpSu3Restricted':
				kernel = self.prg.readAligned8SpSu3Restricted;
				elems = mem_size / 72
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = (elems + global_threads) * 72
			elif kernelname == 'writeAligned8SpSu3Restricted':
				kernel = self.prg.writeAligned8SpSu3Restricted;
				elems = mem_size / 72
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float32(1.), np.uint64(elems))
				bytes_transferred = elems * 72

			elif kernelname == 'copySpSpinor':
				kernel = self.prg.copySpSpinor;
				elems = mem_size / 96
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = elems * 96 * 2
			elif kernelname == 'readSpSpinor':
				kernel = self.prg.readSpSpinor;
				elems = mem_size / 96
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = (elems + global_threads) * 96
			elif kernelname == 'writeSpSpinor':
				kernel = self.prg.writeSpSpinor;
				elems = mem_size / 96
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float32(1.), np.uint64(elems))
				bytes_transferred = elems * 96
			elif kernelname == 'copySpSpinorRestricted':
				kernel = self.prg.copySpSpinorRestricted;
				elems = mem_size / 96
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = elems * 96 * 2
			elif kernelname == 'readSpSpinorRestricted':
				kernel = self.prg.readSpSpinorRestricted;
				elems = mem_size / 96
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = (elems + global_threads) * 96
			elif kernelname == 'writeSpSpinorRestricted':
				kernel = self.prg.writeSpSpinorRestricted;
				elems = mem_size / 96
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float32(1.), np.uint64(elems))
				bytes_transferred = elems * 96
			elif kernelname == 'copySpSpinorSOARestricted':
				kernel = self.prg.copySpSpinorSOARestricted;
				elems = mem_size / 96
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems), np.uint64(stride))
				bytes_transferred = elems * 96 * 2
			elif kernelname == 'readSpSpinorSOARestricted':
				kernel = self.prg.readSpSpinorSOARestricted;
				elems = mem_size / 96
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems), np.uint64(stride))
				bytes_transferred = (elems + global_threads) * 96
			elif kernelname == 'writeSpSpinorSOARestricted':
				kernel = self.prg.writeSpSpinorSOARestricted;
				elems = mem_size / 96
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float32(1.), np.uint64(elems), np.uint64(stride))
				bytes_transferred = elems * 96
			elif kernelname == 'copySpSpinorFullSOARestricted':
				kernel = self.prg.copySpSpinorFullSOARestricted;
				elems = mem_size / 96
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems), np.uint64(stride))
				bytes_transferred = elems * 96 * 2
			elif kernelname == 'readSpSpinorFullSOARestricted':
				kernel = self.prg.readSpSpinorFullSOARestricted;
				elems = mem_size / 96
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems), np.uint64(stride))
				bytes_transferred = (elems + global_threads) * 96
			elif kernelname == 'writeSpSpinorFullSOARestricted':
				kernel = self.prg.writeSpSpinorFullSOARestricted;
				elems = mem_size / 96
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float32(1.), np.uint64(elems), np.uint64(stride))
				bytes_transferred = elems * 96
			elif kernelname == 'copySpSpinorFullAlignedSOARestricted':
				kernel = self.prg.copySpSpinorFullAlignedSOARestricted;
				elems = mem_size / 96
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems), np.uint64(stride))
				bytes_transferred = elems * 96 * 2
			elif kernelname == 'readSpSpinorFullAlignedSOARestricted':
				kernel = self.prg.readSpSpinorFullAlignedSOARestricted;
				elems = mem_size / 96
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems), np.uint64(stride))
				bytes_transferred = (elems + global_threads) * 96
			elif kernelname == 'writeSpSpinorFullAlignedSOARestricted':
				kernel = self.prg.writeSpSpinorFullAlignedSOARestricted;
				elems = mem_size / 96
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float32(1.), np.uint64(elems), np.uint64(stride))
				bytes_transferred = elems * 96
			elif kernelname == 'copySpSpinorViaLocalRestricted':
				kernel = self.prg.copySpSpinorViaLocalRestricted;
				elems = mem_size / 96
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems), cl.LocalMemory(local_threads * 96))
				bytes_transferred = elems * 96 * 2
			elif kernelname == 'readSpSpinorViaLocalRestricted':
				kernel = self.prg.readSpSpinorViaLocalRestricted;
				elems = mem_size / 96
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems), cl.LocalMemory(local_threads * 96))
				bytes_transferred = (elems + global_threads) * 96
			elif kernelname == 'writeSpSpinorViaLocalRestricted':
				kernel = self.prg.writeSpSpinorViaLocalRestricted;
				elems = mem_size / 96
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float32(1.), np.uint64(elems), cl.LocalMemory(local_threads * 96))
				bytes_transferred = elems * 96
			elif kernelname == 'copySpSpinorFromAlignedRestricted':
				kernel = self.prg.copySpSpinorFromAlignedRestricted;
				elems = mem_size / 96
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = elems * 96 * 2
			elif kernelname == 'readSpSpinorFromAlignedRestricted':
				kernel = self.prg.readSpSpinorFromAlignedRestricted;
				elems = mem_size / 96
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = (elems + global_threads) * 96
			elif kernelname == 'writeSpSpinorFromAlignedRestricted':
				kernel = self.prg.writeSpSpinorFromAlignedRestricted;
				elems = mem_size / 96
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float32(1.), np.uint64(elems))
				bytes_transferred = elems * 96
			elif kernelname == 'copySpSpinorFromAlignedSOARestricted':
				kernel = self.prg.copySpSpinorFromAlignedSOARestricted;
				elems = mem_size / 96
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems), np.uint64(stride))
				bytes_transferred = elems * 96 * 2
			elif kernelname == 'readSpSpinorFromAlignedSOARestricted':
				kernel = self.prg.readSpSpinorFromAlignedSOARestricted;
				elems = mem_size / 96
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems), np.uint64(stride))
				bytes_transferred = (elems + global_threads) * 96
			elif kernelname == 'writeSpSpinorFromAlignedSOARestricted':
				kernel = self.prg.writeSpSpinorFromAlignedSOARestricted;
				elems = mem_size / 96
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float32(1.), np.uint64(elems), np.uint64(stride))
				bytes_transferred = elems * 96
			elif kernelname == 'copyAligned8SpSpinorRestricted':
				kernel = self.prg.copyAligned8SpSpinorRestricted;
				elems = mem_size / 96
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = elems * 96 * 2
			elif kernelname == 'readAligned8SpSpinorRestricted':
				kernel = self.prg.readAligned8SpSpinorRestricted;
				elems = mem_size / 96
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = (elems + global_threads) * 96
			elif kernelname == 'writeAligned8SpSpinorRestricted':
				kernel = self.prg.writeAligned8SpSpinorRestricted;
				elems = mem_size / 96
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float32(1.), np.uint64(elems))
				bytes_transferred = elems * 96
			elif kernelname == 'copyAligned16SpSpinorRestricted':
				kernel = self.prg.copyAligned16SpSpinorRestricted;
				elems = mem_size / 96
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = elems * 96 * 2
			elif kernelname == 'readAligned16SpSpinorRestricted':
				kernel = self.prg.readAligned16SpSpinorRestricted;
				elems = mem_size / 96
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = (elems + global_threads) * 96
			elif kernelname == 'writeAligned16SpSpinorRestricted':
				kernel = self.prg.writeAligned16SpSpinorRestricted;
				elems = mem_size / 96
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float32(1.), np.uint64(elems))
				bytes_transferred = elems * 96
			elif kernelname == 'copyAligned32SpSpinorRestricted':
				kernel = self.prg.copyAligned32SpSpinorRestricted;
				elems = mem_size / 96
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = elems * 96 * 2
			elif kernelname == 'readAligned32SpSpinorRestricted':
				kernel = self.prg.readAligned32SpSpinorRestricted;
				elems = mem_size / 96
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = (elems + global_threads) * 96
			elif kernelname == 'writeAligned32SpSpinorRestricted':
				kernel = self.prg.writeAligned32SpSpinorRestricted;
				elems = mem_size / 96
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float32(1.), np.uint64(elems))
				bytes_transferred = elems * 96

			elif kernelname == 'copyDouble':
				kernel = self.prg.copyDouble;
				elems = mem_size / 8
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = elems * 8 * 2
			elif kernelname == 'readDouble':
				kernel = self.prg.readDouble;
				elems = mem_size / 8
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = (elems + global_threads) * 8
			elif kernelname == 'writeDouble':
				kernel = self.prg.writeDouble;
				elems = mem_size / 8
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float64(1.), np.uint64(elems))
				bytes_transferred = elems * 8
			elif kernelname == 'copyDoubleRestricted':
				kernel = self.prg.copyDoubleRestricted;
				elems = mem_size / 8
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = elems * 8 * 2
			elif kernelname == 'readDoubleRestricted':
				kernel = self.prg.readDoubleRestricted;
				elems = mem_size / 8
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = (elems + global_threads) * 8
			elif kernelname == 'writeDoubleRestricted':
				kernel = self.prg.writeDoubleRestricted;
				elems = mem_size / 8
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float64(1.), np.uint64(elems))
				bytes_transferred = elems * 8
			elif kernelname == 'copyDouble2':
				kernel = self.prg.copyDouble2;
				elems = mem_size / 16
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = elems * 16 * 2
			elif kernelname == 'readDouble2':
				kernel = self.prg.readDouble2;
				elems = mem_size / 16
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = (elems + global_threads) * 16
			elif kernelname == 'writeDouble2':
				kernel = self.prg.writeDouble2;
				elems = mem_size / 16
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float64(1.), np.uint64(elems))
				bytes_transferred = elems * 16
			elif kernelname == 'copyDouble4':
				kernel = self.prg.copyDouble4;
				elems = mem_size / 32
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = elems * 32 * 2
			elif kernelname == 'readDouble4':
				kernel = self.prg.readDouble4;
				elems = mem_size / 32
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = (elems + global_threads) * 32
			elif kernelname == 'writeDouble4':
				kernel = self.prg.writeDouble4;
				elems = mem_size / 32
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float64(1.), np.uint64(elems))
				bytes_transferred = elems * 32

			elif kernelname == 'copyDpComplex':
				kernel = self.prg.copyDpComplex;
				elems = mem_size / 16
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = elems * 16 * 2
			elif kernelname == 'readDpComplex':
				kernel = self.prg.readDpComplex;
				elems = mem_size / 16
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = (elems + global_threads) * 16
			elif kernelname == 'writeDpComplex':
				kernel = self.prg.writeDpComplex;
				elems = mem_size / 16
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float64(1.), np.uint64(elems))
				bytes_transferred = elems * 16
			elif kernelname == 'copyDpComplexRestricted':
				kernel = self.prg.copyDpComplexRestricted;
				elems = mem_size / 16
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = elems * 16 * 2
			elif kernelname == 'readDpComplexRestricted':
				kernel = self.prg.readDpComplexRestricted;
				elems = mem_size / 16
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = (elems + global_threads) * 16
			elif kernelname == 'writeDpComplexRestricted':
				kernel = self.prg.writeDpComplexRestricted;
				elems = mem_size / 16
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float64(1.), np.uint64(elems))
				bytes_transferred = elems * 16
			elif kernelname == 'copyAlignedDpComplex':
				kernel = self.prg.copyAlignedDpComplex;
				elems = mem_size / 16
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = elems * 16 * 2
			elif kernelname == 'readAlignedDpComplex':
				kernel = self.prg.readAlignedDpComplex;
				elems = mem_size / 16
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = (elems + global_threads) * 16
			elif kernelname == 'writeAlignedDpComplex':
				kernel = self.prg.writeAlignedDpComplex;
				elems = mem_size / 16
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float64(1.), np.uint64(elems))
				bytes_transferred = elems * 16
			elif kernelname == 'copyAlignedDpComplexRestricted':
				kernel = self.prg.copyAlignedDpComplexRestricted;
				elems = mem_size / 16
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = elems * 16 * 2
			elif kernelname == 'readAlignedDpComplexRestricted':
				kernel = self.prg.readAlignedDpComplexRestricted;
				elems = mem_size / 16
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = (elems + global_threads) * 16
			elif kernelname == 'writeAlignedDpComplexRestricted':
				kernel = self.prg.writeAlignedDpComplexRestricted;
				elems = mem_size / 16
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float64(1.), np.uint64(elems))
				bytes_transferred = elems * 16

			elif kernelname == 'copyDpSu3vec':
				kernel = self.prg.copyDpSu3vec;
				elems = mem_size / 48
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = elems * 48 * 2
			elif kernelname == 'readDpSu3vec':
				kernel = self.prg.readDpSu3vec;
				elems = mem_size / 48
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = (elems + global_threads) * 48
			elif kernelname == 'writeDpSu3vec':
				kernel = self.prg.writeDpSu3vec;
				elems = mem_size / 48
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float32(1.), np.uint64(elems))
				bytes_transferred = elems * 48
			elif kernelname == 'copyDpSu3vecRestricted':
				kernel = self.prg.copyDpSu3vecRestricted;
				elems = mem_size / 48
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = elems * 48 * 2
			elif kernelname == 'readDpSu3vecRestricted':
				kernel = self.prg.readDpSu3vecRestricted;
				elems = mem_size / 48
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = (elems + global_threads) * 48
			elif kernelname == 'writeDpSu3vecRestricted':
				kernel = self.prg.writeDpSu3vecRestricted;
				elems = mem_size / 48
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float32(1.), np.uint64(elems))
				bytes_transferred = elems * 48
			elif kernelname == 'copyAligned16DpSu3vecRestricted':
				kernel = self.prg.copyAligned16DpSu3vecRestricted;
				elems = mem_size / 48
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = elems * 48 * 2
			elif kernelname == 'readAligned16DpSu3vecRestricted':
				kernel = self.prg.readAligned16DpSu3vecRestricted;
				elems = mem_size / 48
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = (elems + global_threads) * 48
			elif kernelname == 'writeAligned16DpSu3vecRestricted':
				kernel = self.prg.writeAligned16DpSu3vecRestricted;
				elems = mem_size / 48
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float32(1.), np.uint64(elems))
				bytes_transferred = elems * 48
			elif kernelname == 'copyAligned32DpSu3vecRestricted':
				kernel = self.prg.copyAligned32DpSu3vecRestricted;
				elems = mem_size / 64
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = elems * 48 * 2
			elif kernelname == 'readAligned32DpSu3vecRestricted':
				kernel = self.prg.readAligned32DpSu3vecRestricted;
				elems = mem_size / 64
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = (elems + global_threads) * 48
			elif kernelname == 'writeAligned32DpSu3vecRestricted':
				kernel = self.prg.writeAligned32DpSu3vecRestricted;
				elems = mem_size / 64
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float32(1.), np.uint64(elems))
				bytes_transferred = elems * 48
			elif kernelname == 'copyDpSu3vecSOARestricted':
				kernel = self.prg.copyDpSu3vecSOARestricted;
				elems = mem_size / 48
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems), np.uint64(stride))
				bytes_transferred = elems * 48 * 2
			elif kernelname == 'readDpSu3vecSOARestricted':
				kernel = self.prg.readDpSu3vecSOARestricted;
				elems = mem_size / 48
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems), np.uint64(stride))
				bytes_transferred = (elems + global_threads) * 48
			elif kernelname == 'writeDpSu3vecSOARestricted':
				kernel = self.prg.writeDpSu3vecSOARestricted;
				elems = mem_size / 48
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float32(1.), np.uint64(elems), np.uint64(stride))
				bytes_transferred = elems * 48
			elif kernelname == 'copyDpSu3vecFullSOARestricted':
				kernel = self.prg.copyDpSu3vecFullSOARestricted;
				elems = mem_size / 48
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems), np.uint64(stride))
				bytes_transferred = elems * 48 * 2
			elif kernelname == 'readDpSu3vecFullSOARestricted':
				kernel = self.prg.readDpSu3vecFullSOARestricted;
				elems = mem_size / 48
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems), np.uint64(stride))
				bytes_transferred = (elems + global_threads) * 48
			elif kernelname == 'writeDpSu3vecFullSOARestricted':
				kernel = self.prg.writeDpSu3vecFullSOARestricted;
				elems = mem_size / 48
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float32(1.), np.uint64(elems), np.uint64(stride))
				bytes_transferred = elems * 48

			elif kernelname == 'copyDpSu3':
				kernel = self.prg.copyDpSu3;
				elems = mem_size / 144
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = elems * 144 * 2
			elif kernelname == 'readDpSu3':
				kernel = self.prg.readDpSu3;
				elems = mem_size / 144
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = (elems + global_threads) * 144
			elif kernelname == 'writeDpSu3':
				kernel = self.prg.writeDpSu3;
				elems = mem_size / 144
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float32(1.), np.uint64(elems))
				bytes_transferred = elems * 144
			elif kernelname == 'copyDpSu3Restricted':
				kernel = self.prg.copyDpSu3Restricted;
				elems = mem_size / 144
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = elems * 144 * 2
			elif kernelname == 'readDpSu3Restricted':
				kernel = self.prg.readDpSu3Restricted;
				elems = mem_size / 144
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = (elems + global_threads) * 144
			elif kernelname == 'writeDpSu3Restricted':
				kernel = self.prg.writeDpSu3Restricted;
				elems = mem_size / 144
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float32(1.), np.uint64(elems))
				bytes_transferred = elems * 144
			elif kernelname == 'copyAligned16DpSu3Restricted':
				kernel = self.prg.copyAligned16DpSu3Restricted;
				elems = mem_size / 144
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = elems * 144 * 2
			elif kernelname == 'readAligned16DpSu3Restricted':
				kernel = self.prg.readAligned16DpSu3Restricted;
				elems = mem_size / 144
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = (elems + global_threads) * 144
			elif kernelname == 'writeAligned16DpSu3Restricted':
				kernel = self.prg.writeAligned16DpSu3Restricted;
				elems = mem_size / 144
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float32(1.), np.uint64(elems))
				bytes_transferred = elems * 144
			elif kernelname == 'copyAligned32DpSu3Restricted':
				kernel = self.prg.copyAligned32DpSu3Restricted;
				elems = mem_size / 160
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = elems * 144 * 2
			elif kernelname == 'readAligned32DpSu3Restricted':
				kernel = self.prg.readAligned32DpSu3Restricted;
				elems = mem_size / 160
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = (elems + global_threads) * 144
			elif kernelname == 'writeAligned32DpSu3Restricted':
				kernel = self.prg.writeAligned32DpSu3Restricted;
				elems = mem_size / 160
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float32(1.), np.uint64(elems))
				bytes_transferred = elems * 144
			elif kernelname == 'copyDpSu3SOARestricted':
				kernel = self.prg.copyDpSu3SOARestricted;
				elems = mem_size / 144
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems), np.uint64(stride))
				bytes_transferred = elems * 144 * 2
			elif kernelname == 'readDpSu3SOARestricted':
				kernel = self.prg.readDpSu3SOARestricted;
				elems = mem_size / 144
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems), np.uint64(stride))
				bytes_transferred = (elems + global_threads) * 144
			elif kernelname == 'writeDpSu3SOARestricted':
				kernel = self.prg.writeDpSu3SOARestricted;
				elems = mem_size / 144
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float32(1.), np.uint64(elems), np.uint64(stride))
				bytes_transferred = elems * 144
			elif kernelname == 'copyDpSu3FullSOARestricted':
				kernel = self.prg.copyDpSu3FullSOARestricted;
				elems = mem_size / 144
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems), np.uint64(stride))
				bytes_transferred = elems * 144 * 2
			elif kernelname == 'readDpSu3FullSOARestricted':
				kernel = self.prg.readDpSu3FullSOARestricted;
				elems = mem_size / 144
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems), np.uint64(stride))
				bytes_transferred = (elems + global_threads) * 144
			elif kernelname == 'writeDpSu3FullSOARestricted':
				kernel = self.prg.writeDpSu3FullSOARestricted;
				elems = mem_size / 144
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float32(1.), np.uint64(elems), np.uint64(stride))
				bytes_transferred = elems * 144

			elif kernelname == 'copyDpSpinor':
				kernel = self.prg.copyDpSpinor;
				elems = mem_size / 192
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = elems * 192 * 2
			elif kernelname == 'readDpSpinor':
				kernel = self.prg.readDpSpinor;
				elems = mem_size / 192
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = (elems + global_threads) * 192
			elif kernelname == 'writeDpSpinor':
				kernel = self.prg.writeDpSpinor;
				elems = mem_size / 192
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float32(1.), np.uint64(elems))
				bytes_transferred = elems * 192
			elif kernelname == 'copyDpSpinorRestricted':
				kernel = self.prg.copyDpSpinorRestricted;
				elems = mem_size / 192
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = elems * 192 * 2
			elif kernelname == 'readDpSpinorRestricted':
				kernel = self.prg.readDpSpinorRestricted;
				elems = mem_size / 192
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = (elems + global_threads) * 192
			elif kernelname == 'writeDpSpinorRestricted':
				kernel = self.prg.writeDpSpinorRestricted;
				elems = mem_size / 192
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float32(1.), np.uint64(elems))
				bytes_transferred = elems * 192
			elif kernelname == 'copyAligned16DpSpinorRestricted':
				kernel = self.prg.copyAligned16DpSpinorRestricted;
				elems = mem_size / 192
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = elems * 192 * 2
			elif kernelname == 'readAligned16DpSpinorRestricted':
				kernel = self.prg.readAligned16DpSpinorRestricted;
				elems = mem_size / 192
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = (elems + global_threads) * 192
			elif kernelname == 'writeAligned16DpSpinorRestricted':
				kernel = self.prg.writeAligned16DpSpinorRestricted;
				elems = mem_size / 192
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float32(1.), np.uint64(elems))
				bytes_transferred = elems * 192
			elif kernelname == 'copyAligned32DpSpinorRestricted':
				kernel = self.prg.copyAligned32DpSpinorRestricted;
				elems = mem_size / 192
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = elems * 192 * 2
			elif kernelname == 'readAligned32DpSpinorRestricted':
				kernel = self.prg.readAligned32DpSpinorRestricted;
				elems = mem_size / 192
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems))
				bytes_transferred = (elems + global_threads) * 192
			elif kernelname == 'writeAligned32DpSpinorRestricted':
				kernel = self.prg.writeAligned32DpSpinorRestricted;
				elems = mem_size / 192
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float32(1.), np.uint64(elems))
				bytes_transferred = elems * 192
			elif kernelname == 'copyDpSpinorSOARestricted':
				kernel = self.prg.copyDpSpinorSOARestricted;
				elems = mem_size / 192
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems), np.uint64(stride))
				bytes_transferred = elems * 192 * 2
			elif kernelname == 'readDpSpinorSOARestricted':
				kernel = self.prg.readDpSpinorSOARestricted;
				elems = mem_size / 192
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems), np.uint64(stride))
				bytes_transferred = (elems + global_threads) * 192
			elif kernelname == 'writeDpSpinorSOARestricted':
				kernel = self.prg.writeDpSpinorSOARestricted;
				elems = mem_size / 192
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float32(1.), np.uint64(elems), np.uint64(stride))
				bytes_transferred = elems * 192
			elif kernelname == 'copyDpSpinorFullSOARestricted':
				kernel = self.prg.copyDpSpinorFullSOARestricted;
				elems = mem_size / 192
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems), np.uint64(stride))
				bytes_transferred = elems * 192 * 2
			elif kernelname == 'readDpSpinorFullSOARestricted':
				kernel = self.prg.readDpSpinorFullSOARestricted;
				elems = mem_size / 192
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems), np.uint64(stride))
				bytes_transferred = (elems + global_threads) * 192
			elif kernelname == 'writeDpSpinorFullSOARestricted':
				kernel = self.prg.writeDpSpinorFullSOARestricted;
				elems = mem_size / 192
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float32(1.), np.uint64(elems), np.uint64(stride))
				bytes_transferred = elems * 192
			elif kernelname == 'copyDpSpinorFullestSOARestricted':
				kernel = self.prg.copyDpSpinorFullestSOARestricted;
				elems = mem_size / 192
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems), np.uint64(stride))
				bytes_transferred = elems * 192 * 2
			elif kernelname == 'readDpSpinorFullestSOARestricted':
				kernel = self.prg.readDpSpinorFullestSOARestricted;
				elems = mem_size / 192
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, self.in_buf, np.uint64(elems), np.uint64(stride))
				bytes_transferred = (elems + global_threads) * 192
			elif kernelname == 'writeDpSpinorFullestSOARestricted':
				kernel = self.prg.writeDpSpinorFullestSOARestricted;
				elems = mem_size / 192
				event = kernel(self.queue, (global_threads,), (local_threads,), self.out_buf, np.float32(1.), np.uint64(elems), np.uint64(stride))
				bytes_transferred = elems * 192

			else:
				raise NameError( "Don't know how to run {0}".format(kernelname) )

			events.append(event)

		cl.wait_for_events(events)

		# throw away warmup runs
		events = events[WARMUP_RUNS:]
		event_times = map(lambda event: (event.profile.end - event.profile.start), events)
		elapsed = np.mean(event_times)
		elapsed_std = np.std(event_times)

		return DataPoint(kernelname, global_threads, local_threads, bytes_transferred, elapsed, elapsed_std, bytes_transferred / elapsed)
