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

MAX_MEM_SIZE = 10 * 1024 * 1024 # 10 MiB
LOCAL_THREADS = 128
GLOBAL_THREADS = 20 * 8 * LOCAL_THREADS

class Runner:

	def __init__(self):
		self.ctx = cl.create_some_context()
		self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

		self.device = self.queue.device
		print '#Device: {0}'.format(self.device.name)
		print '#Memory size: {0} KiB'.format(self.device.global_mem_size / 1024)
		print '#Maximum buffer size: {0} KiB'.format(self.device.max_mem_alloc_size / 1024)

		f = open('kernels.cl', 'r')
		fstr = "".join(f.readlines())
		self.prg = cl.Program(self.ctx, fstr).build()

		self.in_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY, MAX_MEM_SIZE)
		self.out_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, MAX_MEM_SIZE)

	def benchmark(self, kernelname):
		if kernelname == 'copyFloat':
			kernel = self.prg.copyFloat;
			elems = MAX_MEM_SIZE / 4
			event = kernel(self.queue, (GLOBAL_THREADS,), (LOCAL_THREADS,), self.out_buf, self.in_buf, np.uint64(elems))
			bytes_transferred = elems * 4 * 2
		elif kernelname == 'readFloat':
			kernel = self.prg.readFloat;
			elems = MAX_MEM_SIZE / 4
			event = kernel(self.queue, (GLOBAL_THREADS,), (LOCAL_THREADS,), self.out_buf, self.in_buf, np.uint64(elems))
			bytes_transferred = (elems + GLOBAL_THREADS) * 4
		elif kernelname == 'writeFloat':
			kernel = self.prg.writeFloat;
			elems = MAX_MEM_SIZE / 4
			event = kernel(self.queue, (GLOBAL_THREADS,), (LOCAL_THREADS,), self.out_buf, np.float32(1.), np.uint64(elems))
			bytes_transferred = elems * 4
		else:
			print "Don't know how to run {0}".format(kernelname)
			return None

		event.wait()
		elapsed = (event.profile.end - event.profile.start);
		print '{0} {1} {2}'.format(bytes_transferred, elapsed, bytes_transferred / float(elapsed))


if __name__ == '__main__':
	runner = Runner()

	print '#Kernel Bytes nanos GB/s'
	print 'copyFloat ',
	runner.benchmark('copyFloat')
	print 'readFloat ',
	runner.benchmark('readFloat')
	print 'writeFloat ',
	runner.benchmark('writeFloat')

