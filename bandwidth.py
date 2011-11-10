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
import optparse

MAX_MEM_SIZE = 10 * 1024 * 1024 # 10 MiB
LOCAL_THREADS = 128
GLOBAL_THREADS = 20 * 8 * LOCAL_THREADS

class Runner:

	def __init__(self, device = None, local_threads = LOCAL_THREADS, global_threads = GLOBAL_THREADS):
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

		self.in_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY, MAX_MEM_SIZE)
		self.out_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, MAX_MEM_SIZE)

		self.local_threads = local_threads
		self.global_threads = global_threads

	def hasDoublePrecisionSupport(self):
		extensions = self.device.extensions
		return 'cl_khr_fp64' in extensions or 'cl_amd_fp64' in extensions

	def benchmark(self, kernelname, mem_size = MAX_MEM_SIZE, global_threads = None, local_threads = None):
		BENCH_RUNS = 10
		WARMUP_RUNS = 2

		if not global_threads:
			global_threads = self.global_threads
		if not local_threads:
			local_threads = self.local_threads

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
			else:
				raise NameError( "Don't know how to run {0}".format(kernelname) )

			events.append(event)

		cl.wait_for_events(events)

		# throw away warmup runs
		events = events[WARMUP_RUNS:]
		event_times = map(lambda event: (event.profile.end - event.profile.start), events)
		elapsed = np.mean(event_times)
		elapsed_std = np.std(event_times)

		print '{0} {1:.0f} ({2:.1f}%) {3:.3f}'.format(bytes_transferred, elapsed, elapsed_std / elapsed * 100, bytes_transferred / elapsed)


if __name__ == '__main__':
	parser = optparse.OptionParser(description='Benchmark global memory bandwidth')
	parser.add_option('-d', '--device', dest='device', type=int, metavar='I', help='The device to use for the measurement')

	(args, rem) = parser.parse_args()

	if args.device != None:
		runner = Runner(args.device)
	else:
		runner = Runner()

	print '#Kernel Bytes nanos (rel err) GB/s'
	print 'copyFloat ',
	runner.benchmark('copyFloat')
	print 'readFloat ',
	runner.benchmark('readFloat')
	print 'writeFloat ',
	runner.benchmark('writeFloat')
	print 'copyFloatRestricted ',
	runner.benchmark('copyFloatRestricted')
	print 'readFloatRestricted ',
	runner.benchmark('readFloatRestricted')
	print 'writeFloatRestricted ',
	runner.benchmark('writeFloatRestricted')
	print 'copyFloat4 ',
	runner.benchmark('copyFloat4')
	print 'readFloat4 ',
	runner.benchmark('readFloat4')
	print 'writeFloat4 ',
	runner.benchmark('writeFloat4')

