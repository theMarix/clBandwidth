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
import optparse
import csv
from collections import namedtuple

from runner import *

StrideDataPoint = namedtuple('StrideDataPoint', DataPoint._fields + ('stride',))

if __name__ == '__main__':
	parser = optparse.OptionParser(description='Benchmark global memory bandwidth for different memory strides. Be aware, that strides only apply to SOA kernels. If you adjust the kernel you will also have to adjust the struct-elems and struct-elem-size parameters.')
	parser.add_option('-d', '--device', type=int, metavar='I', help='The device to use for the measurement')
	parser.add_option('-p', '--plot', action='store_true', default=False, help='Make a plot of the measurements')
	parser.add_option('-e', '--plot-errorbars', action='store_true', default=False, help='Add error bars to the plot')
	parser.add_option('-b', '--plot-marker', type=float, dest='plot_markers', action='append', metavar='GB/s', help='Add a marker to the plot at the given performance')
	parser.add_option('-o', '--plot-file', metavar='FILE', help='File to store the plot in (Display if unset)')
	parser.add_option('-g', '--global-threads', type=int, metavar='NUM', help='The number of global threads to use')
	parser.add_option('-l', '--local-threads', type=int, metavar='NUM', help='The number of global threads to use')
	parser.add_option('-m', '--mem-size', type=int, metavar='BYTE', help='Memory to transfer in bytes')
	parser.add_option('-k', '--kernel', metavar='KERNEL', default='copyDpSpinorFullSOARestricted', help='The kernel to benchmark')
	parser.add_option('--struct-elems', metavar='N', type=int, default=12, help='The number of elements in one struct')
	parser.add_option('--struct-elems-size', metavar='BYTES', type=int, default=16, help='The size of one element of the struct')
	parser.add_option('--progress', action='store_true', default=False, help='Display a progress bar while running kernels')
	parser.add_option('--export', metavar='FILE', help='Export measurement results to a CSV file')
	parser.add_option('--import', metavar='FILE', action='append', help='Import data from file instead of benchmarking', dest='imports')
	parser.add_option('--sweep-sizes', action='store_true', default=False, help='Also sweep sizes, generates a pseudo-color plot when plotting')
	parser.add_option('--plot-norm-x', metavar='N', default=1, help='Normalize x axis of plot by N')

	(args, rem) = parser.parse_args()

	runner_args = {}
	if args.device != None:
		runner_args['device'] = args.device
	if args.global_threads != None:
		runner_args['global_threads'] = args.global_threads
	if args.local_threads != None:
		runner_args['local_threads'] = args.local_threads
	if args.mem_size != None:
		runner_args['max_mem_size'] = args.mem_size

	if args.imports == None: # no data file given, run benchmark

		runner = Runner(**runner_args)

		datapoints = []

		# The minimum possible stride is elems * struct-elems-size
		# For strides larger than 2 * elems * struct-elems-size no new effects should occur (assuming large values for elems...)
		# The maximum memory required will therefore be 2 * elems * struct-elems * struct-elems-size

		if args.progress:
			from progressbar import ProgressBar
			progress = ProgressBar().start()

		if args.sweep_sizes:
			sizes = range(args.struct_elems * args.struct_elems_size, runner.max_mem_size, args.struct_elems * args.struct_elems_size)
		else:
			sizes = [runner.max_mem_size]

		for size in sizes:

			elems = size / 2 / args.struct_elems / args.struct_elems_size
			min_stride = elems * args.struct_elems_size
			max_stride = 2 * min_stride
			transfer_size = min_stride * args.struct_elems

			for stride in range(min_stride, max_stride, args.struct_elems_size):
				try:
					datapoint = runner.benchmark(args.kernel, mem_size = transfer_size, stride = stride / args.struct_elems_size)
					datapoints.append(StrideDataPoint(*datapoint, stride = stride))
					if args.progress:
						if len(sizes) > 1:
							progress.update(float(size) / runner.max_mem_size * 100)
						else:
							progress.update((stride - min_stride) / float(max_stride - min_stride) * 100)
				except (cl.RuntimeError, cl.LogicError) as ex:
					# On Apples OpenCL retrieving the profiling information sometimes seems to fail for no good reason
					# In addition, sometimes the queue becomes invalid
					print 'Error benchmarking {0}: {1}'.format(kernel, ex)

		if args.progress:
			progress.finish()

		print '#Kernel Bytes nanos (rel err) GB/s'
		for datapoint in datapoints:
			print '{0.kernel} {0.stride} {0.time:.0f} ({1:.1%}) {0.bandwidth}'.format(datapoint, datapoint.time_std / datapoint.time)

		if args.export != None:
			writer = csv.writer(open(args.export, 'wb'), quoting=csv.QUOTE_MINIMAL)
			writer.writerow(datapoints[0]._fields)
			writer.writerows(datapoints)

		to_plot = [datapoints]
		labels = ['']

	else: # data file(s) given. import

		to_plot = []
		for file in args.imports:
			reader = csv.reader(open(file, 'rb'))
			reader.next() # skip headers
			to_plot.append(map(StrideDataPoint._make, reader))
		labels = args.imports

	#reformat data
	even = []
	odd = []
	datapoints = to_plot[0]

	to_plot = [[], [], [], []]
	labels = ['(N/16)%4+0', '(N/16)%4+1', '(N/16)%4+2', '(N/16)%4+3']
	for datapoint in datapoints:
		to_plot[(int(datapoint.stride) / 16 ) % 4].append(datapoint)

	if args.plot:
		import matplotlib.pyplot as plt # by including it here we won't need it unless we want to plot

		plt.figure(figsize=(16,10))

		plt.grid(axis='x')
		plots = []
		idx = 0
		for datapoints in to_plot:
			bandwidths = map(lambda p: float(p.bandwidth), datapoints)
			strides = map(lambda p: int(p.stride) / float(args.plot_norm_x), datapoints)
			if args.plot_errorbars:
				errs = map(lambda p: float(p.time_std) / float(p.time) * float(p.bandwidth), datapoints)

			if args.plot_errorbars:
				plots.append(plt.errorbar(strides, bandwidths, yerr=errs, fmt='.', ecolor='black', label=labels[idx]))
			else:
				plots.append(plt.plot(strides, bandwidths, '.', label=labels[idx]))
			idx += 1

		if args.imports == None:
			plt.title('Global Memory Bandwidth of {0}'.format(args.kernel))
		else:
			plt.title('Global Memory Bandwidth')
			plt.legend(loc='lower right')
		if args.plot_norm_x == 1:
			plt.xlabel('Stride / Bytes')
		else:
			plt.xlabel('Stride / {0} Bytes'.format(args.plot_norm_x))
		plt.ylabel('GB/s')

		# handle markers
		if args.plot_markers:
			for marker in args.plot_markers:
				plt.axhline(y=marker, color='r')

		if args.plot_file:
			plt.savefig(args.plot_file)
		else:
			plt.show()
