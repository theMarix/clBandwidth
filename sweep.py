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

from runner import *

if __name__ == '__main__':
	parser = optparse.OptionParser(description='Benchmark global memory bandwidth')
	parser.add_option('-d', '--device', type=int, metavar='I', help='The device to use for the measurement')
	parser.add_option('-p', '--plot', action='store_true', default=False, help='Make a plot of the measurements')
	parser.add_option('-e', '--plot-errorbars', action='store_true', default=False, help='Add error bars to the plot')
	parser.add_option('-b', '--plot-marker', type=float, dest='plot_markers', action='append', metavar='GB/s', help='Add a marker to the plot at the given performance')
	parser.add_option('-o', '--plot-file', metavar='FILE', help='File to store the plot in (Display if unset)')
	parser.add_option('-g', '--global-threads', type=int, metavar='NUM', help='The number of global threads to use')
	parser.add_option('-l', '--local-threads', type=int, metavar='NUM', help='The number of global threads to use')
	parser.add_option('-m', '--mem-max-size', type=int, metavar='BYTE', help='Maximum memory size in bytes')
	parser.add_option('-s', '--mem-step-size', type=int, default=192, metavar='BYTE', help='Memory step size in bytes')
	parser.add_option('-k', '--kernel', metavar='KERNEL', default='copyDpSpinorFullSOARestricted', help='The kernel to benchmark')
	parser.add_option('--progress', action='store_true', default=False, help='Display a progress bar while running kernels')
	parser.add_option('--export', metavar='FILE', help='Export measurement results to a CSV file')

	(args, rem) = parser.parse_args()

	runner_args = {}
	if args.device != None:
		runner_args['device'] = args.device
	if args.global_threads != None:
		runner_args['global_threads'] = args.global_threads
	if args.local_threads != None:
		runner_args['local_threads'] = args.local_threads
	if args.mem_max_size != None:
		runner_args['max_mem_size'] = args.mem_max_size

	runner = Runner(**runner_args)

	if args.progress:
		from progressbar import ProgressBar
		progress = ProgressBar(maxval=runner.max_mem_size).start()

	datapoints = []

	for size in range(args.mem_step_size, runner.max_mem_size, args.mem_step_size):
		try:
			datapoints.append(runner.benchmark(args.kernel, mem_size = size))
			if args.progress:
				progress.update(size)
		except (cl.RuntimeError, cl.LogicError) as ex:
			# On Apples OpenCL retrieving the profiling information sometimes seems to fail for no good reason
			# In addition, sometimes the queue becomes invalid
			print 'Error benchmarking {0}: {1}'.format(kernel, ex)

	if args.progress:
		progress.finish()

	print '#Kernel Bytes nanos (rel err) GB/s'
	for datapoint in datapoints:
		print '{0.kernel} {0.bytes_transferred} {0.time:.0f} ({1:.1%}) {0.bandwidth}'.format(datapoint, datapoint.time_std / datapoint.time)

	if args.export != None:
		writer = csv.writer(open(args.export, 'wb'), quoting=csv.QUOTE_MINIMAL)
		writer.writerow(datapoints[0]._fields)
		writer.writerows(datapoints)

	if args.plot:
		import matplotlib.pyplot as plt # by including it here we won't need it unless we want to plot

		bandwidths = map(lambda p: p.bandwidth, datapoints)
		sizes = map(lambda p: p.bytes_transferred, datapoints)
		if args.plot_errorbars:
			errs = map(lambda p: p.time_std / p.time * p.bandwidth, datapoints)

		plt.title('Global Memory Bandwidth of {0}'.format(args.kernel))
		plt.xlabel('Transferred Bytes')
		plt.ylabel('GB/s')
		if args.plot_errorbars:
			plt.errorbar(sizes, bandwidths, yerr=errs, fmt='.', ecolor='black')
		else:
			plt.plot(sizes, bandwidths, '.')

		# handle markers
		if args.plot_markers:
			for marker in args.plot_markers:
				plt.axhline(y=marker, color='r')

		if args.plot_file:
			plt.savefig(args.plot_file)
		else:
			plt.show()
