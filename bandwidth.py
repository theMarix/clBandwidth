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

from runner import *

if __name__ == '__main__':
	parser = optparse.OptionParser(description='Benchmark global memory bandwidth')
	parser.add_option('-d', '--device', type=int, metavar='I', help='The device to use for the measurement')
	parser.add_option('-p', '--plot', action='store_true', default=False, help='Make a plot of the measurements')
	parser.add_option('-e', '--plot-errorbars', action='store_true', default=False, help='Add error bars to the plot')
	parser.add_option('-m', '--plot-marker', type=float, dest='plot_markers', action='append', metavar='GB/s', help='Add a marker to the plot at the given performance')
	parser.add_option('-o', '--plot-file', metavar='FILE', help='File to store the plot in (Display if unset)')
	parser.add_option('-g', '--global-threads', type=int, metavar='NUM', help='The number of global threads to use')
	parser.add_option('-l', '--local-threads', type=int, metavar='NUM', help='The number of global threads to use')
	parser.add_option('-s', '--mem-size', type=int, metavar='BYTE', help='Memory size in byte')

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

	runner = Runner(**runner_args)

	datapoints = []

	print '#Kernel Bytes nanos (rel err) GB/s'
	for kernel in runner.get_kernel_names():
		try:
			datapoints.append(runner.benchmark(kernel))
		except (cl.RuntimeError, cl.LogicError) as ex:
			# On Apples OpenCL retrieving the profiling information sometimes seems to fail for no good reason
			# In addition, sometimes the queue becomes invalid
			print 'Error benchmarking {0}: {1}'.format(kernel, ex)


	for datapoint in datapoints:
		print '{0.kernel} {0.bytes_transferred} {0.time:.0f} ({1:.1%}) {0.bandwidth}'.format(datapoint, datapoint.time_std / datapoint.time)

	if args.plot:
		import matplotlib.pyplot as plt # by including it here we won't need it unless we want to plot

		bandwidths = map(lambda p: p.bandwidth, datapoints)
		xticks = map(lambda p: p.kernel, datapoints)
		if args.plot_errorbars:
			errs = map(lambda p: p.time_std / p.time * p.bandwidth, datapoints)
		ind = np.arange(len(datapoints)) + 0.5

		plt.figure(figsize=(2 + len(datapoints) * .25,12))
		plt.axes([0.125,.4,.775,.5])
		plt.title('Global Memory Bandwidth')
		plt.ylabel('GB/s')
		if args.plot_errorbars:
			plt.bar(ind, bandwidths, align='center', yerr=errs, ecolor='black')
		else:
			plt.bar(ind, bandwidths, align='center')

		# handle markers
		if args.plot_markers:
			for marker in args.plot_markers:
				plt.axhline(y=marker, color='r')

		# Avoid the error bars triggering the lower quadrant to show
		ymin, ymax = plt.ylim()
		plt.ylim(0, ymax)

		plt.xticks(ind, xticks, rotation='vertical')

		if args.plot_file:
			plt.savefig(args.plot_file)
		else:
			plt.show()
