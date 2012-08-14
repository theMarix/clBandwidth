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
import optparse
from progress import ProgressBar

from runner import *
from datatypes import getType
import data

if __name__ == '__main__':
	parser = optparse.OptionParser(description='Benchmark global memory bandwidth for different memory strides. Be aware, that strides only apply to SOA kernels.')
	parser = optparse.OptionParser(description='Benchmark global memory bandwidth')
	parser.add_option('-d', '--device', type=int, metavar='I', help='The device to use for the measurement')
	parser.add_option('-s', '--mem-size', type=int, metavar='BYTE', help='Memory size in byte')
	parser.add_option('-m', '--max-add-stride', type=int, default=1024, metavar='ELEMS', help='Maximum additional memory stride in elements')
	parser.add_option('-n', '--min-add-stride', type=int, metavar='ELEMS', default=0, help='Minimum additional memory stride in elements')
	parser.add_option('-i', '--increment-stride', type=int, metavar='ELEMS', default=1, help='Steps for memory stride in elements')
	parser.add_option('-t', '--type', default='float', metavar='TYPE', help='The basic scalar type to use')
	parser.add_option('-e', '--struct-elems', type=int, default=2, metavar='N', help='Use a struct of N elems of the basic scalar type')
	parser.add_option('-o', '--output-file', help='File to write the results to')
	parser.add_option('-p', '--progress', default=False, action='store_true', help='Show a progress indicator')
	parser.add_option('--plain-pointers', default=False, action='store_true', help="Don't restrict pointers and mark them read-only.")

	(args, rem) = parser.parse_args()

	runner_args = {}
	if args.device != None:
		runner_args['device'] = args.device
	if args.mem_size:
		runner_args['default_mem_size'] = args.mem_size

	runner = Runner(**runner_args)

	datapoints = []

	scalar_type = getType(args.type)
	data_type = Struct(scalar_type, args.struct_elems)

	bench_args = {}
	if args.plain_pointers:
		bench_args['plain_pointers'] = True

	# make sure all runs are the same size
	elems = runner.default_mem_size / data_type.size

	progress = ProgressBar(not args.progress)
	try:
		for add_stride in progress(range(args.min_add_stride, args.max_add_stride, args.increment_stride)):
			bench_args['stride'] = elems + add_stride
			try:
				datapoints.append(runner.benchmark(data_type, **bench_args))
			except (cl.RuntimeError, cl.LogicError) as ex:
				# On Apples OpenCL retrieving the profiling information sometimes seems to fail for no good reason
				# In addition, sometimes the queue becomes invalid
				print 'Error benchmarking {0}: {1}'.format(args.type, ex)
	except (cl.MemoryError, cl.LogicError):
		print 'Not enough memory, dumping already collected results.'

	if args.output_file:
		data.dump(args.output_file, datapoints)
	else:
		print '#Type Transferred/Bytes Stride/Elems nanos (rel err) GB/s'
		for datapoint in datapoints:
			print '{0.typename} {0.bytes_transferred} {0.stride} {0.time:.0f} ({1:.1%}) {0.bandwidth}'.format(datapoint, datapoint.time_std / datapoint.time)

