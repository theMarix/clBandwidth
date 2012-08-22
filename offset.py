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
# (c) 2012 Matthias Bach <bach@compeng.uni-frankfurt.de>

import pyopencl as cl
import optparse
from progress import ProgressBar

from runner import *
from datatypes import getType
import data

if __name__ == '__main__':
	parser = optparse.OptionParser(description='Benchmark global memory bandwidth dependence on access offset to buffer start')
	parser.add_option('-d', '--device', type=int, metavar='I', help='The device to use for the measurement')
	parser.add_option('-s', '--mem-size', type=int, metavar='BYTE', help='Memory size in byte')
	parser.add_option('-m', '--max-offset', type=int, default=1024, metavar='ELEMS', help='Maximum memory offset in elements')
	parser.add_option('-n', '--min-offset', type=int, metavar='ELEMS', default=0, help='Minimum memory offset in elements')
	parser.add_option('-i', '--increment-offset', type=int, metavar='ELEMS', default=1, help='Steps for memory offset in elemts')
	parser.add_option('-t', '--type', default='float', metavar='TYPE', help='The basic scalar type to use')
	parser.add_option('-e', '--struct-elems', type=int, metavar='N', help='Use a struct of N elems of the basic scalar type')
	parser.add_option('--soa', default=False, action='store_true', help='Use SOA storage')
	parser.add_option('--n-soa-buffers', default=False, action='store_true', help='Use distinct buffers for each element fo the struct')
	parser.add_option('-o', '--output-file', help='File to write the results to')
	parser.add_option('-p', '--progress', default=False, action='store_true', help='Show a progress indicator')

	(args, rem) = parser.parse_args()

	runner_args = {}
	if args.device != None:
		runner_args['device'] = args.device
	if args.mem_size:
		runner_args['default_mem_size'] = args.mem_size
	if args.n_soa_buffers:
		runner_args['n_soa_buffers'] = True

	runner = Runner(**runner_args)

	datapoints = []

	scalar_type = getType(args.type)
	if args.struct_elems:
		data_type = Struct(scalar_type, args.struct_elems)
	else:
		data_type = scalar_type

	bench_args = {}
	if args.soa:
		bench_args['stride'] = -1

	progress = ProgressBar(not args.progress)
	try:
		for offset in progress(range(args.min_offset, args.max_offset, args.increment_offset)):
			bench_args['offset'] = offset
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
		print '#Type Transferred/Bytes Offset/Elems nanos (rel err) GB/s'
		for datapoint in datapoints:
			print '{0.typename} {0.bytes_transferred} {0.offset} {0.time:.0f} ({1:.1%}) {0.bandwidth}'.format(datapoint, datapoint.time_std / datapoint.time)

