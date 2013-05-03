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
import optimizer

if __name__ == '__main__':
	parser = optparse.OptionParser(description='Benchmark global memory bandwidth for multiple memory sizes')
	parser.add_option('-d', '--device', type=int, metavar='I', help='The device to use for the measurement')
	parser.add_option('-m', '--max-mem-size', type=int, default=100*1024*1024, metavar='BYTE', help='Maximum memory size in byte')
	parser.add_option('-n', '--min-mem-size', type=int, metavar='BYTE', default=1*1024*1024, help='Minimum memory size in byte')
	parser.add_option('-s', '--step-mem-size', type=int, metavar='BYTE', default=1*1024*1024, help='Steps for memory size in byte')
	parser.add_option('-t', '--type', default='float', metavar='TYPE', help='The basic scalar type to use')
	parser.add_option('-e', '--struct-elems', type=int, metavar='N', help='Use a struct of N elems of the basic scalar type')
	parser.add_option('--soa', default=False, action='store_true', help='Use SOA storage')
	parser.add_option('--n-soa-buffers', default=False, action='store_true', help='Use distinct buffers for each element fo the struct')
	parser.add_option('-o', '--output-file', help='File to write the results to')
	parser.add_option('-p', '--progress', default=False, action='store_true', help='Show a progress indicator')
	parser.add_option('--optimizer', help='The optimizer to use')
	parser.add_option('--plain-pointers', default=False, action='store_true', help="Dont't hint compiler at cacheability of buffers")

	(args, rem) = parser.parse_args()

	runner_args = {}
	if args.device != None:
		runner_args['device'] = args.device
	if args.optimizer != None:
		runner_args['optimizer'] = optimizer.getOptimizer(args.optimizer)
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
	bench_args['plain_pointers'] = args.plain_pointers

	progress = ProgressBar(not args.progress)
	try:
		for size in progress(range(args.min_mem_size, args.max_mem_size, args.step_mem_size)):
			bench_args['mem_size'] = size
			datapoints.append(runner.benchmark(data_type, **bench_args))
	except (cl.MemoryError, cl.LogicError):
		print 'Not enough memory, dumping already collected results.'

	if args.output_file:
		data.dump(args.output_file, datapoints)
	else:
		print '#Type Bytes nanos (rel err) GB/s'
		for datapoint in datapoints:
			print '{0.typename} {0.bytes_transferred} {0.time:.0f} ({1:.1%}) {0.bandwidth}'.format(datapoint, datapoint.time_std / datapoint.time)
