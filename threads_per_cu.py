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
# (c) 2013 Matthias Bach <bach@compeng.uni-frankfurt.de>

import pyopencl as cl
import optparse

from runner import *
from datatypes import getType
import data
import optimizer

if __name__ == '__main__':
	parser = optparse.OptionParser(description='Benchmark global memory bandwidth')
	parser.add_option('-d', '--device', type=int, metavar='I', help='The device to use for the measurement')
	parser.add_option('-s', '--mem-size', type=int, metavar='BYTE', help='Memory size in byte')
	parser.add_option('-t', '--type', default='float', metavar='TYPE', help='The basic scalar type to use')
	parser.add_option('-e', '--struct-elems', type=int, metavar='N', help='Use a struct of N elems of the basic scalar type')
	parser.add_option('--soa', default=False, action='store_true', help='Use SOA storage')
	parser.add_option('-n', '--n-soa-buffers', default=False, action='store_true', help='Use distinct buffers for each element fo the struct')
	parser.add_option('--offset', default=0, type=int, metavar='N', help='Offset (in elements) to use for array vs. buffer')
	parser.add_option('-o', '--output-file', help='File to write the results to')
	parser.add_option('--optimizer', help='The optimizer to use')
	parser.add_option('-g', '--group-size', type=int, default=64)
	parser.add_option('-c', '--groups-per-cu', type=int, default=4)

	(args, rem) = parser.parse_args()

	runner_args = {}
	if args.device != None:
		runner_args['device'] = args.device
	if args.mem_size != None:
		runner_args['default_mem_size'] = args.mem_size
	if args.optimizer != None:
		runner_args['optimizer'] = optimizer.getOptimizer(args.optimizer)
	if args.n_soa_buffers:
		runner_args['n_soa_buffers'] = True
	runner_args['local_threads'] = args.group_size

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
	if args.offset:
		bench_args['offset'] = args.offset
	bench_args['max_groups_per_cu'] = args.groups_per_cu

	try:
		datapoints.append(runner.benchmark(data_type, **bench_args))
	except (cl.RuntimeError, cl.LogicError) as ex:
		# On Apples OpenCL retrieving the profiling information sometimes seems to fail for no good reason
		# In addition, sometimes the queue becomes invalid
		print 'Error benchmarking {0}: {1}'.format(args.type, ex)

	if args.output_file:
		data.dump(args.output_file, datapoints)
	else:
		print '#Type Bytes nanos (rel err) GB/s'
		for datapoint in datapoints:
			print '{0.typename} {0.bytes_transferred} {0.time:.0f} ({1:.1%}) {0.bandwidth}'.format(datapoint, datapoint.time_std / datapoint.time)
