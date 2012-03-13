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

from runner import *
from datatypes import getType

if __name__ == '__main__':
	parser = optparse.OptionParser(description='Benchmark global memory bandwidth')
	parser.add_option('-d', '--device', type=int, metavar='I', help='The device to use for the measurement')
	parser.add_option('-s', '--mem-size', type=int, metavar='BYTE', help='Memory size in byte')
	parser.add_option('-t', '--type', default='float', metavar='TYPE', help='The basic scalar type to use')
	parser.add_option('-e', '--struct-elems', type=int, metavar='N', help='Use a struct of N elems of the basic scalar type')
	parser.add_option('--soa', default=False, action='store_true', help='Use SOA storage')

	(args, rem) = parser.parse_args()

	runner_args = {}
	if args.device != None:
		runner_args['device'] = args.device
	if args.mem_size != None:
		runner_args['max_mem_size'] = args.mem_size

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

	print '#Type Bytes nanos (rel err) GB/s'
	try:
		datapoints.append(runner.benchmark(data_type, **bench_args))
	except (cl.RuntimeError, cl.LogicError) as ex:
		# On Apples OpenCL retrieving the profiling information sometimes seems to fail for no good reason
		# In addition, sometimes the queue becomes invalid
		print 'Error benchmarking {0}: {1}'.format(args.type, ex)


	for datapoint in datapoints:
		print '{0.kernel} {0.bytes_transferred} {0.time:.0f} ({1:.1%}) {0.bandwidth}'.format(datapoint, datapoint.time_std / datapoint.time)
