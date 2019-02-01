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

import argparse

import data
import optimizer
import datatypes

if __name__ == '__main__':
	# parse the user request
	parser = argparse.ArgumentParser(description='Filter data file')
	parser.add_argument('file', metavar='FILE', help='The file to filter')
	parser.add_argument('-m', '--metric', default='bandwidth', help='The metric to filter by')
	parser.add_argument('--min', type=float, help='Minimum value cutoff')
	parser.add_argument('--max', type=float, help='Maximum value cutoff')
	parser.add_argument('--mod', type=int, help='Only show lines where the value is a multiple of this value')
	parser.add_argument('--max-error', type=float, help='Maximum error of the time measurement in percent')
	parser.add_argument('--bad-strides', metavar='OPTIMIZER', help='Use the given optimizer to filter out bad strides')
	parser.add_argument('-o', '--output', metavar='FILE', help='Dump to the given file instead of the screen')
	parser.add_argument('--invert', action='store_true', default=False, help='Invert the --mod criteria')

	args = parser.parse_args()

	stride_optimizer = optimizer.getOptimizer(args.bad_strides) if args.bad_strides else None

	def check_mod(val):
		if args.invert:
			return int(val) % args.mod != 0
		else:
			return int(val) % args.mod == 0

	# filter the data
	dataset = [ val for val in data.load(args.file)
	                if (args.min == None or float(getattr(val, args.metric)) > args.min)
	                and (args.max == None or float(getattr(val, args.metric)) < args.max)
	                and (args.mod == None or check_mod(getattr(val, args.metric)))
	                and (args.max_error == None or float(val.time_std) / float(val.time) * 100 < args.max_error)
	                and (stride_optimizer == None or stride_optimizer.getStrideBadness(datatypes.getType(val.typename), int(val.stride_bytes)) == 0)
	          ]

	# dump
	if args.output:
		data.dump(args.output, dataset)
	else:
		for val in dataset:
			print val




