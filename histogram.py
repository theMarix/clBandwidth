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
import matplotlib.pyplot as plt
import numpy as np

import data

if __name__ == '__main__':
	# parse the user request
	parser = argparse.ArgumentParser(description='Plot memory bandwidth')
	parser.add_argument('files', metavar='FILE', nargs='+')
	parser.add_argument('--labels', metavar='LABEL', nargs='*', help='Labels to mark the line from each input file.')
	parser.add_argument('--metric', default='bandwidth', help='Value to use for the histogram.')
	parser.add_argument('--xlabel', help='Label to use for the xaxis. Otherwise the value of --xaxis is used')
	parser.add_argument('--scale', type=float, default=1, help='Rescale by the given value')
	parser.add_argument('--title', help='Title to add to the plot')
	parser.add_argument('--output', metavar='FILE', help='Plot to the given file instead of the screen')
	parser.add_argument('--bins', type=int, help='How many bins to plot') # TODO maybe replace by bin-size

	args = parser.parse_args()

	if args.labels and len(args.files) != len(args.labels):
		print 'Please specify exactly one label per file or none at all'
	if not args.labels:
		args.labels = args.files

	if not args.xlabel:
		args.xlabel = args.metric

	if args.labels:
		labels = args.labels
	else:
		labels = args.files

	# read the data
	datasets = [data.load(filename) for filename in args.files];

	# create the plot
	fig = plt.figure(figsize=(10,4))

	vals = [[float(getattr(val, args.metric)) for val in dataset] for dataset in datasets]
	plt.hist(vals, label=args.labels)

	if args.title:
		plt.title(args.title)
	#plt.ylabel('Bandwidth GB/s')
	plt.xlabel(args.xlabel)

	plt.legend(loc=0)

	if args.output:
		plt.savefig(args.output)
	else:
		plt.show()