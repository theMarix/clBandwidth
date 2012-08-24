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

MARKERS = ['o'] * 7 + ['D'] * 7 + ['p'] * 7

if __name__ == '__main__':
	# parse the user request
	parser = argparse.ArgumentParser(description='Plot memory bandwidth')
	parser.add_argument('files', metavar='FILE', nargs='+')
	parser.add_argument('--labels', metavar='LABEL', nargs='*', help='Labels to mark the line from each input file.')
	parser.add_argument('--xaxis', default='bytes_transferred', help='Value to use for the x axis. Should be one of bytes_transferred (default), stride or offset')
	parser.add_argument('--xlabel', help='Label to use for the xaxis. Otherwise the value of --xaxis is used')
	parser.add_argument('--xscale', type=float, default=1, help='Rescale x axis by the given value')
	parser.add_argument('--title', help='Title to add to the plot')
	parser.add_argument('--output', metavar='FILE', help='Plot to the given file instead of the screen')
	parser.add_argument('--plot-errors', default=False, action='store_true', help='Add errorbars to the plot.')

	args = parser.parse_args()

	if args.labels and len(args.files) != len(args.labels):
		print 'Please specify exactly one label per file or none at all'
	if not args.labels:
		args.labels = args.files

	if not args.xlabel:
		args.xlabel = args.xaxis

	if args.labels:
		labels = args.labels
	else:
		labels = args.files

	# read the data
	datasets = [data.load(filename) for filename in args.files];

	# create the plot
	fig = plt.figure(figsize=(10,4))
	for dataset, label, marker in zip(datasets, args.labels, MARKERS):
		xvals, yvals, errs = zip(*[(int(getattr(val, args.xaxis)) / args.xscale, float(val.bandwidth), float(val.bandwidth) * float(val.time_std) / float(val.time)) for val in dataset])
		if args.plot_errors:
			plt.errorbar(xvals, yvals, yerr=errs, marker=marker, label=label)
		else:
			plt.plot(xvals, yvals, marker, label=label)
	plt.ylim(ymin=0)

	if args.title:
		plt.title(args.title)
	plt.ylabel('Bandwidth GB/s')
	plt.xlabel(args.xlabel)

	extra_artists = []

	if len(args.labels) > 1:
		leg = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
		extra_artists.append(leg)

	if args.output:
		plt.savefig(args.output, bbox_inches='tight', bbox_extra_artists=extra_artists)
	else:
		plt.show()
