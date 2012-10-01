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

import csv
from collections import namedtuple

# Result of kernel invocation for a given set of parameters. stride and offset in elements, time is in nanos, bandwidth in GB/s
# input_bytes is the actually used part of the input buffer in bytes
# output_bytes is the actually used part of the input buffer in bytes
# note that if a kernel performs multiple passes of the data input_bytes + output_bytes != bytes_transferred
DataPoint = namedtuple('DataPoint', 'typename global_threads local_threads stride stride_bytes offset offset_bytes input_bytes output_bytes bytes_transferred time time_std bandwidth')

def dump(filename, data):
	writer = csv.writer(open(filename, 'wb'), quoting=csv.QUOTE_MINIMAL)
	writer.writerow(['clBandwidth data file', 'file version', 3])
	writer.writerow(data[0]._fields)
	writer.writerows(data)

def load(filename):
	reader = csv.reader(open(filename, 'rb'))
	meta = reader.next()
	if len(meta) < 3 or meta[0] != 'clBandwidth data file' or meta[1] != 'file version':
		if meta[0] == 'typename':
			# old data-file
			# headers are already skipped
			data_ver = 1
			pass
		else:
			raise Exception('File does not seem to be a valid datafile')
	else:
		data_ver = int(meta[2])
		# data format >= version 2
		reader.next() # skip headers

	if data_ver < 3:
		print 'WARNING: Old data file - errors are variance of distribution, not standard error of mean.'

	return map(DataPoint._make, reader)
