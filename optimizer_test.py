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

from datatypes import *
import optimizer

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Test an optimizer')
	parser.add_argument('optimizer', help='The optimizer to use')
	parser.add_argument('type', default='float', help='The basic scalar type to use')
	parser.add_argument('struct', type=int, help='Use a struct of N elems of the basic scalar type')
	parser.add_argument('elems', type=int, help='The number of structs to be stored')

	args = parser.parse_args()

	scalar_type = getType(args.type)
	data_type = Struct(scalar_type, args.struct)

	print optimizer.getOptimizer(args.optimizer).recommendStride(data_type, args.elems)

