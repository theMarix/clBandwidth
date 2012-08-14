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

import re

class Type(object):
	def __init__(self, name, size):
		self.name = name
		self.size = size
	def __repr__(self):
		return 'Type({0}, {1})'.format(self.name, self.size)
	def __str__(self):
		return self.name

class Struct(Type):
	def __init__(self, scalar, elems):
		super(Struct, self).__init__('struct_of_{0}_{1}'.format(elems, scalar.name), scalar.size * elems)
		self.scalar = scalar
		self.elems = int(elems, 10)
	def __repr__(self):
		return 'Struct({0}, {1})'.format(repr(self.scalar), self.size)

types = [
	Type('char', 1),
	Type('uchar', 1),
	Type('short', 2),
	Type('ushort', 2),
	Type('int', 4),
	Type('uint', 4),
	Type('long', 8),
	Type('ulong', 8),
	Type('half', 2),
	Type('float', 4),
	Type('double', 8)
];

def getScalarType(name):
	for cand in types:
		if cand.name == name:
			return cand
	raise NameError('{0} is not a valid type name'.format(name))

def getType(name):
	struct_match = re.match(r'struct_of_(\d+)_(\w*)', name)
	if struct_match:
		return Struct(getScalarType(struct_match.group(2)), struct_match.group(1))
	else:
		return getScalarType(name)

# module initialization

# generate vector types
__vectorElems = [2, 3, 4, 8, 16]
for scalar in types[:]:
	for n in __vectorElems:
		types.append(Type('{0}{1}'.format(scalar.name, n), n * scalar.size))
