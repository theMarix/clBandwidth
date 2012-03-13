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

class Type:
	def __init__(self, name, size):
		self.name = name
		self.size = size

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

def getType(name):
	for cand in types:
		if cand.name == name:
			return cand
	raise NameError('{0} is not a valid type name'.format(name))

# module initialization

# generate vector types
__vectorElems = [2, 3, 4, 8, 16]
for scalar in types[:]:
	for n in __vectorElems:
		types.append(Type('{0}{1}'.format(scalar.name, n), n * scalar.size))
