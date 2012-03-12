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
	Type('float', 4),
	Type('float2', 8),
	Type('float4', 16),
	Type('int', 4),
	Type('int2', 8),
	Type('int4', 16)
];

def getType(name):
	for cand in types:
		if cand.name == name:
			return cand
	raise NameError('{0} is not a valid type name'.format(name))
