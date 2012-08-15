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

from datatypes import *

def getOptimizer(name):
	""" Get the optimizer with the given name. """
	return _optimizers[name]


class Optimizer(object):
	""" A generic optimizer. Should not be used directly. """

	def __init__(self):
		# The maximum number of bytes to be added when recommending a stride
		self.MAX_ADD_STRIDE = 16 * 1024

	def recommendStride(self, datatype, elems):
		""" Recommend a stride for the given number of elements that will allow high performance.

		    Returns the recommended stride in elements. """

		if not isinstance(datatype, Struct):
			raise Exception('Stride can only be calculated for struct types')

		scalar_bytes = datatype.scalar.size

		# stride cannot be smaller than the number of elements
		for stride in range(elems, elems + self.MAX_ADD_STRIDE / scalar_bytes):
			if self.getStrideBadness(datatype, stride) == 0:
				return stride

		raise Exception('Failed to find a proper stride for {1} elements of type {0}'.format(datatype, elems))


class CypressOptimizer(Optimizer):
	""" An optimizers for Cypress GPUs """

	def __init__(self):
		super(CypressOptimizer, self).__init__()
		self.CRITICAL_STRIDE = 16 * 1024
		self.CRITICAL_STRIDE_RANGE = 768
		self.OPTIMUM_OFFSET = 256

	def getAlignmentBadness(self, offset):
		""" Calculate a badness value for the offset given in bytes """
		return 1 if (offset % self.OPTIMUM_OFFSET) else 0

	def getStrideBadness(self, datatype, stride):
		""" Get a badness value for a given stride in bytes. """
		if not isinstance(datatype, Struct):
			raise Exception('Stride badness can only be calculated for struct types')

		badness = self.getAlignmentBadness(stride)

		for i in range(1, datatype.elems):
			# check how close we are to the CRITICAL_STRIDE
			dist_to_critical = (i * stride) % self.CRITICAL_STRIDE
			if dist_to_critical >= self.CRITICAL_STRIDE / 2:
				dist_to_critical -= self.CRITICAL_STRIDE

			# check whether we are in the affected are of the critical stride
			if dist_to_critical >= -self.CRITICAL_STRIDE_RANGE and dist_to_critical <= self.CRITICAL_STRIDE_RANGE:
				badness += 1

		return badness


class CaymanOptimizer(CypressOptimizer):
	""" An optimizer for Cayman GPUs """

	def __init__(self):
		super(CaymanOptimizer, self).__init__()
		self.CRITICAL_STRIDE = 32 * 1024
		self.CRITICAL_STRIDE_RANGE = 768
		self.OPTIMUM_OFFSET = 256


class TahitiOptimizer(Optimizer):
	""" An optimizer for Tahiti GPUs """

	def __init__(self):
		super(TahitiOptimizer, self).__init__()
		self.OPTIMUM_OFFSET = 512

	def getAlignmentBadness(self, offset):
		""" Calculate a badness value for the offset given in bytes """
		return 1 if (offset % self.OPTIMUM_OFFSET) else 0

	def getStrideBadness(self, datatype, stride):
		""" Get a badness value for a given stride in bytes. """
		if not isinstance(datatype, Struct):
			raise Exception('Stride badness can only be calculated for struct types')

		badness = self.getAlignmentBadness(stride)

		# we don't know any fancy rules for the Tahiti GPUs, so just return the alignment value
		return badness

class NVIDIAOptimizer(Optimizer):
	""" An optimizer for Tahiti GPUs """

	def __init__(self):
		super(NVIDIAOptimizer, self).__init__()
		self.OPTIMUM_OFFSET = 256

	def getAlignmentBadness(self, offset):
		""" Calculate a badness value for the offset given in bytes """
		return 1 if (offset % self.OPTIMUM_OFFSET) else 0

	def getStrideBadness(self, datatype, stride):
		""" Get a badness value for a given stride in bytes. """
		if not isinstance(datatype, Struct):
			raise Exception('Stride badness can only be calculated for struct types')

		badness = self.getAlignmentBadness(stride)

		# we don't know any fancy rules for the Tahiti GPUs, so just return the alignment value
		return badness


_optimizers = {
	'Cypress': CypressOptimizer(),
	'Cayman': CaymanOptimizer(),
	'Tahiti': TahitiOptimizer(),
	'NVIDIA': NVIDIAOptimizer()
}
