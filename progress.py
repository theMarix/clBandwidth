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

class DummyBar():
	"""A mockup of progressbar.ProgressBar that does nothing."""

	def __call__(self, iterable):
		"""Just pass through the iterable"""
		return iterable

def ProgressBar(dummy_only=False):
	""" Return a progress bar object as implemented by the ProgressBar module

	if dummyOnly is given return a object that does not actually produce any
	output"""

	if dummy_only:
		return DummyBar()
	else:
		import progressbar
		return progressbar.ProgressBar()
