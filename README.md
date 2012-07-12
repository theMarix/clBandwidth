OpenCL Bandwidth Measurements
=============================

**DISCLAIMER: I am currently rewriting this tool. Therefore this README represents future state and features are broken all the time.**

This is a collection of OpenCL kernels that should be able to reproduce the global memory performance numbers given in the [AMD Accelerated Parallel Processing OpenCL™ Programming Guide](http://developer.amd.com/sdks/AMDAPPSDK/assets/AMD_Accelerated_Parallel_Processing_OpenCL_Programming_Guide.pdf) and NVIDIA's [OpenCL Best Practices Guide](http://developer.download.nvidia.com/compute/DevZone/docs/html/OpenCL/doc/OpenCL_Best_Practices_Guide.pdf). Those kernels are wrapped by python script which handles all the boilerplate and the actual measurements.

In addition to reproducing the numbers from the programming guide this also allows to analyze the performance of different memory access patterns – enabling performance optimization of bandwidth limited code. Therefore the script allows to define structs for which it will dynamically generate kernels and perform measurements. Allowing to quickly test different memory layouts.

Requirements
------------

 * [Python](http://www.python.org/)
 * [PyOpenCL](http://mathema.tician.de/software/pyopencl)
 * An OpenCL platform

Usage
-----

There are multiple scripts to choose from. For each script you will get additional invocation options by invoking it with ``--help``:

 * ``bandwidth.py`` -- Compare the achievable bandwidth of multiple data types for a given memory size.
 * ``memsize.py`` -- Check the effect of memory size on the performance of a data-type.
 * ``offset.py`` -- Check the effect of different offsets into the buffer on performance of a data-type.
 * ``stride.py`` -- Check the effect of different strides for SOA datatypes
 * ``plot.py`` -- Plot data from the CSV files dumped by the measurement scripts

License
-------

ClBandwidth is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ClBandwidth is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with clBandwidth.  If not, see [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/).

&copy; 2011--2012 Matthias Bach <bach@compeng.uni-frankfurt.de>
