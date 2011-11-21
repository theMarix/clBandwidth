OpenCL Bandwidth Measurements
=============================

This is a collection of OpenCL kernels that should be able to reproduce the global memory performance numbers given in the [AMD Accelerated Parallel Processing OpenCL™ Programming Guide](http://developer.amd.com/sdks/AMDAPPSDK/assets/AMD_Accelerated_Parallel_Processing_OpenCL_Programming_Guide.pdf) and NVIDIA's [OpenCL Best Practices Guide](http://developer.download.nvidia.com/compute/DevZone/docs/html/OpenCL/doc/OpenCL_Best_Practices_Guide.pdf). Those kernels are wrapped by python script handline all the boilerplate and the actual measurements.

Requirements
------------

 * [Python](http://www.python.org/)
 * [PyOpenCL](http://mathema.tician.de/software/pyopencl)
 * An OpenCL platform

Usage
-----

There are multiple scripts to choose from. For each script you will get additional invocation options by invoking it with ``--help``:

 * ``bandwidth.py`` - Compare the bandwidth of multiple kernels for a given memory size.
 * ```sweepMemSize.py`` - Check the performance of a single kernel over a certain memory size range.
