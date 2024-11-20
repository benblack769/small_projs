Personal repository of useful C++ header micro-libraries:

* `Array2d.hpp`: A clean and dynamic 2d array implementation using a single C-ordered buffer
* `backwards_iter.h`: A wrapper to turn a bi-directional iterable into a forward iterable iterating in reverse. I.e. swapping `rbegin()` for `begin()`, etc
* `c++pythonGenerators.h`: A proof of concept of implementing asynchronous python generator style coroutine logic using c++ threads
* `construct_help.hpp`: Simple helpers to more easily construct vectors
* `for_each_extend.hpp`: Zipped and container-centric for each helpers
* `intrinsic_help.h`: Concise and simple c++ convenience wrappers for Intel floating point SIMD operations.
* `opencl_executor.h`: A header-only OpenCL 1.2 RAII wrapper
* `point.hpp`: A 2d point (x,y) class with various operators and math overloads
* `protect_global.hpp`: A system for capture and deferred resetting of global variables for mock test frameworks
* `range_array.hpp`: A 2d range iterator container. Depends on `point.hpp`
* `two_d_array.h`:  more feature rich dynamic 2d array implementation including dynamic resizing features
* `unpair.hpp`: A corollary of the standard library's tuple unpacking feature for std::pair
* `zip_for.hpp`: A zipped for each implementation