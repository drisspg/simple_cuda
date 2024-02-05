# Simple Cuda

![surfing](https://github.com/drisspg/simple_cuda/assets/32754868/4f8d27e2-dcc2-40a3-9aff-a814340878e4)


Small little project to track exercises as I go through the *Programming Massively Parallel Processors* book.

I also want to get some more familiarity with using Cmake from scratch, I wouldn't use any of this code its mostly for learning.



## Structure

Code goes into 2 places. I plan on figuring out what utilities and generals things I want to that are agnostic to individual examples. These will go into src/.
They will go into one of two places, currently this header only so I plan on putting stuff src/include. But I want to learn more about libraries so I might put some stuff in src/ and create a library that the examples will depend on.

The examples will go into examples/. I plan on using the CMakeLists.txt in the root to build all the examples. By default cmake will build every example, which is fine for now. I might add a way to build a specific example later.



## Building

```Shell
# In the root directory
mkdir build && cd build

# Configure the build
cmake -DCMAKE_BUILD_TYPE=RelWithDebugInfo -G Ninja ..

# Build the examples
ninja

```

There are some options that can be set when configuring the build. These are set with the -D flag when running cmake.
For example to build in debug:

```Shell
cmake -DCMAKE_BUILD_TYPE=Debug ..
```

Release

```Shell
cmake -DCMAKE_BUILD_TYPE=Release ..
```

etc..


This will build all the examples and put the executables in build/bin.

### Example E2E run for the tiled_mm example in chapter5
```Shell

cd build

cmake -DCMAKE_BUILD_TYPE=RelWithDebugInfo -G Ninja ..

./bin/tiled_mm

# Profie with NCU
ncu -k "regex:Matrix" ./bin/tiled_mm  
```

### Dependencies 
- Cmake
- Ninja
- Cuda
- c++=std20 compilier
