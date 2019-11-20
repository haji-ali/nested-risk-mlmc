This repository contains the C++ implementation for the numerical experiments in the
following two publications

[1] A. Haji-Ali, MB. Giles _"Multilevel nested simulation for efficient risk estimation"_,
     SIAM/ASA Journal on Uncertainty Quantification, 7(2), 497--525, 2019.
     
[2] A. Haji-Ali, MB. Giles _"Sub-sampling and other considerations for efficient risk estimation in large portfolios"_,
    preprint.

The numerical examples in these publications showcase using Multilevel Monte Carlo (MLMC) along
with adaptive sampling and sub-sampling to compute estimates of nested expectations that appear
in risk estimation of the form `E[H(X|Y)]`, where `H` is the Heaviside function. [1] has a
simple "model" problem with a known solution where `X` and `Y` are one-dimensional normal
random variables. [2] applies the same methodologies to a more complicated problem where `X` is
an aggregate of the losses of a large number of options in a financial portfolio and `Y` is the
multi-dimensional asset price at some short horizon.

# Usage
To compile the code, simply run

``` sh
make cxx
```

to build executables that use OpenMP for parallelization. Alternatively, run 

``` sh
make cuda
```

to build executables that use CUDA for parallelization. This was tested on CUDA 10 and requires
a device with at least 3.5 compute capabilities.

After compiling the code, you can run the test problem in [1] using

``` sh
./build/model_sampler
./build/model_sampler.cu
```

for OpenMP/CUDA, respectively. For the test problem in [2], run

``` sh
./build/portfolio_sampler
./build/portfolio_sampler.cu
```

# Files
### mlmc.hpp, mlmc\_test.cpp
Contains a class that implements MLMC for a general functional. A simple real-valued MLMC with
4 moments is defined using `typedef mlmc_t<real, 4> mlmc_real`.
The key function is this file `mlmc_real::run`. 

See `mlmc_test.cpp`, built with `make test`, for a simple example using this class.

### nested\_sampler.hpp
Contains the main logic for nested and adaptive sampling. The key classes here are
`inner_sampler_base` and `inner_sampler`. The first is the base class of the classes
`model::sampler` and `portfolio::sampler` and includes two random streams, one per-core and one
per-warp (the later being important for CUDA executables). On the other hand, the class
`inner_sampler<DYNAMIC, SAMPLER>` implements several functionalities on top of a some inner
sampler, `SAMPLER`, like `model::sampler` or `portfolio:sampler`. The parameter `DYNAMIC`
controls if dynamic parallelism is used in a CUDA implementation. The main function in this
header file is `sample<SAMPLER>` which samples a MLMC difference for a level `ell` using the
inner sampler `SAMPLER`.

### model\_sampler.cpp, portfolio\_sampler.cpp
These two files contain the main samplers for the model, in `model::sampler`, and portfolio, in
`portfolio::sampler`, problems. `portfolio::sampler` also implements random sub-sampling for
estimating the sum of losses.

These files also contain the `main` functions where the problem parameters can be modified to
produce all the results in [1] and [2]. Additionally, `portfolio_gen_data` in
`portfolio_sampler.cpp` generates the sample portfolio.

### par\_nvcc.hpp, par\_omp.hpp
Contains several definitions that make it easier to implement CUDA/OpenMP enabled code,
including `random_state` which interfaces CUDA random generators or
[Random123](http://www.thesalmons.org/john/random123/) generators, the latter being used for
OpenMP executables.

### gbm_approx.hpp
Contains an Euler-Maruyama/Milstein implementation for generating approximate samples of a
Geometric Brownian Motion.

### common.hpp, common.cpp
Contains several definitions that are common to all numerical experiments.

### table_output.hpp
A simple class, `table_t`, that outputs text tables with auto-expanding column widths.

