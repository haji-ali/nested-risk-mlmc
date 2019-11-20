#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <assert.h>
#include <algorithm>
#include <iomanip>
#include <memory.h>
#include <limits>

#include "common.hpp"
#include "mlmc.hpp"
#include "gbm_approx.hpp"

int main(// int argc, char **argv
    ){
    mlmc_real::mlmc_input_t input;
    std::normal_distribution<double> distribution(0.0, 1.0);
    std::default_random_engine gen;
    auto randn = [&gen, &distribution] () { return distribution(gen); };
    input.M0 = 100;

    input.sample_level = [&, randn](uint32_t ell, uint64_t M){
                             mlmc_real::mc_diff_output_t mc_out;
                             uint32_t N = 2 * (1 << ell);
                             double mu=1, sig=0.1, S0=1, K=1;
                             std::clock_t start = std::clock();
                             for (uint64_t m=0;m<M;m++){
                                 double S[2] = {S0, S0};
                                 simulate_stock_dt<true >(randn, S, N, mu, sig, 1, 1, false);
                                 double f = std::max(S[0]-K, 0.);
                                 double d = f-std::max(S[1]-K, 0.);
                                 for (uint32_t j=0;j<mlmc_real::moments;j++){
                                     mc_out.sums_fine[j] += std::pow(f, j+1);
                                     mc_out.sums_diff[j] += std::pow(d, j+1);
                                 }
                             }
                             mc_out.M = M;
                             mc_out.work = N*M;
                             mc_out.time = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
                             return mc_out;
                         };
    input.verbose = true;

    mlmc_real::run(input);
    return 0;
}
