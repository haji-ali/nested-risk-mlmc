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
#include "nested_sampler.hpp"
#include "mlmc.hpp"

namespace model {
    enum class PhiType {
        Indicator=0,
        Max=1,
    };

    struct data {
        PAR_BOTH data(){}

        PAR_BOTH data(PhiType phi_type, real_t tau, real_t max_loss){
            this->stau = std::sqrt(tau);
            this->s1tau = std::sqrt(1-tau);
            this->max_loss = max_loss;
            this->phi_type = phi_type;
        }

        PAR_BOTH uint32_t phi_size() const {
            return 1;
        }

        real_t stau;  // Sqrt of tau
        real_t s1tau;  // Sqrt of 1-tau
        real_t max_loss;
        PhiType phi_type;
    };
    PAR_DEVICE_CONST data gc_data;

    class sampler : public inner_sampler_base {
    private:
        real_t Y;
    public:
#ifdef DEBUG
        PAR_DEVICE void* debug(){ return 0; }
#endif
        PAR_BOTH static uint32_t GetWorkspaceSize(const data&) {
            return 0;
        }

        PAR_DEVICE sampler(void*) : Y(NAN) {
            // no need for workspace
        }

        PAR_DEVICE void sample_outer(uint64_t, work_t&){
            Y = randn_sync();
        }
        PAR_DEVICE void copy_outer(const sampler& rhs){
            Y = rhs.Y;
        }

        inline PAR_DEVICE real_t init_inner_sampler(uint64_t&, work_t&){ return -gc_data.max_loss;}   // Nothing needed here

        PAR_DEVICE real_t sample_inner(uint64_t, uint64_t, work_t &total_work){
            total_work++;

            real_t X = randn();
            real_t Y2 = randn();
            real_t r = POW2(gc_data.stau * Y + gc_data.s1tau * X);
#ifdef MODEL_ANTITHETIC
            r -= 0.5f * POW2(gc_data.stau * Y2 + gc_data.s1tau * X);
            r -= 0.5f * POW2(-gc_data.stau * Y2 + gc_data.s1tau * X);
#else
            r -= POW2(gc_data.stau * Y2 + gc_data.s1tau * X);
#endif
            return r;
        }

        static PAR_DEVICE void add_phi(real_t inner_avg, real_t coeff, real_t *phi){
            switch (gc_data.phi_type)
                {
                case PhiType::Max:
                    *phi += coeff * MAX(inner_avg, 0);
                    break;
                case PhiType::Indicator:
                default:
                    *phi += coeff * (inner_avg >= 0.f);
                }
        }
        };

    extern "C" void model_update_data(const model::data *data) {
        PAR_SET_CONST(model::gc_data, *data);
        PAR_SET_CONST(gc_phi_size, data->phi_size());
    }

    extern "C" void model_init(const model::data *data,
                               uint64_t seed){
        model_update_data(data);
        init_samplers<model::sampler>(
            seed, model::sampler::GetWorkspaceSize(*data));
    }

    extern "C" void model_sample(uint32_t ell, uint64_t M,
                                 sample_output *out){
        out->M = M;
        sample<model::sampler>(ell, 0, &out->M,
                                     out->fsums,
                                     out->diffsums,
                                     &out->total_work,
                                     out->inner_stats);
    }
}


#ifdef INCLUDE_MAIN_MLMC
int main(// int argc, char **argv
    ){
    int cores = 1;
#ifdef __NVCC__
    cores = 2048 * c_warp_size;   // 65536
    std::cout << "CUDA Enabled" << std::endl;
#elif defined(_OPENMP)
    cores = omp_get_max_threads();
    std::cout << "OpenMP Enabled" << std::endl;
#else
    std::cout << "No parallelism" << std::endl;
#endif

    std::cout << "Cores:" << cores << std::endl;

    mlmc_real::mlmc_input_t input;

    // Input parameters
    real_t tau = 0.02;
    real_t eta = 0.025;          // Exact value to compute
    uint64_t calc_seed = 0;       // Seed used to sample options
    uint32_t N0 = 32;          // Minimum number of inner samples
    uint32_t beta = 4;        // Refinement factor for inner sampler
    real_t adaptive_r = 1.5;   // r-value for adaptive algorithm of inner sampler, 0 for non-adaptive
    bool antithetic = true;    // Use antithetic estimator for the inner sampler
    real_t max_loss = tau*(POW2(norm_cdf_inv(eta/2))-1);
    input.M0 = 1024;
    input.verbose = true;
    input.targetTOL = 1e-3;

    nested_sampler_params nested_param(mlmc_real::moments, mlmc_real::moments,
                                       N0, beta, adaptive_r, antithetic);
    init_nestedmc_sampler(cores, &nested_param);

    model::data mod_data(model::PhiType::Indicator, 0.02, max_loss);
    model::model_init(&mod_data, calc_seed);

    uint32_t phi_size = mod_data.phi_size();
    assert(phi_size==1);  // Only real value are supported
    input.sample_level = [&, phi_size] (uint32_t ell, uint64_t M) {
                             mlmc_real::mc_diff_output_t mc_out;
                             std::clock_t start = std::clock();
                             sample_output out(mlmc_real::moments, mlmc_real::moments,
                                               phi_size);
                             model::model_sample(ell, M, &out);
                             std::copy(out.fsums, out.fsums+mlmc_real::moments,
                                       mc_out.sums_fine);
                             std::copy(out.diffsums, out.diffsums+mlmc_real::moments,
                                 mc_out.sums_diff);
                             mc_out.work = out.total_work;
                             mc_out.M = out.M;
                             mc_out.time = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
                             return mc_out;
                         };
        
    mlmc_real::run(input);
    cleanup_nestedmc_sampler();
    return 0;
}
#endif

