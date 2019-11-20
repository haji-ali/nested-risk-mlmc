#ifndef __NESTED_SAMPLERS_HPP__
#define __NESTED_SAMPLERS_HPP__

const uint32_t c_max_phi_size =  4;
const uint32_t c_max_moments =  4;
const real_t c_conf2 = 9.;
const bool c_optimize_for_diff = false;     // 'true' produces correct difference computation is correct but not fine computation
const bool c_extrapolate_adaptive = false; 

template<bool DYNAMIC, typename SAMPLER>
struct sample_inner_template {};  // Defined later for specific values of DYNAMIC

//// DATA for nested sampler
struct nested_sampler_params {
    nested_sampler_params() {}
    nested_sampler_params(uint32_t _fine_moments, uint32_t _diff_moments,
                          uint32_t _N0, uint32_t _beta,
                          real_t _adaptive_r,
                          bool _antithetic) : N0(_N0), beta(_beta), 
                                              antithetic(_antithetic),
                                              fine_moments(_fine_moments),
                                              diff_moments(_diff_moments),
                                              adaptive_r(_adaptive_r) {
    }

    uint32_t N0;
    uint32_t beta;
    bool antithetic;
    uint32_t fine_moments;
    uint32_t diff_moments;
    real_t adaptive_r;

    PAR_BOTH bool adaptive() { return adaptive_r>1 && adaptive_r<2; }
};

class inner_sampler_base {
public:
#ifdef EXTRA_INNER_DIAGNOSITCS
    uint64_t randn_count;
    uint64_t rand_count;
    PAR_BOTH inner_sampler_base(){
        reset_diagnostics();
    }
    PAR_BOTH void reset_diagnostics(){
        randn_count = 0;
        rand_count = 0;
    }
#endif /// EXTRA_INNER_DIAGNOSITCS
    RANDSTATE rand_state;
    RANDSTATE rand_state_sync;

    // This fields are used for saving the first and second sums
    // and number of inner samples. This is when using dynamic
    // parallelism for CUDA.
    void* dyn_block;   // should be size dyn_block_size
    static const int dyn_block_size = sizeof(sum_t) * 2 + sizeof(work_t) + sizeof(uint64_t);// + sizeof(RANDSTATE);

    inline PAR_DEVICE real_t randn(RANDSTATE *state){
#ifdef EXTRA_INNER_DIAGNOSITCS
        randn_count++;
#endif
        return state->randn();
    }

    inline PAR_DEVICE real_t randu(RANDSTATE *state){
#ifdef EXTRA_INNER_DIAGNOSITCS
        rand_count++;
#endif
        return state->randu();
    }

    inline PAR_DEVICE real_t randn_sync(){
        return randn(&rand_state_sync);
    }
    inline PAR_DEVICE real_t randn(){
        return randn(&rand_state);
    }

    inline PAR_DEVICE real_t randu_sync(){
        return randu(&rand_state_sync);
    }
    inline PAR_DEVICE real_t randu(){
        return randu(&rand_state);
    }
};

PAR_DEVICE inline void add_moments(real_t *phi, size_t /*phi_size*/,
                                   sum_t* out, uint32_t moments_count){
    for (uint32_t i=0;i<gc_phi_size;i++){
        real_t p = 1.f;
        for (uint32_t j=0;j<moments_count;j++){
            p *= phi[i];
            *(out++) += p;
        }
    }
}
PAR_DEVICE inline void add_moments(real_t *phi, sum_t* out,
                                   uint32_t moments_count){
    add_moments(phi, gc_phi_size, out, moments_count);
}

template<bool DYNAMIC, class SAMPLER>
class inner_sampler {
public:
    static PAR_DEVICE void sample_inner_sum(SAMPLER *sampler, uint32_t dyn_inner,
                                            uint64_t& count, real_t& start,
                                            sum_t& out_s1, work_t& out_total_work){
        start = sampler[0].init_inner_sampler(count, out_total_work);
        return sample_inner_template<DYNAMIC, SAMPLER>::call(
            sampler, dyn_inner, count, out_s1, NULL, out_total_work);
    }
    static PAR_DEVICE real_t sample_inner_mean(SAMPLER *sampler, uint32_t dyn_inner,
                                             uint64_t count, work_t& out_total_work){
        real_t start = sampler[0].init_inner_sampler(count, out_total_work);
        sum_t s1=0;
        sample_inner_template<DYNAMIC, SAMPLER>::call
            (sampler, dyn_inner, count, s1, NULL, out_total_work);
        return start + s1/count;
    }
    static PAR_DEVICE real_t sample_inner_delta2(SAMPLER *sampler, uint32_t dyn_inner, uint64_t count,
                                               work_t& out_total_work) {
        real_t start = sampler[0].init_inner_sampler(count, out_total_work);
        sum_t s1=0,s2=0;
        sample_inner_template<DYNAMIC, SAMPLER>::call
            (sampler, dyn_inner, count, s1, &s2, out_total_work);
        real_t out_var = static_cast<real_t>(s2 / (count-1) - POW2(s1) / (count*(count-1)));
        real_t dist =  start + static_cast<real_t>(s1/count);
        return POW2(dist) / out_var;
    }

    static PAR_DEVICE void sample_inner(SAMPLER &sampler,
                                        uint64_t& count,
                                        sum_t& out_s1,
                                        sum_t* out_s2,
                                        work_t& out_total_work) {
        uint32_t slave_samplers = c_warp_size;
        FREQ_ASSERT(count % slave_samplers == 0);
        OMP_PRAGMA(parallel) {
            sum_t s1=0, s2=0;
            work_t total_work=0.;
            OMP_PRAGMA(for) for (uint64_t i=0;i<count/slave_samplers;i++){
                real_t d = sampler.sample_inner(i, count, total_work);
                s1 += d;
                if (out_s2)
                    s2 += d*d;
            }
            OMP_PRAGMA(critical) {
#ifdef __NVCC__
                out_s1 += warpSum_All(s1);
                if (out_s2)
                    *out_s2 += warpSum_All(s2);
                out_total_work += warpSum_All(total_work);
#else // __NVCC__
                out_s1 += s1;
                if (out_s2)
                    *out_s2 += s2;
                out_total_work += total_work;
#endif // __NVCC__
            }
        }
    }


    static PAR_DEVICE void sample_phi(SAMPLER *sampler, uint32_t dyn_inner, uint64_t count,
                                      work_t& total_work, real_t *out) {
        for (uint32_t i=0;i<gc_phi_size;i++)
            out[i] = 0;
        SAMPLER::add_phi(sample_inner_mean(sampler, dyn_inner, count, total_work), 1.f, out);
    }

    static PAR_DEVICE uint64_t sample_phi_antithetic(SAMPLER *sampler,
                                                   uint32_t dyn_inner,
                                                   uint64_t count_f,
                                                   uint64_t count_c,
                                                   work_t& total_work,
                                                   real_t *out_f,
                                                   real_t *out_c) {
        for (uint32_t i=0;i<gc_phi_size;i++)
            out_f[i] = out_c[i] = 0.f;

        if (c_optimize_for_diff and count_f == count_c){
            // result is 0.
            return 0;
        }
        uint64_t count_big, count_small;
        real_t *out_big, *out_small;
        if (count_f > count_c){
            out_big = out_f;
            out_small = out_c;
            count_big = count_f;
            count_small = count_c;
        }
        else{
            out_big = out_c;
            out_small = out_f;
            count_big = count_c;
            count_small = count_f;
        }
        FREQ_ASSERT(count_big % count_small == 0);
        uint64_t groups_small = count_big / count_small;
        real_t coeff_small = 1.f/groups_small;
        real_t start_small = 0;
        sum_t s1_big=0;
        for (uint64_t i=0;i<groups_small;i++) {
            sum_t s1_small = 0;
            sample_inner_sum(sampler, dyn_inner, count_small, start_small, s1_small, total_work);
            real_t mean_small = static_cast<real_t>(start_small + s1_small/count_small);
            SAMPLER::add_phi(mean_small, coeff_small, out_small);
            s1_big += s1_small;
        }
        // This assumes that start_small does not change and
        // is the same as start_big (start does not depend on N)
        real_t mean_big = static_cast<real_t>(start_small + s1_big/count_big);
        SAMPLER::add_phi(mean_big, 1.f, out_big);
        return count_big;
    }


    template<bool WITH_STATS>
    class get_samples_count {
        real_t tmp;
        uint64_t base_count;
        uint64_t max_count;
    public:
        PAR_DEVICE get_samples_count(uint32_t ell){
            // assume gc_sampler_data.beta is a power of 2
            const uint64_t N0 = gc_sampler_data.N0;
            max_count = N0 * uint_pow(static_cast<uint64_t>(gc_sampler_data.beta),
                                      static_cast<uint64_t>(ell));
            if (gc_sampler_data.adaptive()){
                base_count = N0 * uint_pow(static_cast<uint64_t>(gc_sampler_data.beta >> 1),
                                           static_cast<uint64_t>(ell));
                tmp = max_count * std::pow(c_conf2/max_count, gc_sampler_data.adaptive_r/2);
            }
            else{
                base_count = max_count;
                tmp=0;
            }
        }
        PAR_DEVICE uint64_t operator () (SAMPLER* sampler, uint32_t dyn_inner,
                                       work_t &total_work,
                                       uint64_t* inner_stats) const {
            uint64_t cur_count = max_count;
            cur_count = base_count;
            const real_t r = gc_sampler_data.adaptive_r;
            while (true) {
                if (max_count <= 2*cur_count) {
                    cur_count = max_count;
                    break; // Not worth going through the adaptive algorithm
                }
                inner_stats[0] += cur_count;
                real_t delta2 = sample_inner_delta2(sampler, dyn_inner, cur_count, total_work);
                real_t new_count = std::pow(delta2, -r/2) * tmp;
                if (new_count <= cur_count){
                    // Expected number of samples is less than what we already
                    // did
                    break;
                }
                cur_count *= 2;
                if (c_extrapolate_adaptive and new_count <= cur_count){
                    // Not worth doing another step of the adaptive algorithm
                    break;
                }
            }
#ifdef EXTRA_INNER_DIAGNOSITCS
            if (WITH_STATS){
                inner_stats[1] += 1;
                inner_stats[2] += (cur_count == max_count);
                inner_stats[3] += (cur_count == base_count);
            }
#endif
            return cur_count;
        }
    };
};


#ifdef  __NVCC__
template<class SAMPLER>
PAR_GLOBAL void kernel_sample_inner(SAMPLER *sampler,
                                    uint32_t dyn_inner,
                                    uint64_t totalN,
                                    uint64_t *out_totalN,
                                    sum_t* out_s1,
                                    sum_t* out_s2,
                                    work_t* out_total_work) {
    uint64_t thread_count = blockDim.x*gridDim.x;
    // Since we are using warp, then we should multiply by c_warp_size
    // since sample_inner_template will then divide by c_warp_size and then
    // accumulate the result from the whole warp
    uint64_t N = c_warp_size * totalN / thread_count;  // N per thread
    uint32_t core = PAR_CORE_IDX;
    FREQ_ASSERT(core < dyn_inner && N >= c_warp_size);

    // Copy to local memory for efficiency
    SAMPLER lsampler = sampler[core];

    lsampler.copy_outer(sampler[0]);  // Copy outer from master sampler

    sum_t ls1=0, ls2=0; work_t ltotal_work=0;
    inner_sampler<true, SAMPLER>::sample_inner(lsampler, N, ls1, out_s2?&ls2:NULL, ltotal_work);

    // Copy back ...
    sampler[core] = lsampler;
    // NOTE: This won't overwrite the dyn_block since the assignment will only copy the pointer.

    if (!PAR_IS_WARP_MASTER)
        return;   // Only accumulate in the masters
    atomicAdd(out_s1, ls1);
    if (out_s2)
        atomicAdd(out_s2, ls2);
    atomicAdd(out_total_work, ltotal_work);
    atomicAdd(out_totalN, N);
}

template<typename SAMPLER>
struct sample_inner_template<true, SAMPLER> {
    static PAR_DEVICE void call(SAMPLER *sampler,
                                uint32_t dyn_inner,
                                uint64_t& count,
                                sum_t& out_s1,
                                sum_t* out_s2,
                                work_t& out_total_work){
        // The sampler should be on the heap
        FREQ_ASSERT(count >= c_warp_size);
        FREQ_ASSERT(count >= dyn_inner && count % dyn_inner == 0);
        uint32_t inner_cores = dyn_inner;
        uint32_t blocks = inner_cores / c_warp_size;

        sum_t* p_s1 = reinterpret_cast<sum_t*>(sampler[0].dyn_block);
        sum_t* p_s2 = p_s1 + 1;
        work_t* p_total_work = reinterpret_cast<work_t*>(p_s2 + 1);
        uint64_t* p_count = reinterpret_cast<uint64_t*>(p_total_work + 1);

        *p_s1 = *p_s2 = *p_total_work = 0;
        *p_count = 0;

        kernel_sample_inner<SAMPLER>
            <<<blocks, c_warp_size>>>(sampler, dyn_inner, count, p_count,
                                      p_s1,
                                      out_s2?p_s2:NULL,
                                      p_total_work);
        cudaDeviceSynchronize();   // Need to wait for the child grid to finish

        out_s1 += *p_s1;
        if (out_s2)
            *out_s2 += *p_s2;
        out_total_work += *p_total_work;
        FREQ_ASSERT(*p_count == count);
    }
};
#endif

template<typename SAMPLER>
struct sample_inner_template<false, SAMPLER> {
    static PAR_DEVICE void call(SAMPLER *sampler,
                                uint32_t dyn_inner,
                                uint64_t& count,
                                sum_t& out_s1,
                                sum_t* out_s2,
                                work_t& out_total_work){
        FREQ_ASSERT(dyn_inner == 1);
        inner_sampler<true, SAMPLER>::sample_inner(sampler[0], count, out_s1,
                                                   out_s2, out_total_work);
    }
};


template<bool DYNAMIC, typename SAMPLER>
PAR_GLOBAL void kernel_sample(SAMPLER* base_samplers,
                              uint32_t cores,
                              uint32_t dyn_inner,  // Number of threads used to simulate inner expectation
                              uint32_t ell, uint64_t totalM,
                              sum_t *out_f_sums,
                              sum_t *out_diff_sums,
                              work_t *out_total_work,
                              uint64_t *out_inner_stats) {
    OMP_PRAGMA(parallel) {
        work_t total_work = 0.;
        uint64_t inner_stats[c_inner_stats_count] = {0};
        sum_t f_sums[c_max_moments * c_max_phi_size] = {0};
        sum_t diff_sums[c_max_moments * c_max_phi_size] = {0};
        const uint32_t core = PAR_CORE_IDX;
        FREQ_ASSERT(core < cores);
        real_t f_phi[c_max_phi_size];
        real_t c_phi[c_max_phi_size];
        typedef inner_sampler<DYNAMIC, SAMPLER> inner;
        SAMPLER* sampler = base_samplers + core*dyn_inner;
        SAMPLER lsampler = *sampler;
        uint64_t f_samples_count, c_samples_count;
        const bool fine_only = (ell == 0) || (out_diff_sums == NULL);
        const typename inner::template get_samples_count<true>
            samples_count_ell(ell);
        const typename inner::template get_samples_count<false>
            samples_count_ell_1(fine_only?0:(ell-1));

#ifdef __NVCC__
        const bool warp_outer = !DYNAMIC;  // If dynamic, then there won't be any warp divergence
        // Now, we need to figure out how many samples we will be doing
        const uint32_t threads_count =  blockDim.x*gridDim.x;
        const uint32_t M_threads_count = threads_count / (warp_outer?c_warp_size:1);
        const uint64_t M = totalM / M_threads_count  
            + ((core/(warp_outer?c_warp_size:1)) < (totalM % M_threads_count));
#else
        uint64_t M = totalM;
#endif
        if (not DYNAMIC) {
            // Use local instance for more efficient access,
            // if dynamic keep in heap so we can pass the sampler to child kernels
            sampler = &lsampler;
        }
#ifdef EXTRA_INNER_DIAGNOSITCS
        for (uint32_t i=0;i<dyn_inner;i++)
            sampler[i].reset_diagnostics();
#endif
        // Generate single sample
        OMP_PRAGMA(for) for (uint64_t m=0;m<M;m++){
            // Send to all other wraps
            sampler[0].sample_outer(ell, total_work);
            if (DYNAMIC) {
                // Need to generate the outer samples in the whole first warp to avoid
                // warp divergence because of different inner rand_state_sync.
                for (uint32_t i=1;i<c_warp_size;i++)
                    sampler[i].sample_outer(ell, total_work);
           }

            f_samples_count = samples_count_ell(sampler, dyn_inner, total_work, inner_stats);
            if (fine_only) {
                inner_stats[0] += f_samples_count;
                inner::sample_phi(sampler, dyn_inner, f_samples_count, total_work, f_phi);
            }
            else {
                c_samples_count = samples_count_ell_1(sampler, dyn_inner, total_work, inner_stats);
                if (gc_sampler_data.antithetic){
                    inner_stats[0] += inner::sample_phi_antithetic(sampler, dyn_inner,
                                                                   f_samples_count,
                                                                   c_samples_count, total_work,
                                                                   f_phi, c_phi);
                }
                else{
                    inner_stats[0] += f_samples_count + c_samples_count;
                    inner::sample_phi(sampler, dyn_inner, f_samples_count, total_work, f_phi);
                    inner::sample_phi(sampler, dyn_inner, c_samples_count, total_work, c_phi);
                }
            }
            add_moments(f_phi, f_sums, gc_sampler_data.fine_moments);
            if (not fine_only){
                for (uint32_t i=0;i<gc_phi_size;i++)
                    f_phi[i] -= c_phi[i];
                add_moments(f_phi, diff_sums, gc_sampler_data.diff_moments);
            }
        }
#if defined(EXTRA_INNER_DIAGNOSITCS) && defined(__NVCC__)
        // This summation could be done in parallel, but it's not worh it
        inner_stats[4] = inner_stats[5] = 0;
        for (uint32_t i=0;i<dyn_inner;i++){
             inner_stats[4] += warp_outer ? warpSum_All(sampler[i].rand_count) : sampler[i].rand_count;
             inner_stats[5] += warp_outer ? warpSum_All(sampler[i].randn_count) : sampler[i].randn_count;
        }
#elif defined(EXTRA_INNER_DIAGNOSITCS)
        inner_stats[4] = sampler[0].rand_count;
        inner_stats[5] = sampler[0].randn_count;
#endif
        // Copy back to heap
        if (not DYNAMIC){
            *(base_samplers + core*dyn_inner) = *sampler;
        }

#ifdef __NVCC__
        if (warp_outer and !PAR_IS_WARP_MASTER)
            return;   // Only sum the masters
#endif
        OMP_PRAGMA(critical)
        {
            for (uint32_t i=0;i<gc_phi_size*gc_sampler_data.fine_moments;i++)
                atomicAdd(&out_f_sums[i], f_sums[i]);
            if (not fine_only)
                for (uint32_t i=0;i<gc_phi_size*gc_sampler_data.diff_moments;i++)
                    atomicAdd(&out_diff_sums[i], diff_sums[i]);
            atomicAdd(out_total_work, total_work);
            for (uint32_t i=0;i<c_inner_stats_count;i++)
                atomicAdd(&out_inner_stats[i], inner_stats[i]);
        }
    }
}

#ifdef __NVCC__
template<class SAMPLER>
PAR_GLOBAL void init_sampler(SAMPLER* samplers,
                             void *workspace,
                             uint64_t workspace_size,
                             void *dyn_blocks,
                             uint32_t count, uint64_t seed){
    uint32_t core = PAR_CORE_IDX;
    if (core >= count)
        return;
    samplers[core] = SAMPLER(static_cast<char*>(workspace) + core * workspace_size);
    samplers[core].dyn_block = static_cast<char*>(dyn_blocks) +
        inner_sampler_base::dyn_block_size * core;

    uint32_t warp = core / c_warp_size;
    uint32_t warp_count = count / c_warp_size;
    samplers[core].rand_state_sync.seed(seed, warp);
    samplers[core].rand_state.seed(seed, warp_count + core);
}
#endif

class nestedmc_data {
public:
    char *d_data, *h_data;

    uint32_t lvls_count;
    uint32_t block_size_per_lvl;
    uint32_t cores;
    uint32_t fine_moments;
    uint32_t diff_moments;

    void *d_workspace;
    void *d_samplers;
    void *d_dyn_blocks;

    PAR_HOST void zero() {
        PAR_DEVICE_MEMZERO(d_data, lvls_count * block_size_per_lvl);
    }
    uint64_t* d_inner_stats(uint32_t ell){
        return reinterpret_cast<uint64_t*>(d_data + ell * sizeof(uint64_t)*c_inner_stats_count);
    }
    work_t* d_total_work(uint32_t ell){
        return reinterpret_cast<work_t*>(d_data +
                                         lvls_count*sizeof(uint64_t)*c_inner_stats_count +
                                         ell*sizeof(work_t));
    }
    sum_t* d_f(uint32_t ell){
        return reinterpret_cast<sum_t*>(d_data +
                                         lvls_count*(sizeof(uint64_t)*c_inner_stats_count+sizeof(sum_t)) +
                                         ell*fine_moments*c_max_phi_size*sizeof(sum_t));
    }
    sum_t* d_diff(uint32_t ell){
        return reinterpret_cast<sum_t*>(d_data+
                                         lvls_count*(sizeof(uint64_t)*c_inner_stats_count+sizeof(sum_t)+fine_moments*c_max_phi_size*sizeof(sum_t)) +
                                         ell*diff_moments*c_max_phi_size*sizeof(sum_t));
    }
    uint64_t* h_inner_stats(uint32_t ell){
        return reinterpret_cast<uint64_t*>(h_data + ell * sizeof(uint64_t)*c_inner_stats_count);
    }
    work_t* h_total_work(uint32_t ell){
        return reinterpret_cast<work_t*>(h_data +
                                         lvls_count*sizeof(uint64_t)*c_inner_stats_count +
                                         ell*sizeof(work_t));
    }
    sum_t* h_f(uint32_t ell){
        return reinterpret_cast<sum_t*>(h_data +
                                         lvls_count*(sizeof(uint64_t)*c_inner_stats_count+sizeof(sum_t)) +
                                         ell*fine_moments*c_max_phi_size*sizeof(sum_t));
    }
    sum_t* h_diff(uint32_t ell){
        return reinterpret_cast<sum_t*>(h_data+
                                         lvls_count*(sizeof(uint64_t)*c_inner_stats_count+sizeof(sum_t)+fine_moments*c_max_phi_size*sizeof(sum_t)) +
                                         ell*diff_moments*c_max_phi_size*sizeof(sum_t));
    }

    PAR_HOST void device_to_host(uint32_t phi_size,
                                 sum_t *out_f,
                                 sum_t *out_diff,
                                 work_t &out_total_work,
                                 uint64_t *out_inner_stats){
        assert(phi_size <= c_max_phi_size);
        PAR_BOTH_MEMCPY(h_data, d_data, lvls_count*block_size_per_lvl);

        if (out_f)
            memcpy(out_f, h_f(0), lvls_count*fine_moments*phi_size*sizeof(sum_t));
        if (out_diff)
            memcpy(out_diff, h_diff(0), lvls_count*diff_moments*phi_size*sizeof(sum_t));
        memcpy(&out_total_work, h_total_work(0), lvls_count*sizeof(work_t));
        if (out_inner_stats)
            memcpy(out_inner_stats, h_inner_stats(0), lvls_count*sizeof(uint64_t)*c_inner_stats_count);
    }

    PAR_HOST nestedmc_data(uint32_t _fine_moments,
                            uint32_t _diff_moments, uint32_t _cores,
                            uint32_t _lvls_count) {
        assert(_cores % c_warp_size == 0);
        lvls_count = _lvls_count;
        cores = _cores;
        fine_moments = _fine_moments;
        diff_moments = _diff_moments;
        block_size_per_lvl = sizeof(work_t)+(c_max_phi_size*fine_moments+c_max_phi_size*diff_moments)*sizeof(sum_t)+sizeof(uint64_t)*c_inner_stats_count;
        d_workspace = 0;
        d_samplers = 0;
        d_dyn_blocks = 0;
        PAR_BOTH_MALLOC(h_data, d_data, lvls_count*(block_size_per_lvl));
    }

    PAR_HOST void allocate_samplers(uint32_t cores,
                                     uint32_t sampler_size,
                                     uint64_t workspace_size){
        cleanup_samplers();
        PAR_DEVICE_MALLOC(d_samplers, sampler_size * cores);
        PAR_DEVICE_MALLOC(d_workspace, workspace_size * cores);
        PAR_DEVICE_MALLOC(d_dyn_blocks, inner_sampler_base::dyn_block_size * cores);

    }
    PAR_HOST void cleanup_samplers(){
        PAR_DEVICE_FREE(d_workspace);
        PAR_DEVICE_FREE(d_samplers);
        PAR_DEVICE_FREE(d_dyn_blocks);
        d_workspace = 0;
        d_samplers = 0;
        d_dyn_blocks = 0;
    }

    PAR_HOST ~nestedmc_data(){
        cleanup_samplers();
        PAR_BOTH_FREE(h_data, d_data);
    }
};

template<typename SAMPLER>
PAR_HOST void init_samplers(uint64_t seed,
                            uint64_t workspace_size){
    // Allocate workspace memory (in device memory)
    g_p_nestedmc_data->allocate_samplers(g_p_nestedmc_data->cores,
                                         sizeof(SAMPLER),
                                         workspace_size);
#ifdef __NVCC__
    init_sampler<SAMPLER><<<g_p_nestedmc_data->cores/c_warp_size,
        c_warp_size>>>(
            static_cast<SAMPLER*>(g_p_nestedmc_data->d_samplers),
            g_p_nestedmc_data->d_workspace,
            workspace_size,
            g_p_nestedmc_data->d_dyn_blocks,
            g_p_nestedmc_data->cores, seed);
#else
    OMP_PRAGMA(parallel)
    {
        uint32_t core = PAR_CORE_IDX;
        FREQ_ASSERT(core < g_p_nestedmc_data->cores);
        SAMPLER* samplers = static_cast<SAMPLER*>(g_p_nestedmc_data->d_samplers);
        samplers[core] = SAMPLER(static_cast<char*>(g_p_nestedmc_data->d_workspace) + core * workspace_size);
        samplers[core].rand_state_sync.seed(seed, core);
        samplers[core].rand_state.seed(seed, core);
    }
#endif
}

template<typename SAMPLER>
PAR_HOST void sample(uint32_t ell,
                     uint64_t override_dyn_inner,
                     uint64_t *M,
                     sum_t *out_fsums,
                     sum_t *out_diffsums,
                     work_t *out_total_work,
                     uint64_t *out_inner_stats) {
    bool fine_only = ell==0 or out_diffsums == NULL;
    g_p_nestedmc_data->zero();

    // The following definitions allow us to call the template
    // function kernel_sample with boolean variables instead of constants.
#define PARAMS                                              \
    static_cast<SAMPLER*>(g_p_nestedmc_data->d_samplers),   \
        g_p_nestedmc_data->cores, dyn_inner,                \
        ell, *M, g_p_nestedmc_data->d_f(0),                 \
        fine_only?NULL:g_p_nestedmc_data->d_diff(0),        \
        g_p_nestedmc_data->d_total_work(0),                 \
        g_p_nestedmc_data->d_inner_stats(0)
#define KERNEL_SAMPLE_1(DYNAMIC)                \
    if (DYNAMIC) {KERNEL_SAMPLE_0(true)(PARAMS);}       \
    else {KERNEL_SAMPLE_0(false)(PARAMS);}

#ifdef __NVCC__
    uint32_t cores = g_p_nestedmc_data->cores;
    uint32_t dyn_inner = (ell>=7) ? 256 : 1;
    if (override_dyn_inner > 0)
        dyn_inner = override_dyn_inner;

    uint64_t threadPerBlock = (dyn_inner>1) ? 1:c_warp_size;
    uint64_t blocksPerGrid = MIN(*M, cores/(threadPerBlock*dyn_inner));

#define KERNEL_SAMPLE_0(DYNAMIC)                                    \
    kernel_sample<DYNAMIC, SAMPLER><<<blocksPerGrid, threadPerBlock>>>
#else
    (void)override_dyn_inner;   // Unused parameter when not using CUDA
    const uint32_t dyn_inner = 1;
#define KERNEL_SAMPLE_0(DYNAMIC) kernel_sample<false, SAMPLER>
#endif

    KERNEL_SAMPLE_1(dyn_inner>1);

    uint32_t phi_size;
    PAR_GET_CONST(phi_size, gc_phi_size);
    g_p_nestedmc_data->device_to_host(phi_size, out_fsums,
                                      out_diffsums, *out_total_work,
                                      out_inner_stats);
}

#endif   /// __NESTED_SAMPLERS_HPP__
