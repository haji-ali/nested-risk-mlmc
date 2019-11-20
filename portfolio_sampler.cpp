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
#include "gbm_approx.hpp"

namespace portfolio {
    enum class ComputeMethod {
        BS = 0,
        Exact = 1,
        Unbiased = 2,
        Count = 3,
    };

    enum class OptionType {
        Call=1,
        Put=-1
    };

    enum class  PhiType {
        Indicator=0,
        Max=1,
        Moments=2,
        FMoments=3
    };

    enum class Subsampling {
        None=0,
        Random=1,
        Random_Uniform=-1,
    };

    struct option_params {
        real_t strike;
        real_t maturity;
        real_t start;          //  Starting option value (risk-free), only used in BS
        real_t w;              // Weight of option
        real_t delta_est;      // Estimate of delta
        uint32_t stock_ind;    // Index of stock
        OptionType type;
        prob_t prob;
        real_t W_est;
    };
    struct stock_params{
        real_t S0, mu, sigma;
    };

    static const uint32_t compute_method_count = static_cast<uint32_t>(ComputeMethod::Count);

    typedef span<option_params> options_array;
    typedef span<stock_params> stocks_array;

    struct data {
        // Option parameters
        options_array options[compute_method_count];
        stocks_array stocks;

        // SDE parameters
        real_t r, rho, s1_rho2;

        // Risk parameters
        real_t max_loss;
        real_t tau;

        // Time discretization parameters
        uint32_t dt_h0inv;
        uint32_t dt_beta;     // Assumes it's a power of two
        real_t dt_prob_expo;   // Exponent to determine probability

        PhiType phi_type;
        uint32_t _phi_size;

        Subsampling subsample;

        bool use_delta_cv;
        bool use_coarse_delta_cv;
        bool use_antithetic_risk;
        bool use_milstein;

        inline PAR_BOTH uint32_t phi_size() const {
            return (phi_type == PhiType::Indicator || phi_type == PhiType::Max) ? 1 : _phi_size;
        }
    };

    struct data_ex : data {

        PAR_BOTH data_ex() {}
        PAR_BOTH data_ex(const data& other): data(other)
            {stau = std::sqrt(tau);}

        prob_t compute_prob[compute_method_count];
        span<prob_t> cdfs[compute_method_count];
        array delta_coalesce;
        uint32_t options_count;     // Sum of options[i].size()
        real_t stau;
        prob_t prob_norm;   // Sum of all probablities
        prob_t total_std;   // Sum of all probablities*W_est

        bool zero_constant;

        inline PAR_BOTH void reset() {
            for (uint32_t i=0;i<compute_method_count;i++){
                options[i].reset();
                cdfs[i].reset();
                compute_prob[i] = 0;
            }
            delta_coalesce.reset();
            r = rho = 0;
            options_count = 0;
            // Risk parameters
            stau = max_loss = tau = 0;

            // Time discretization parameters
            dt_h0inv = 0;
            dt_beta = 0;
            dt_prob_expo = 0;
            phi_type = PhiType::Indicator;
            _phi_size = 1;

            subsample = Subsampling::None;
            prob_norm = 0.;
        }
    };

    PAR_DEVICE_CONST data_ex gc_data;

    inline PAR_BOTH real_t simulate_stock(real_t S0, real_t mu, real_t sigma, real_t t, real_t dW) {
        return S0*std::exp(sigma * dW + (mu - 0.5*sigma*sigma)*t);
    }

    inline PAR_BOTH real_t calc_option_blackscholes(OptionType type,
                                                    real_t S0, real_t strike,
                                                    real_t r, real_t t, real_t sigma) {
        real_t v1 = std::log(S0/strike);
        real_t v2 = (r + (sigma*sigma)/2.0) * t;
        real_t v3 = sigma * std::sqrt(t);
        real_t d1 = (v1+v2)/v3;
        real_t d2 = d1 - sigma * std::sqrt(t);
        real_t mod = static_cast<real_t>(type); 
        return mod*S0 * norm_cdf(mod*d1) - mod*strike * std::exp(-r*t) * norm_cdf(mod*d2);
    }

    inline PAR_BOTH real_t calc_delta_blackscholes(OptionType type,
                                                   real_t S0, real_t strike,
                                                   real_t r, real_t t, real_t sigma) {
        real_t v1 = std::log(S0/strike);
        real_t v2 = (r + (sigma*sigma)/2.0) * t;
        real_t v3 = sigma * std::sqrt(t);
        real_t d1 = (v1+v2)/v3;
        return norm_cdf(d1) - (type == OptionType::Put);
    }

    inline PAR_BOTH real_t calc_gamma_blackscholes(real_t S0, real_t strike,
                                                   real_t r, real_t t,
                                                   real_t sigma) {
        real_t v1 = std::log(S0/strike);
        real_t v2 = (r + (sigma*sigma)/2.0) * t;
        real_t v3 = sigma * std::sqrt(t);
        real_t d1 = (v1+v2)/v3;
        // Same for both put and call
        return norm_pdf(d1)  / (S0 * sigma * std::sqrt(t));
    }

    inline PAR_BOTH real_t calc_theta_blackscholes(OptionType type,
                                                   real_t S0, real_t strike,
                                                   real_t r, real_t t, real_t sigma) {
        real_t v1 = std::log(S0/strike);
        real_t v2 = (r + (sigma*sigma)/2.0) * t;
        real_t v3 = sigma * std::sqrt(t);
        real_t d1 = (v1+v2)/v3;
        real_t d2 = d1 - sigma * std::sqrt(t);
        real_t mod = static_cast<real_t>(type); 
        return -S0 * norm_pdf(d1) * sigma  / (2 * std::sqrt(t))
            - mod*r*strike*std::exp(-r*t) * norm_cdf(mod*d2);
    }

    inline PAR_DEVICE real_t option_diff_from_blackscholes(const option_params& option,
                                                           real_t R_tau,
                                                           real_t r,
                                                           real_t sigma,
                                                           real_t tau) {
        real_t start = option.start;
        real_t finish = std::exp(-r * tau) *
            calc_option_blackscholes(option.type,
                                     R_tau,
                                     option.strike,
                                     r,
                                     option.maturity - tau,
                                     sigma);
        return option.w * (start-finish);
    }

    inline PAR_DEVICE real_t option_derv_from_bs(const option_params& option,
                                                 real_t S0,
                                                 real_t r,
                                                 real_t sigma) {
        return option.w * calc_delta_blackscholes(option.type,
                                                  S0, option.strike,
                                                  r, option.maturity,
                                                  sigma);
    }

    inline PAR_DEVICE real_t option_diff_from_stock(const option_params& option,
                                                    real_t S0_T, real_t St_T, real_t r) {
        // We are modeling the loss: if negative, it's a profit
        real_t mod = static_cast<real_t>(option.type); 
        real_t start = MAX(mod*(S0_T - option.strike), 0);
        real_t finish = MAX(mod*(St_T - option.strike), 0);
        return  option.w * std::exp(-r * option.maturity) * (start-finish);
    }

    inline PAR_DEVICE real_t option_derv_from_stock(const option_params& option,
                                                    real_t St_T, real_t S0, real_t r) {
        real_t df = (option.type == OptionType::Call) - (St_T < option.strike);
        return option.w * (St_T/S0) * std::exp(-r * option.maturity) * df;
    }

    class sampler  : public inner_sampler_base {
    public:
#ifdef DEBUG
        PAR_DEVICE void* debug(){ return Y.W_tau.begin(); }
#endif

        struct risk_sample_t {
            array W_tau;
            array R_tau;
            real_t constant;

            PAR_DEVICE risk_sample_t& copy(const risk_sample_t& rhs){
                FREQ_ASSERT(W_tau.size() == rhs.W_tau.size());
                FREQ_ASSERT(R_tau.size() == rhs.R_tau.size());
                for (uint32_t i=0;i<W_tau.size();i++)
                    W_tau[i] = rhs.W_tau[i];
                for (uint32_t i=0;i<R_tau.size();i++)
                    R_tau[i] = rhs.R_tau[i];
                constant = rhs.constant;
                return *this;
            }
        };

        inline PAR_BOTH static uint64_t GetWorkspaceSize(const data& d) {
            return d.stocks.size() * 2 * sizeof(real_t);
        }

        inline PAR_DEVICE sampler(void* workspace) {
            FREQ_ASSERT(workspace);
            Y.W_tau = array::create(reinterpret_cast<real_t*>(workspace), gc_data.stocks.size());
            Y.R_tau = array::create(reinterpret_cast<real_t*>(workspace) + gc_data.stocks.size(),
                                    gc_data.stocks.size());
        }

        static inline PAR_DEVICE void add_phi(real_t inner_avg, real_t coeff, real_t *phi){
            real_t u = inner_avg;
            real_t p = 1;
            switch (gc_data.phi_type){
            case PhiType::Max:
                *phi += coeff * MAX(u, 0);
                break;
            case PhiType::Moments:
                for (uint32_t i=0;i<gc_data.phi_size();i++){
                    p *= u;
                    phi[i] += coeff * p;
                }
                break;
            case PhiType::FMoments: {
                real_t tmp = M_PI * u;
                for (uint32_t k=0;k<gc_data.phi_size()/2;k++) {
                    phi[2*k] += coeff * std::cos((k+1) * tmp);
                    phi[2*k+1] += coeff * std::sin((k+1) * tmp);
                }
                if (gc_data.phi_size() % 2){
                    uint32_t k = gc_data.phi_size()/2;
                    phi[2*k] += coeff * std::cos((k+1) * tmp);
                }
            }
                break;
            case PhiType::Indicator:
            default:
                *phi += coeff * (u >= 0.f);
                break;
            }
        }

        inline PAR_DEVICE void sample_outer(uint64_t, work_t&){
            const auto& stocks = gc_data.stocks;
            real_t dW_sys = gc_data.stau * randn_sync();
            if (stocks.size() == 1){
                Y.W_tau[0] = dW_sys;
                Y.R_tau[0] = simulate_stock(stocks[0].S0, stocks[0].mu,
                                            stocks[0].sigma,
                                            gc_data.tau, Y.W_tau[0]);
            }
            else for (uint32_t i=0;i<stocks.size();i++){
                    real_t dW_ind = gc_data.stau * randn_sync();
                    Y.W_tau[i] = gc_data.rho * dW_sys + gc_data.s1_rho2*dW_ind;
                    Y.R_tau[i] = simulate_stock(stocks[i].S0, stocks[i].mu,
                                                stocks[i].sigma, gc_data.tau,
                                                Y.W_tau[i]);
                }

            Y.constant = gc_data.zero_constant?0:(-gc_data.max_loss - compute_delta_cv());
        }
        inline PAR_DEVICE real_t init_inner_sampler(uint64_t &, work_t& total_work){
            size_t bs_ind = static_cast<size_t>(ComputeMethod::BS);
            if (gc_data.subsample == Subsampling::None){
                // Pre-compute detereminstic options
                sum_t det_options = 0;
                for (uint32_t j=0;j<gc_data.options[bs_ind].size();j++)
                    det_options +=
                        sample_inner_option(total_work, ComputeMethod::BS,
                                            gc_data.options[bs_ind][j]);
                return static_cast<real_t>(det_options) + Y.constant;
            }
            return Y.constant;
        }

        PAR_DEVICE void copy_outer(const sampler& rhs){
            Y.copy(rhs.Y);
        }

        inline PAR_DEVICE real_t sample_inner(uint64_t, uint64_t, work_t &total_work) {
            if (gc_data.subsample == Subsampling::None)
                return sample_inner_all_options(total_work);
            return sample_inner_random_option(total_work);
        }

        inline PAR_DEVICE real_t sample_inner_all_options(work_t &total_work) {
            sum_t ret = 0;
            // In this case, we should subsample all portfolios, except deterministic one
            for (uint32_t i=1;i<compute_method_count;i++)
                for (uint32_t j=0;j<gc_data.options[i].size();j++)
                    ret += sample_inner_option(total_work, static_cast<ComputeMethod>(i),
                                               gc_data.options[i][j]);
            return static_cast<real_t>(ret);
        }

        inline PAR_DEVICE prob_t sample_inner_random_option_ind(ComputeMethod& cur_method,
                                                                size_t& option_index){
            prob_t prob_norm = gc_data.prob_norm;
            const prob_t* new_probs = gc_data.compute_prob;
            prob_t prob;
            size_t cur_method_i;
            real_t uc = randu_sync();
            cur_method_i = rand_int_pdf(uc * prob_norm, new_probs,
                                        new_probs + compute_method_count);
            cur_method = static_cast<ComputeMethod>(cur_method_i);

            if (gc_data.subsample == Subsampling::Random_Uniform) {
                real_t u = randu();
                option_index = std::ceil(u*gc_data.options[cur_method_i].size())-1;
                prob = 1./gc_data.options_count;
                FREQ_ASSERT(option_index < gc_data.options[cur_method_i].size());
            }
            else {
                const auto& cdfs = gc_data.cdfs[cur_method_i];
                real_t u = randu() * new_probs[cur_method_i];
                option_index = rand_int_cdf(u, cdfs.begin(), cdfs.end());
                prob = gc_data.options[cur_method_i][option_index].prob / prob_norm;
            }
            return prob;
        }

        inline PAR_DEVICE real_t sample_inner_random_option(work_t &total_work) {
            size_t option_index;
            ComputeMethod cur_method;
            prob_t prob = sample_inner_random_option_ind(cur_method, option_index);
            // Copy locally for efficient access
            option_params option = gc_data.options[static_cast<uint32_t>(cur_method)][option_index];
            return static_cast<real_t>(
                static_cast<prob_t>(sample_inner_option(total_work, cur_method, option)) / prob);
        }

        inline PAR_DEVICE real_t sample_inner_option(work_t &total_work,
                                                     ComputeMethod method,
                                                     const option_params& option) {
            switch (method) {
            case ComputeMethod::BS:
                return sample_inner_bs(option, total_work);
            case ComputeMethod::Exact:
                return sample_inner_exact(option, total_work);
            case ComputeMethod::Unbiased:
                return sample_inner_unbiased(option, total_work);
            default:
                FREQ_ASSERT(false);  // Should not come here
                return 0;
            }
        }
        PAR_DEVICE real_t sample_delta_unbiased(uint32_t option_index,
                                                work_t& total_work){
            const auto& option = gc_data.options[
                static_cast<size_t>(ComputeMethod::Unbiased)][option_index];
            const auto& stock = gc_data.stocks[option.stock_ind];

            real_t S0 = stock.S0;
            real_t sigma = stock.sigma;
            real_t r = gc_data.r;
            real_t T = option.maturity - gc_data.tau;
            uint64_t N = gc_data.dt_h0inv;

            // Need to sample
            real_t dW_0_tau = gc_data.stau * randn();
            real_t S0_t_1_i, S0_t_2_i = 0;
            S0_t_1_i = simulate_stock(stock.S0, r,
                                      stock.sigma, gc_data.tau, dW_0_tau);
            S0_t_2_i = simulate_stock(stock.S0, r,
                                      stock.sigma, gc_data.tau, -dW_0_tau);

            sum_t ST_f[] = {S0_t_1_i, S0_t_2_i};
            real_t sig[] = {sigma, -sigma};

            // Need to define this as a functor instead of using a lambda
            // so that we can specify PAR_DEVICE (without using experimental features from CUDA)
            struct tmp_randn {
                sampler* pthis;
                PAR_DEVICE real_t operator() () { return pthis->randn(); }
            };

            real_t dW = simulate_stock_dt<false>(tmp_randn{this},
                                                 ST_f, N, r, sig,
                                                 T,
                                                 sizeof(sig)/sizeof(sig[0]),
                                                 gc_data.use_milstein);

            real_t ST[] = {simulate_stock(S0_t_1_i, r, sigma, T, dW),
                           simulate_stock(S0_t_2_i, r, sigma, T, -dW)};
            total_work += 2*N * option.W_est;

            real_t val_f=0;
            val_f += option_derv_from_stock(option, ST_f[0], S0, r);
            val_f += option_derv_from_stock(option, ST_f[1], S0, r);
            val_f -= option_derv_from_stock(option, ST[0], S0, r);
            val_f -= option_derv_from_stock(option, ST[1], S0, r);
            // We use the exact delta and the exact sample as control variate
            return val_f / (2.0 * option.w);
        }

    protected:
        risk_sample_t Y;

        inline PAR_DEVICE real_t delta_factor(uint32_t stock_ind) const {
            const auto& stock = gc_data.stocks[stock_ind];
            return Y.R_tau[stock_ind] - stock.S0;
        }
        inline PAR_DEVICE real_t sample_inner_bs(const option_params& option,
                                                 work_t &total_work) {
            ///// OPTION
            total_work += option.W_est;
            uint32_t stock_ind = option.stock_ind;
            const auto& stock = gc_data.stocks[stock_ind];
            real_t val = option_diff_from_blackscholes(option,
                                                       Y.R_tau[stock_ind],
                                                       gc_data.r,
                                                       stock.sigma,
                                                       gc_data.tau);
            real_t cv = gc_data.use_delta_cv?
                (delta_factor(stock_ind)*
                 option_derv_from_bs(option, stock.S0, gc_data.r, stock.sigma)):0;
            return val + cv;
        }

        inline PAR_DEVICE real_t sample_inner_exact(const option_params& option,
                                                    work_t &total_work){
            bool anti = gc_data.use_antithetic_risk;
            ///// STOCK
            real_t S0_T_1_i, S0_T_2_i, St_T_i;
            total_work += option.W_est;

            uint32_t stock_ind = option.stock_ind;
            const auto& stock = gc_data.stocks[stock_ind];

            real_t dW_0_tau = gc_data.stau * randn();
            real_t dW_tau_T = std::sqrt(option.maturity - gc_data.tau) * randn();

            // For the starting stock, simulate another risk and advance with
            // the same Brownian motion
            real_t S0_t_1 = simulate_stock(stock.S0, gc_data.r, stock.sigma,
                                           gc_data.tau, dW_0_tau);
            S0_T_1_i = simulate_stock(S0_t_1, gc_data.r, stock.sigma,
                                      option.maturity - gc_data.tau, dW_tau_T);
            if (anti){
                real_t S0_t_2 = simulate_stock(stock.S0, gc_data.r, stock.sigma,
                                               gc_data.tau, -dW_0_tau);   // Antitheitc
                S0_T_2_i = simulate_stock(S0_t_2, gc_data.r, stock.sigma,
                                          option.maturity - gc_data.tau, dW_tau_T);
            }
            St_T_i = simulate_stock(Y.R_tau[stock_ind], gc_data.r, stock.sigma,
                                    option.maturity - gc_data.tau, dW_tau_T);

            ///// OPTION

            real_t val1 = option_diff_from_stock(option, S0_T_1_i, St_T_i, gc_data.r);
            real_t val2 = val1;
            if (anti)
                val2 = option_diff_from_stock(option, S0_T_2_i, St_T_i, gc_data.r);

            real_t delta_val1 = 0.;
            real_t delta_val2 = 0;
            if (gc_data.use_delta_cv){
                real_t diff = delta_factor(stock_ind);
                delta_val1 = diff *
                    option_derv_from_stock(option, S0_T_1_i, stock.S0, gc_data.r);
                delta_val2 = delta_val1;
                if (anti){
                    delta_val2 = diff *
                        option_derv_from_stock(option, S0_T_2_i, stock.S0,
                                               gc_data.r);
                }
            }
            return 0.5*(val1 + val2 + delta_val1 + delta_val2);
        }

        inline PAR_DEVICE real_t sample_inner_unbiased(const option_params& option,
                                                       work_t &total_work){
            prob_t e = static_cast<prob_t>(gc_data.dt_prob_expo);
            prob_t x = static_cast<prob_t>(gc_data.dt_beta);
            FREQ_ASSERT(e < 0 && x > 1);
            // Assume: e<0
            // Probability is \propto x^(e L)
            // Norm is \sum_{i=0}^\infty x^(e i) = 1 / (1. - x^e)
            // Probability is \propto x^(e L) * (1. - x^e)
            // Probability = x^(e L) * (1. - x^e)
            // CDF = 1-x^(e * (L+1))
            // Probability inverse is = x^(-e L) / (1. - x^e)
            uint32_t L = std::floor(std::log(randu_sync()) / (std::log(x) * e));
            real_t ret = sample_inner_approx(option, total_work, L);
            prob_t L_prob_1 = std::pow(x, -e * L) / (1. - std::pow(x,e));
            return static_cast<real_t>(static_cast<prob_t>(ret) * L_prob_1);
        }

        inline PAR_DEVICE real_t sample_inner_approx(const option_params& option,
                                                     work_t &total_work, uint32_t L,
                                                     bool fine_only=false){
            uint32_t stock_ind = option.stock_ind;
            const auto& stock = gc_data.stocks[stock_ind];

            real_t S0_t_1_i, S0_t_2_i = 0;
            real_t dW_0_tau = gc_data.stau * randn();
            S0_t_1_i = simulate_stock(stock.S0, gc_data.r,
                                      stock.sigma, gc_data.tau, dW_0_tau);
            S0_t_2_i = simulate_stock(stock.S0, gc_data.r,
                                      stock.sigma, gc_data.tau, -dW_0_tau);


            uint64_t N_l = uint_pow(static_cast<uint64_t>(gc_data.dt_beta), static_cast<uint64_t>(L));
            uint64_t N = gc_data.dt_h0inv * N_l;
            total_work += N_l * option.W_est;   // We expect W_est to include h0
            // Compute everything even if use_antithetic_risk is false.
            // We should have it true anyways
            // Using sum_t because using single precision seems to produce wrong results
            sum_t S[] = {S0_t_1_i, Y.R_tau[stock_ind], S0_t_2_i,
                         S0_t_1_i, Y.R_tau[stock_ind], S0_t_2_i};

            // Need to define this as a functor instead of using a lambda
            // so that we can specify PAR_DEVICE (without using experimental features from CUDA)
            struct tmp_randn { 
                sampler* pthis;
                PAR_DEVICE real_t operator() () { return pthis->randn(); }
            };

            if (!fine_only && L){
                simulate_stock_dt<true>(tmp_randn{this},
                                        S, N, gc_data.r, stock.sigma,
                                        option.maturity - gc_data.tau, 3,
                                        gc_data.use_milstein,
                                        gc_data.dt_beta);
                total_work +=  (N_l / gc_data.dt_beta) * option.W_est;
            }
            else
                simulate_stock_dt<false>(tmp_randn{this},
                                         S, N, gc_data.r, stock.sigma,
                                         option.maturity - gc_data.tau, 3,
                                         gc_data.use_milstein);

            real_t val_f=0, val_c=0, val_d_f=0, val_d_c=0;
            const int c = 3;
            const real_t m=gc_data.use_antithetic_risk?0.5:1.;
            const real_t diff = delta_factor(stock_ind);
            const bool do_d_f = gc_data.use_delta_cv && (gc_data.use_coarse_delta_cv || L == 0);
            const bool do_d_c = gc_data.use_delta_cv && gc_data.use_coarse_delta_cv;

            for (int i=0;i<(gc_data.use_antithetic_risk?2:1);i++){
                int ic = 2*i;  // Either 0 or 2, the antithetic pairs
                val_f += m*option_diff_from_stock(option, S[ic+0], S[1], gc_data.r);

                if (do_d_f)
                    val_d_f += m*option_derv_from_stock(option, S[ic+0], stock.S0, gc_data.r);

                if (fine_only or L == 0) continue;

                val_c += m*option_diff_from_stock(option, S[ic+c+0], S[c+1], gc_data.r);
                if (do_d_c)
                    val_d_c += m*option_derv_from_stock(option, S[ic+c+0], stock.S0, gc_data.r);
            }
            return ((val_f - val_c) + diff*(val_d_f - val_d_c));
        }

        inline PAR_DEVICE real_t compute_delta_cv() const {
            if (!gc_data.use_delta_cv)
                return 0;
            real_t sum=0;
            const auto& stocks = gc_data.stocks;
            for (uint32_t i=0;i<stocks.size();i++)
                sum += delta_factor(i) * gc_data.delta_coalesce[i];
            return sum;
        }
    };


    extern "C" void portfolio_cleanup() {
        portfolio::data_ex data;
        PAR_GET_CONST(data, portfolio::gc_data);
        for (uint32_t i=0;i<portfolio::compute_method_count;i++){
            PAR_DEVICE_FREE(data.options[i].begin());
            PAR_DEVICE_FREE(data.cdfs[i].begin());
            data.cdfs[i].reset();
            data.options[i].reset();
        }
        PAR_DEVICE_FREE(data.delta_coalesce.begin());
        PAR_SET_CONST(portfolio::gc_data, data);
    }

    extern "C" void portfolio_update_data(const portfolio::data *data){
        portfolio_cleanup();
        portfolio::data_ex d_data = *data;

        d_data.zero_constant = false;

        d_data.total_std = 0;
        d_data.prob_norm = 0;
        d_data.options_count = 0;
        d_data.stocks = portfolio::stocks_array::create(
            parDuplicateInDevice(&data->stocks[0], data->stocks.size()),
            data->stocks.size());

        std::vector<real_t> delta(data->stocks.size());
        for (uint32_t i=0;i<portfolio::compute_method_count;i++){
            // CDF is normalised for every compute option
            const auto &op_ordered = data->options[i];
            std::vector<prob_t> cdf(op_ordered.size());
            real_t c=0;
            prob_t prev_prob = std::numeric_limits<prob_t>::max();
            d_data.compute_prob[i] = 0;
            for (uint32_t j=0;j<op_ordered.size();j++){
                const portfolio::option_params& option = op_ordered[j];
                FREQ_ASSERT(option.prob > 0 && prev_prob >= option.prob);
                (void)prev_prob;   // Unused variable when FREQ_ASSERT is empty

                cdf[j] = c;
                prev_prob = option.prob;
                c += option.prob;

                delta[option.stock_ind] += option.w * option.delta_est;
                d_data.prob_norm += option.prob;
                d_data.total_std += option.prob*option.W_est;
                d_data.compute_prob[i] += option.prob;
            }
            auto d_options = parDuplicateInDevice(&op_ordered[0], op_ordered.size());
            auto d_cdf = parDuplicateInDevice(&cdf[0], cdf.size());

            d_data.options[i] = portfolio::options_array::create(
                d_options, op_ordered.size());
            d_data.cdfs[i] = span<prob_t>::create(d_cdf, cdf.size());
            d_data.options_count += op_ordered.size();
        }
        auto d_delta = parDuplicateInDevice(&delta[0], delta.size());
        d_data.delta_coalesce = array::create(d_delta, delta.size());
        PAR_SET_CONST(portfolio::gc_data, d_data);
        PAR_SET_CONST(gc_phi_size, data->phi_size());
    }

    extern "C" void portfolio_init(const portfolio::data *data,
                                   uint64_t seed){
        portfolio_update_data(data);

        init_samplers<portfolio::sampler>(
            seed, portfolio::sampler::GetWorkspaceSize(*data));
    }
    extern "C" void portfolio_comp_params(portfolio::data *data) {
        for (uint32_t i=0;i<portfolio::compute_method_count;i++){
            for (uint32_t j=0;j<data->options[i].size();j++){
                auto& option = data->options[i][j];
                const auto& stock = data->stocks[option.stock_ind];
                option.start = portfolio::calc_option_blackscholes(
                    option.type, stock.S0, option.strike, data->r,
                    option.maturity, stock.sigma);
                option.delta_est = portfolio::calc_delta_blackscholes(
                    option.type, stock.S0, option.strike, data->r,
                    option.maturity, stock.sigma);
            }
        }
    }

    PAR_GLOBAL void kernel_sample_delta(sampler* base_samplers,
                                        uint32_t cores,
                                        uint32_t option_index,
                                        uint64_t totalM,
                                        uint32_t moments,
                                        sum_t *out_f_sums,
                                        sum_t *out_total_work) {
        OMP_PRAGMA(parallel)
            {
                uint32_t core = PAR_CORE_IDX;
#ifdef __NVCC__
                uint32_t samples_per_block = blockDim.x / c_warp_size;
                uint64_t M = totalM / (gridDim.x*samples_per_block);
                M += (core/c_warp_size) < (totalM % (gridDim.x*samples_per_block));
#else
                uint32_t M = totalM;
#endif

                FREQ_ASSERT(core < cores);

                work_t total_work = 0.;
                sum_t f_sums[c_max_moments] = {0};

                sampler cur_sampler = *(base_samplers + core);
                // Generate single sample
                OMP_PRAGMA(for) for (uint64_t m=0;m<M;m++){
                    real_t delta = cur_sampler.sample_delta_unbiased(option_index, total_work);
                    add_moments(&delta, 1, f_sums, moments);
                }

                // Copy back to heap
                *(base_samplers + core) = cur_sampler;

#ifdef __NVCC__
                if (!PAR_IS_WARP_MASTER)
                    return;   // Only sum the masters
#endif

                OMP_PRAGMA(critical)
                    {
                        for (uint32_t i=0;i<moments;i++)
                            atomicAdd(&out_f_sums[i], f_sums[i]);
                        atomicAdd(out_total_work, total_work);
                    }
            }
    }

    extern "C" void portfolio_sample(uint32_t ell, uint64_t override_dyn_inner, uint64_t M,
                                     sample_output *out){
        out->M = M;
        sample<portfolio::sampler>(ell, override_dyn_inner,
                                   &out->M, out->fsums,
                                   out->diffsums, &out->total_work,
                                   out->inner_stats);
    }


    extern "C" void portfolio_sample_delta_unbiased(uint32_t option_index,
                                                    uint64_t *M,
                                                    sum_t *sums,
                                                    uint32_t moments,
                                                    work_t *total_work){
        uint32_t cores = g_p_nestedmc_data->cores;
        g_p_nestedmc_data->zero();

#ifdef __NVCC__
        int blocksPerGrid = MIN(static_cast<uint64_t>(cores / c_warp_size), *M);
        kernel_sample_delta<<<blocksPerGrid, c_warp_size>>>
#else
        kernel_sample_delta
#endif
            (static_cast<sampler*>(g_p_nestedmc_data->d_samplers),
             cores,
             option_index,
             *M,
             moments,
             g_p_nestedmc_data->d_f(0),
             g_p_nestedmc_data->d_total_work(0));
        g_p_nestedmc_data->device_to_host(1, sums, NULL, *total_work, NULL);
    }


    extern "C" portfolio::data portfolio_gen_data(uint32_t P, uint32_t Q,
                                                  uint64_t seed, double* probs,
                                                  bool uniform_weights,
                                                  bool uniform_prob) {
        portfolio::data por_data;
        por_data.r = 0.05;
        por_data.rho = 0.2;
        por_data.s1_rho2 = std::sqrt(1-POW2(por_data.rho));
        por_data.max_loss = 0.7;
        por_data.tau = 0.02;
        por_data.dt_h0inv = 8;
        por_data.dt_beta = 4;
        por_data.dt_prob_expo = -1.5;
        por_data.phi_type = portfolio::PhiType::Indicator;
        por_data._phi_size = 1;
        por_data.subsample = portfolio::Subsampling::None;


        typedef std::uniform_real_distribution<real_t> rand_t;
        typedef std::uniform_int_distribution<uint32_t> randi_t;
        std::mt19937 gen(seed);
        std::vector<real_t> S0(Q), sig(Q), mu(Q);
        std::vector<uint32_t> asset(P), type(P);
        std::vector<real_t> maturity(P, 1), strike(P), weight(P);

        auto fill_rand = [&gen](auto& arr, auto dist){
                             std::generate(arr.begin(), arr.end(), std::bind(dist, gen));};

        fill_rand(S0, rand_t(90, 110));
        fill_rand(sig, rand_t(0.01, 0.4));
        fill_rand(mu, rand_t(0.03, 0.3));
        fill_rand(type, randi_t(0, 1));
        fill_rand(asset, randi_t(0, Q-1));
        fill_rand(maturity, rand_t(1, 5));
        fill_rand(strike, rand_t(80, 120));
        if (!uniform_weights)
            fill_rand(weight, std::normal_distribution<real_t>(0,1));

        por_data.use_delta_cv = true;
        por_data.use_coarse_delta_cv = false;
        por_data.use_antithetic_risk = true;
        por_data.use_milstein = true;

        portfolio::option_params *ops = new portfolio::option_params[P];
        portfolio::stock_params *stock = new portfolio::stock_params[Q];
        assert(P>= 2*Q);
        for (uint32_t i=0;i<Q;i++){
            type[i] = 0;
            type[Q+i] = 1;
            asset[i] = asset[Q+i] = i;

            stock[i].S0 = S0[i];
            stock[i].mu = mu[i];
            stock[i].sigma = sig[i];
        }

        for (uint32_t i=0;i<P;i++){
            ops[i].w = weight[i];
            ops[i].strike = strike[i];
            ops[i].type = type[i]?portfolio::OptionType::Put:portfolio::OptionType::Call;
            ops[i].maturity = maturity[i];
            ops[i].stock_ind = asset[i];

            ops[i].prob = -1;
            ops[i].W_est = 1;
        }

        por_data.stocks = portfolio::stocks_array::create(stock, Q);
        // Divide across compute classes
        uint32_t cur=0;
        for (uint32_t i=0;i<portfolio::compute_method_count;i++){
            uint32_t count=probs[i]*P;
            por_data.options[i] = portfolio::options_array::create(ops+cur, count);
            cur += count;
        }
        assert(cur==P);
        // Update W_est
        real_t s=2, g=1;
        real_t factor = (1 - pow(static_cast<real_t>(por_data.dt_beta), ((-s-g)/2.))) /
            (1 - pow(static_cast<real_t>(por_data.dt_beta), ((-s+g)/2.)));
        for (auto op : por_data.options[static_cast<int>(portfolio::ComputeMethod::Unbiased)])
            op.W_est *= factor * por_data.dt_h0inv;

        // Estimate delta
        portfolio_comp_params(&por_data);

        // Delta hedge
        std::vector<sum_t> call_delta(Q,0), put_delta(Q,0);
        sum_t total_delta = 0;
        for (uint32_t i=0;i<portfolio::compute_method_count;i++){
            for (const auto &op : por_data.options[i]){
                sum_t tmp = op.delta_est * op.w;
                total_delta += tmp;
                if (op.type == portfolio::OptionType::Call)
                    call_delta[op.stock_ind] += tmp;
                else
                    put_delta[op.stock_ind] += tmp;
            }
        }
        sum_t total_weight=0;
        for (uint32_t i=0;i<portfolio::compute_method_count;i++){
            for (auto &op : por_data.options[i]){
                if (op.type == portfolio::OptionType::Call)
                    op.w *= -put_delta[op.stock_ind] / call_delta[op.stock_ind];
                total_weight += op.w;
            }
        }
        // Normalize weights
        // Update probabilities and sort options
        for (uint32_t i=0;i<portfolio::compute_method_count;i++){
            for (auto &op : por_data.options[i]){
                op.w /= total_weight;
                op.prob = uniform_prob ? 1 : std::abs(op.w) / std::sqrt(op.W_est);
            }

            std::sort(por_data.options[i].begin(), por_data.options[i].end(),
                      [](const portfolio::option_params& x,
                         const portfolio::option_params& y){return (x.prob>y.prob);});
        }
        return por_data;
    }

    extern "C" void portfolio_free_data(portfolio::data* pdata) {
        delete [] pdata->stocks.begin();
        delete [] pdata->options[0].begin();
        pdata->stocks.reset();
        for (uint32_t i=0;i<portfolio::compute_method_count;i++)
            pdata->options[i].reset();
    }

    extern "C" void portfolio_estimate_delta_for_unbiased(data *pdata, double TOL) {
        uint32_t option_index = 0;
        work_t work=0;
        for (auto& op : pdata->options[static_cast<int>(portfolio::ComputeMethod::Unbiased)]){
            uint64_t M = 1000, total_M=0;
            sum_t sums[2] = {};
            sum_t total_sums[2] = {};
            while (true) {
                portfolio_sample_delta_unbiased(option_index, &M, sums, 2, &work);
                total_sums[0] += sums[0];
                total_sums[1] += sums[1];
                total_M += M;
                double Vl = total_sums[1]/total_M - POW2(total_sums[0]/total_M);
                if (3. * std::sqrt(Vl/total_M) < TOL)
                    break;
            }
            option_index++;

            op.delta_est += total_sums[0]/total_M;
        }
    }
}

#ifdef INCLUDE_MAIN_MLMC

int main(// int argc, char **argv
    ){
    int cores = 1;
#ifdef __NVCC__
    cores = 2048 * c_warp_size;   // 65536
#elif defined(_OPENMP)
    cores = omp_get_max_threads();
#endif

    mlmc_real::mlmc_input_t input;
    
    // Input parameters
    uint32_t P = 1000;    // Option count
    uint32_t Q = 16;        // Stock count
    double probs[] = {0.3, 0.5, 0.2};  // proportion of computation methods: Exact, Simulation, Approx
    bool uniform_prob = false;    // Use uniform probabilities for random sub-sampler
    bool uniform_weights = false; // Use uniform weights (=1) for options instead of random ones
    uint64_t portfolio_seed = 1;  // Seed used to generate portfolio
    uint64_t calc_seed = 0;       // Seed used to sample options
    uint32_t N0 = 32;          // Minimum number of inner samples
    uint32_t beta = 4;         // Refinement factor for inner sampler
    real_t adaptive_r = 1.5;   // r-value for adaptive algorithm of inner sampler, 0 for non-adaptive
    bool antithetic = true;    // Use antithetic estimator for the inner sampler
    double max_loss = 1.7;    // Max loss to compute probability of.
    portfolio::Subsampling subsample = portfolio::Subsampling::Random;
    bool use_cv = true;        // Use control variates in inner sampler
    input.M0 = 2048;
    input.verbose = true;
    input.targetTOL = 1e-4;
    
    nested_sampler_params nested_param(mlmc_real::moments, mlmc_real::moments,
                                       N0, beta, adaptive_r, antithetic);
    init_nestedmc_sampler(cores, &nested_param);
    
    portfolio::data pdata = portfolio::portfolio_gen_data(P, Q, portfolio_seed,
                                                          probs, uniform_weights,
                                                          uniform_prob);
    pdata.max_loss = max_loss;
    pdata.subsample = subsample;
    pdata.use_coarse_delta_cv = false;      // Because our options are only Lipschitz
    pdata.use_delta_cv = pdata.use_antithetic_risk = use_cv;

    portfolio::portfolio_init(&pdata, calc_seed);

    if (use_cv &&
        pdata.options[static_cast<int>(portfolio::ComputeMethod::Unbiased)].size() > 0) {
        if (input.verbose)
            std::cout << "Estimating the Deltas of the approximate options" << std::endl;
        portfolio::portfolio_estimate_delta_for_unbiased(&pdata, input.targetTOL);
        portfolio::portfolio_update_data(&pdata);
    }
    
    uint32_t phi_size = pdata.phi_size();
    assert(phi_size==1);  // Only real value are supported
    input.sample_level = [&, phi_size] (uint32_t ell, uint64_t M) {
                             mlmc_real::mc_diff_output_t mc_out;
                             std::clock_t start = std::clock();
                             sample_output out(mlmc_real::moments, mlmc_real::moments,
                                               phi_size);
                             portfolio::portfolio_sample(ell, 0, M, &out);
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
    
    portfolio::portfolio_cleanup();
    cleanup_nestedmc_sampler();
    portfolio::portfolio_free_data(&pdata);
    return 0;
}
#endif

