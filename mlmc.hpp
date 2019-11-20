#ifndef __MLMC_HPP__
#define __MLMC_HPP__

#include <vector>
#include <iomanip>
#include <limits>
#include <type_traits>
#include "table_output.hpp"

template<typename T, uint32_t S=2>
struct mlmc_t {
    static const uint32_t moments = S;

    struct mc_diff_output_t {
        mc_diff_output_t() : sums_fine{0}, sums_diff{0}, work(0), time(0), M(0) {}
        
        T sums_fine[S];
        T sums_diff[S];

        double work;
        double time;   // In seconds
        uint64_t M;

        T dEl() const { return sums_diff[0]/M; };
        T fEl() const { return sums_fine[0]/M; };
        T dVl() const { return sums_diff[1]/M - POW2(dEl()); };
        T fVl() const { return sums_fine[1]/M  - POW2(sums_fine[0]/M); };
        double Wl() const { return work/M; };

        template <int SS=S >
        typename std::enable_if< SS>=4, T >::type
        dKl() const { return sums_diff[3]/M / dVl(); };
        
        template <int SS=S >
        typename std::enable_if< SS>=4, T >::type
        fKl() const { return sums_fine[3]/M / fVl(); };

        mc_diff_output_t& operator+=(const mc_diff_output_t& lhs){
            for (uint32_t s=0;s<S;s++){
                sums_fine[s] += lhs.sums_fine[s];
                sums_diff[s] += lhs.sums_diff[s];
            }
            work += lhs.work;
            time += lhs.time;
            M += lhs.M;
            return *this;
        }
    };
    
    struct mlmc_output_t {
        mlmc_output_t() : starting_level(0), bias_est(std::nan("")), TOL(std::nan("")) {
            // Reasonable defaults
        }
        
        std::vector< mc_diff_output_t > levels;
        uint32_t starting_level;
        double bias_est;
        double stat_error;
        double TOL;

        uint32_t L() const { return levels.size()-starting_level-1; };

        // Takes into account starting_level
        double total_variance() const{
            uint32_t ell=starting_level;
            double total = levels[ell].fVl()/levels[ell].M;
            for (ell=starting_level+1;ell<levels.size();ell++){
                total += levels[ell].dVl()/levels[ell].M;
            }
            return total;
        }

        std::string to_string() const {
            using namespace std;
            stringstream str;
            double discarded_time=0, discarded_work=0;
            for (uint32_t ell=0;ell<starting_level;ell++){
                discarded_work += levels[ell].work;
                discarded_time += levels[ell].time;
            }

            double total_time=discarded_time, total_work=discarded_work;
            for (uint32_t ell=starting_level;ell<levels.size();ell++){
                total_work += levels[ell].work;
                total_time += levels[ell].time;
            }

            T Eg = levels[starting_level].fEl();
            for (uint32_t ell=starting_level+1;ell<levels.size();ell++)
                Eg += levels[ell].dEl();
            str << "TOL: " << TOL << std::endl;

            str << "RMS: " << std::sqrt(POW2(bias_est)+POW2(stat_error)) <<
                " (" <<
                bias_est <<
                ", " <<
                stat_error <<
                ") " << std::endl;

            str << "Eg: " << Eg << " --- [" << Eg-TOL << "," << Eg+TOL << "]" << std::endl;

            str << "Discarded Work: " << discarded_work << std::endl;
            str << "Discarded Time: " << discarded_time << std::endl;
            str << "Total Work: " << total_work << std::endl;
            str << "Total Time: " << total_time << std::endl;

            str << std::scientific << setprecision(4);
            
            table_t mlmc_table(levels.size());

            mlmc_table.add("ell", 3,
                           [&] (stringstream& o, uint i) {
                               if (i<starting_level)
                                   o << "*" + std::to_string(i);
                               else o << i;
                           });

            mlmc_table.add("dEl", 10,
                           [&] (stringstream& o, uint i) {o << levels[i].dEl();});


            mlmc_table.add("fEl", 10,
                           [&] (stringstream& o, uint i) {o << levels[i].fEl();});


            mlmc_table.add("dVl", 10,
                           [&] (stringstream& o, uint i) {o << levels[i].dVl();});


            mlmc_table.add("fVl", 10,
                           [&] (stringstream& o, uint i) {o << levels[i].fVl();});

            helper_add_Kl(mlmc_table);

            mlmc_table.add("Wl", 10,
                           [&] (stringstream& o, uint i) {o << levels[i].Wl();});


            mlmc_table.add("M", 5,
                           [&] (stringstream& o, uint i) {o << levels[i].M;});


            mlmc_table.add("Time", 10,
                           [&] (stringstream& o, uint i) {o << levels[i].time/levels[i].M;});
            
            str << mlmc_table;
            return str.str();
        }

        private:
        //////////////////////////////////////////////////////////// Helpers
        template <int SS=S >
        typename std::enable_if< SS>=4, void >::type helper_add_Kl(table_t &mlmc_table) const{
            using namespace std;
            mlmc_table.add("dKl", 10,
                           [&] (stringstream& o, uint i) {o << levels[i].dKl();});
            mlmc_table.add("fKl", 10,
                           [&] (stringstream& o, uint i) {o << levels[i].fKl();});
        }
        template <int SS=S >
        typename std::enable_if< SS<4, void >::type helper_add_Kl(table_t &) const {}
        ////////////////////////////////////////////////////////////
    };

    static double default_bias_estimate(const mlmc_output_t& out, double rate=0.5){
        if (out.levels.size() > 2){
            double bias0 = std::abs(out.levels[out.levels.size()-1].dEl());
            double bias1 = std::abs(out.levels[out.levels.size()-2].dEl())*rate;
            return bias0>bias1?bias0:bias1;
        }
        return std::numeric_limits<double>::infinity();
    }

    struct mlmc_input_t {
        mlmc_input_t() : targetTOL(1e-3), startTOL(0.5), theta(0.5),
                         starting_level_coeff(1.5), Ca(3), verbose(true), M0(100) {
            // Reasonable defaults
            est_bias = [] (const mlmc_output_t& out) {return default_bias_estimate(out);};
        }
        typedef std::function<mc_diff_output_t (uint32_t ell, uint64_t M)> fn_sample_level_t;
        typedef std::function<double (const mlmc_output_t& out)> fn_est_bias_t;

        fn_sample_level_t sample_level;
        fn_est_bias_t est_bias;

        double targetTOL;
        double startTOL;
        double theta;
        double starting_level_coeff;
        double Ca;
        bool verbose;
        uint64_t M0;

        uint32_t zero_protection;  // TODO?
    };
    
    static uint32_t better_starting_level(const mlmc_output_t& out, double starting_level_coeff){
        uint32_t ell0 = out.starting_level;
        while (true){
            bool change = false;
            double lhs = std::sqrt(out.levels[ell0].fVl() * out.levels[ell0].Wl() );
            for (uint32_t ell0p=ell0+1;ell0p<out.levels.size();ell0p++){
                // test ell0p
                lhs += std::sqrt(out.levels[ell0p].dVl() * out.levels[ell0p].Wl());
                double rhs = std::sqrt(out.levels[ell0p].fVl() * out.levels[ell0p].Wl());

                if (lhs / rhs > starting_level_coeff){
                    // Change starting level
                    change = true;
                    break;
                }
            }
            if (not change) break;

            ell0 += 1; // Check next level
        }
        return ell0;
    }


    static std::vector<uint64_t> get_optimal_Ml(double tol2, uint64_t M0, const mlmc_output_t &out){
        std::vector<uint64_t> Ml(out.levels.size());
        uint32_t ell=out.starting_level;
        double lambda = out.levels[ell].M > 0 ?
            std::sqrt(out.levels[ell].Wl() * out.levels[ell].fVl()):0;
        for (ell=out.starting_level+1;ell<out.levels.size();ell++){
            lambda += out.levels[ell].M > 0 ?
                std::sqrt(out.levels[ell].Wl() * out.levels[ell].dVl()):0;
        }
        lambda /= tol2;

        ell=out.starting_level;
        Ml[ell] = (out.levels[ell].M == 0) ? M0 :
            static_cast<uint64_t>(std::ceil( lambda * std::sqrt(out.levels[ell].fVl()/out.levels[ell].Wl())));
        for (ell=out.starting_level+1;ell<out.levels.size();ell++){
            Ml[ell] = (out.levels[ell].M == 0) ? M0 :
                static_cast<uint64_t>(std::ceil( lambda * std::sqrt(out.levels[ell].dVl()/out.levels[ell].Wl())));
            //std::cout << ell << " -> " << Ml[ell] << std::endl;
        }
        // TODO: Maybe do not save it in an array?
        return Ml;
    }

    static mlmc_output_t run(mlmc_input_t& in){
        std::clock_t start = std::clock();

        mlmc_output_t out;
        double sqrt_2_inv = 1./std::sqrt(2);
        // Start with 2 levels
        out.levels.resize(1);
        uint32_t total_itrs = static_cast<uint32_t>(-(std::log2(in.targetTOL)-std::log2(in.startTOL)));
        out.TOL = in.targetTOL * (1 << total_itrs);
        total_itrs *= 2;  // Since we are multiplying by sqrt2

        for (uint32_t itr=0;itr<=total_itrs;itr++){
            out.starting_level = better_starting_level(out, in.starting_level_coeff);

            while (true){
                std::vector<uint64_t> Ml = get_optimal_Ml(in.theta * POW2(out.TOL)/ in.Ca,
                                                          in.M0, out);
                for (uint32_t ell=0;ell<out.levels.size();ell++) {
                    while (out.levels[ell].M < Ml[ell]){
                        // Add missing samples
                        uint64_t M = Ml[ell] - out.levels[ell].M;
                        if (in.verbose)
                            std::cout << "# Sampling " << M << " of " << ell << std::endl;
                        out.levels[ell] += in.sample_level(ell, M);
                    }
                }

                out.stat_error = std::sqrt(in.Ca * out.total_variance());
                // std::cout << "Stat error " << std::setprecision(15) <<
                //     POW2(out.stat_error) << ", " << in.theta * POW2(out.TOL) << std::endl;
                if (POW2(out.stat_error) >= in.theta * POW2(out.TOL))
                    continue;   // We have better variance estimates, so redo this step

                out.bias_est = in.est_bias(out);
                if (POW2(out.bias_est) >= (1-in.theta) * POW2(out.TOL)){
                    out.levels.resize(out.levels.size()+1);
                    continue;
                }
                // Iteration done
                if (in.verbose)
                    std::cout << out.to_string();
                break;
            }
            out.TOL *= sqrt_2_inv;   // Double work of a Monte Carlo sampler
        }
        double total_time = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
        if (in.verbose)
            std::cout << "# Total time taken: " << total_time << std::endl;
        return out;
    }
};


typedef mlmc_t<real_t, 4> mlmc_real;

#endif // __MLMC_HPP__
