#ifndef __GBM_APPROX_HPP__
#define __GBM_APPROX_HPP__
template <class T>
struct Accessor {
    PAR_BOTH Accessor(const T &t) : inner(t){
    }
    PAR_BOTH inline const T& operator[](int) const      { return inner; }
    T inner;
};

template <class T>
struct Accessor<T*> {
    PAR_BOTH Accessor(const T *t) : inner(t) { }
    PAR_BOTH inline const T& operator[](int idx) const      { return inner[idx]; }
    const T *inner;
};

template <bool with_coarse=true, class A, class B, class T, class F>
inline PAR_DEVICE real_t simulate_stock_dt(F randn,
                                           T *S,   // In: S0, out ST
                                           uint64_t N,
                                           A _mu,
                                           B _sigma,
                                           real_t t,
                                           size_t size,
                                           bool milstein,
                                           uint32_t coarse_step=2)  {
    auto mu = Accessor<A>(_mu);
    auto sigma = Accessor<B>(_sigma);
    real_t W = 0;
    real_t dt = t/N;
    real_t sqrt_dt = std::sqrt(dt);
    real_t dW_c = 0;
    real_t dt_c = t/(N/coarse_step);
    FREQ_ASSERT(not with_coarse or N % coarse_step == 0);
    for (uint64_t n=0;n<N;n++) {
        real_t dW = sqrt_dt * randn();
        W += dW;
        // Fine step
        for (uint32_t i=0;i<size;i++) {
            real_t m = milstein ? (0.5*POW2(sigma[i]) * (POW2(dW)-dt)) : 0;
            S[i] += S[i] * (mu[i]*dt + sigma[i]*dW + m);
        }
        if (with_coarse){
            // Coarse step
            dW_c += dW;
            if (((n+1) % coarse_step) == 0){
                for (uint32_t i=0;i<size;i++) {
                    real_t m = milstein ? (0.5*POW2(sigma[i]) * (POW2(dW_c)-dt_c)) : 0;
                    S[i+size] += S[i+size] * (mu[i]*dt_c + sigma[i]*dW_c + m);
                }
                dW_c = 0;
            }
        }
    }
    return W;
}
#endif // __GBM_APPROX_HPP__
