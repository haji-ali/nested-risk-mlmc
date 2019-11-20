#ifndef __PAR_OMP_HPP__
#define __PAR_OMP_HPP__

#if defined(_OPENMP)
#include <omp.h>
#elif !defined(__NVCC__)
#pragma message("compiling WITHOUT openmp")
#endif

#include "Random123/threefry.h"
#include "Random123/boxmuller.hpp"
#include "Random123/uniform.hpp"

// No device memory
#define PAR_BOTH_MALLOC(h, d, s) h = (char*)malloc(s); d=h
#define PAR_BOTH_MEMCPY(h, d, s) assert(h==d) // Not needed since they should be the same pointer
#define PAR_DEVICE_MEMZERO(d, s) memset(d, 0,  s)
#define PAR_BOTH_FREE(h, d) assert(h==d); free(h)

template<typename T>
inline void PAR_DEVICE_MALLOC(T*& d, size_t s){
    d = reinterpret_cast<T*>(malloc(s));
}

#define PAR_DEVICE_FREE(d) if (d) free(d)
#define PAR_SET_CONST(dev, host) {dev = host;}
#define PAR_GET_CONST(host, dev) {host = dev;}

#if defined(_OPENMP)
// https://stackoverflow.com/questions/45762357/how-to-concatenate-strings-in-the-arguments-of-pragma
#define DO_PRAGMA_(x) _Pragma (#x)
#define DO_PRAGMA(x) DO_PRAGMA_(x)
#define OMP_PRAGMA(x) DO_PRAGMA(omp x)
#define PAR_CORE_IDX (omp_get_thread_num())
#else    /// defined(_OPENMP)
#define OMP_PRAGMA(x) // Ignore
#define PAR_CORE_IDX (0)
#endif   /// defined(_OPENMP)

const uint32_t c_warp_size = 1;
const uint32_t c_thread_per_block = 1;

#define PAR_DEVICE_CONST
#define PAR_DEVICE
#define PAR_GLOBAL
#define PAR_HOST
#define PAR_BOTH PAR_HOST PAR_DEVICE

#define PAR_CLZ(x) __builtin_clz((x))


inline PAR_DEVICE uint32_t clog2(uint32_t x)
{
    return 32 - __builtin_clz(x-1);
}

inline PAR_DEVICE uint64_t clog2(uint64_t x)
{
    return 64 - __builtin_clzl(x-1);
}

template<class T>
inline PAR_HOST T* parDuplicateInDevice(const T* begin, size_t count){
    T* d;
    size_t s = (count) * sizeof(T);
    PAR_DEVICE_MALLOC(d, s);
    memcpy(d, begin, s);
    return d;
}


class random_state {
public:
    random_state(uint64_t _seed=0, uint64_t _core=0) : is_randn_saved(false), cached(0) {
        seed(_seed, _core);
    }

    real_t randn(){
        if (is_randn_saved){
            is_randn_saved = false;
            return saved_randn;
        }
        r123::double2 d2 = r123::boxmuller(rand(), rand());
        is_randn_saved = true;
        saved_randn = static_cast<real_t>(d2.y);
        return static_cast<real_t>(d2.x);
    }
    real_t randu(){
        return r123::u01<real_t>(rand());
    }

    uint64_t rand(){
        if (cached <= 0) {
            cur_rand = rng(ctr, key);
            ctr.incr();
            cached = cur_rand.size();
        }
        return cur_rand[--cached];
    }

    void seed(uint64_t _seed, uint64_t _core){
        is_randn_saved = false;
        cached = 0;
        assert(key.size() == 2);
        key[0] = _seed;
        key[1] = _core;
        ctr.fill(0);
    }
    real_t last_rand_u;
private:
    typedef r123::Threefry2x64 CBRNG;
    CBRNG rng;
    bool is_randn_saved;
    real_t saved_randn;

    CBRNG::ctr_type ctr;
    CBRNG::key_type key;
    CBRNG::ctr_type cur_rand;
    int cached;
};
typedef random_state RANDSTATE;

template<typename T>
inline T atomicAdd(T* address, T val){
    *address += val;
    return *address;
}


#endif // __PAR_OMP_HPP__
