#ifndef __PAR_HPP__
#define __PAR_HPP__

#include <curand_kernel.h>

#define PAR_BOTH_MALLOC(h, d, s) {gpuErrchk(cudaMalloc((void**)&(d), s)); h = (char*)malloc(s);}
#define PAR_BOTH_MEMCPY(h, d, s) gpuErrchk(cudaMemcpy(h, d, s, cudaMemcpyDeviceToHost))
#define PAR_DEVICE_MEMZERO(d, s) gpuErrchk(cudaMemset(d, 0,  s))
#define PAR_BOTH_FREE(h, d) {gpuErrchk(cudaFree(d));free(h);}
#define PAR_DEVICE_MALLOC(d, s) gpuErrchk(cudaMalloc(&(d), (s)))
#define PAR_DEVICE_FREE(d) {if (d) gpuErrchk(cudaFree(d))}
#define PAR_SET_CONST(dev, host) {auto x=(host); gpuErrchk(cudaMemcpyToSymbol(dev, &x, sizeof(x)));}
#define PAR_GET_CONST(host, dev) {gpuErrchk(cudaMemcpyFromSymbol(&host, dev, sizeof(host)));}
#define PAR_IS_WARP_MASTER ((threadIdx.x % c_warp_size) == 0)
#define PAR_CORE_IDX (blockDim.x * blockIdx.x + threadIdx.x)

const uint32_t c_warp_size = 32;

#define PAR_DEVICE_CONST __constant__
#define PAR_DEVICE __device__
#define PAR_GLOBAL __global__
#define PAR_HOST __host__
#define PAR_BOTH PAR_HOST PAR_DEVICE

inline PAR_DEVICE uint32_t clog2(uint32_t x)
{
    return 32 - __clz(x-1);
}

inline PAR_DEVICE uint64_t clog2(uint64_t x)
{
    return 64 - __clzll(x-1);
}


inline PAR_DEVICE uint64_t atomicAdd(uint64_t* address, uint64_t val)
{
    // For some reason, atomicAdd is defined for unsigned long long but not for uint64_t
    typedef unsigned long long ull64;
    FREQ_ASSERT(sizeof(ull64) == sizeof(uint64_t)); // TEMP
    return static_cast<uint64_t>(atomicAdd(reinterpret_cast<ull64*>(address),
                                           static_cast<ull64>(val)));
}


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s (%s:%d)\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
class random_state : public curandStateXORWOW_t {
public:
    inline PAR_DEVICE real_t randu(){
#if defined REAL_DOUBLE
        return curand_uniform_double(this);
#else
        return curand_uniform(this);
#endif
    }
    inline PAR_DEVICE real_t randn(){
#if defined REAL_DOUBLE
        return curand_normal_double(this);
#else
        return curand_normal(this);
#endif
    }

    inline PAR_DEVICE void seed(uint64_t _seed, uint64_t _core){
        curand_init(_seed, _core, 0, this);
    }
};

typedef random_state RANDSTATE;

#if defined(__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ >= 9
template<typename T>
inline PAR_DEVICE T warpGet(T val, int lane){
    return __shfl_sync(0xFFFFFFFF, val, lane, c_warp_size);
}
template<typename T>
inline PAR_DEVICE T warpDown(T val, int delta){
    return __shfl_down_sync(0xFFFFFFFF, val, delta, c_warp_size);
}
template<typename T>
inline PAR_DEVICE T warpXOR(T val, int mask){
    return __shfl_xor_sync(0xFFFFFFFF, val, mask, c_warp_size);
}
#else
template<typename T>
inline PAR_DEVICE T warpGet(T val, int lane){
    int* a = reinterpret_cast<int*>(&val);
    for (int* b=a;b!=a+sizeof(T)/sizeof(int);b++)
        *b = __shfl(*b, lane, c_warp_size);
    return *reinterpret_cast<T*>(a);
}
inline template<typename T>
PAR_DEVICE T warpDown(T val, int delta){
    int* a = reinterpret_cast<int*>(&val);
    for (int* b=a;b!=a+sizeof(T)/sizeof(int);b++){
        *(b) = __shfl_down(*(b), delta, c_warp_size);
    }
    return *reinterpret_cast<T*>(a);
}
inline template<typename T>
PAR_DEVICE T warpXOR(T val, int mask){
    int* a = reinterpret_cast<int*>(&val);
    //for (int* b=a;b!=a+sizeof(T)/sizeof(int);b++)
    for (int* b=a;b!=a+sizeof(T)/sizeof(int);b++)
        *(b) = __shfl_xor(*(b), mask, c_warp_size);
    return *reinterpret_cast<T*>(a);
}
inline template<>
PAR_DEVICE float warpGet(float val, int lane){
    return __shfl(val, lane, c_warp_size);
}
template<>
inline PAR_DEVICE float warpDown(float val, int delta){
    return __shfl_down(val, delta, c_warp_size);
}
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
inline PAR_DEVICE double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                             (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

template<class T>
PAR_HOST T* parDuplicateInDevice(const T* begin, size_t count){
    T* d;
    size_t s = (count) * sizeof(T);
    PAR_DEVICE_MALLOC(d, s);
    gpuErrchk(cudaMemcpy(d, begin, s, cudaMemcpyHostToDevice));
    return d;
}

template<typename T>
PAR_DEVICE T warpSum(T val){
    for (int offset = c_warp_size/2; offset > 0; offset /= 2)
        val += warpDown(val, offset);
    return val;
}
template<typename T>
PAR_DEVICE T warpSum_All(T val){
    for (int mask = c_warp_size/2; mask > 0; mask /= 2)
        val += warpXOR(val, mask);
    return val;
}

#define OMP_PRAGMA(x)

#endif // __PAR_HPP__
