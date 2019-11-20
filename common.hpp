#ifndef __COMMON_HPP__
#define __COMMON_HPP__

#define EXTRA_INNER_DIAGNOSITCS
typedef unsigned int uint32_t;

static_assert(sizeof(uint64_t) == sizeof(unsigned long long),
              "Both uint64_t and unsigned long long should be 8 bytes!");
#define FREQ_ASSERT(x) assert(x)

#ifdef REAL_DOUBLE
typedef double real_t;
#else
typedef float real_t;
#endif

typedef double work_t;
typedef double sum_t;
typedef real_t prob_t;

#if defined __NVCC__
#include "par_nvcc.hpp" 
#else // __NVCC__
#include "par_omp.hpp"
#endif // __NVCC__

#ifdef EXTRA_INNER_DIAGNOSITCS
const int c_inner_stats_count = 6;
#else
const int c_inner_stats_count = 1;
#endif


// NOTE: sample_output is almost the same as mlmc::mc_diff_output_t Except that since it will
// be used in python, hence we should stick to a C interface. Also, it allows for dynamic
// number of moments. Finally it includes some inner_stats
struct sample_output {
    sample_output(uint32_t fsize, uint32_t dsize=0, uint32_t phi_size=1) : M(0), total_work(0) {
        fsums = new sum_t[phi_size*fsize];
        memset(fsums, 0, sizeof(sum_t)*phi_size*fsize);

        diffsums = new sum_t[phi_size*dsize];
        memset(diffsums, 0, sizeof(sum_t)*phi_size*dsize);
        memset(inner_stats, 0, sizeof(inner_stats));
    }
    ~sample_output(){
        delete[] fsums;
        delete[] diffsums;
    }
    uint64_t M;
    sum_t *fsums;     // Size is phi_size x moments_count
    sum_t *diffsums;  // Size is phi_size x moments_count
    work_t total_work;
    uint64_t inner_stats[c_inner_stats_count];
};

class nested_sampler_params;

//// DATA, extern's are defined in common.cpp
extern PAR_DEVICE_CONST nested_sampler_params gc_sampler_data;
extern PAR_DEVICE_CONST uint32_t gc_phi_size;    // TODO: Not ideal to have it here
class nestedmc_data;
extern nestedmc_data* g_p_nestedmc_data;

extern "C" PAR_HOST  void init_nestedmc_sampler(uint32_t cores,
                                                const nested_sampler_params *params);
extern "C" PAR_HOST void cleanup_nestedmc_sampler();

template<class T>
class span {
public:
     // This doesn't work with PAR_DEVICE_CONST const
    // inline PAR_BOTH span(const span& rhs) : _data(rhs._data), _size(rhs._size) {}
    // inline PAR_BOTH span() : _data(0), _size(0) {}
    // inline PAR_BOTH span(T* d, size_t s) : _data(d), _size(s) {}

    inline PAR_BOTH T& operator[](size_t index) { return _data[index]; }
    inline PAR_BOTH const T& operator[](size_t index) const { return _data[index];}
    inline PAR_BOTH size_t size() const { return _size; };
    inline PAR_BOTH T* begin() { return _data; }
    inline PAR_BOTH const T* begin() const { return _data; }

    inline PAR_BOTH T* end() { return _data+_size; }
    inline PAR_BOTH const T* end() const { return _data+_size; }

    inline PAR_BOTH void reset() { _data=0;_size=0; }

    inline PAR_BOTH const span<T>& copy(const span<T> &rhs){
        assert(_size == rhs._size);
        if (rhs._data != _data) {
            // std::copy(rhs._data, rhs._data+_size, _data);
            auto first = rhs._data;
            auto result = _data;
            auto last = rhs._data+_size;
            while (first!=last) {
                *result = *first;
                ++result; ++first;
            }
        }
        return *this;
    }

    static PAR_BOTH span<T> create(T* d, size_t s) {
        span<T> ret;
        ret._data = d;
        ret._size = s;
        return ret;
    }

protected:
    T* _data;
    size_t _size;
};

template<class T>
class const_span {
public:
    //inline PAR_DEVICE const_span() : _data(0), _size(0) {}   // This doesn't work with DEVICE const
    inline PAR_BOTH const_span(const span<T>& rhs) : _data(rhs.begin()), _size(rhs.size()) {}
    inline PAR_BOTH const_span(const T* d, size_t s) : _data(d), _size(s) {}
    inline PAR_BOTH const T& operator[](size_t index) const { return _data[index];}
    inline PAR_BOTH size_t size() const { return _size; };
    inline PAR_BOTH const T* begin() const { return _data; }
    inline PAR_BOTH const T* end() const { return _data+_size; }
    inline PAR_BOTH void reset() { _data=0;_size=0; }
protected:
    const T* _data;
    size_t _size;
};
typedef span<real_t> array;
typedef const_span<real_t> const_array;


//template<typename T> inline PAR_DEVICE PAR_HOST T MOD2(T a, T b){return a & (b-1);}
template<typename T> inline PAR_DEVICE PAR_HOST T POW2(T a){return a*a;}
template<typename T, typename T2> inline PAR_BOTH T MAX(T a, T2 b)
{return ((a)>(b)?(a):(b));}
template<typename T, typename T2> inline PAR_BOTH T MIN(T a, T2 b)
{return ((a)<(b)?(a):(b));}
template <typename T, typename T2>
inline PAR_BOTH bool is_eq(T a, T2 b, T rel_tol=1e-09, T abs_tol=0.0){
    return std::abs(a-b) <= MAX(rel_tol * MAX(std::abs(a), std::abs(b)), abs_tol);
}


template <typename T>
PAR_DEVICE T uint_pow(T x, T n)
{ // Does not work for x==0
    if (n==0) return 1;
    if ((x & (x-1)) == 0) { // Power of two
        return static_cast<T>(1) << (n*clog2(x));
    }
    // TODO: Think of way to optimise
    T r;
    for (r=1;n--;r*=x);
    return r;
}


template <class InputIterator>
inline PAR_DEVICE uint32_t rand_int_cdf(real_t u,
                                      InputIterator cdf_begin,
                                      InputIterator cdf_end){
    // Binary search with a fixed number of steps
    //uint32_t t = static_cast<uint32_t>(u*cdf.size());
    //return (t == cdf.size())?t-1:t;
    FREQ_ASSERT(cdf_end>cdf_begin);
    uint32_t pos[] = {0, static_cast<uint32_t>(cdf_end-cdf_begin)};
    if (pos[1] == 1) return 0;  /// Because CLZ(0) is 1
    uint32_t ceil_log2 = clog2(pos[1]);
    for (uint32_t i=0;i<ceil_log2;i++) {
        uint32_t mid = (pos[0] + pos[1]) / 2;
        pos[mid > 0 && u < *(cdf_begin + mid)] = mid;
    }
    return pos[0];
}

template <class InputIterator>
inline PAR_DEVICE size_t rand_int_pdf(real_t u,
                                      InputIterator pdf_begin,
                                      InputIterator pdf_end){
    FREQ_ASSERT(pdf_end > pdf_begin);
    real_t p=0;
    size_t ret = 0;
    size_t size = pdf_end - pdf_begin;
    for (size_t i=0;i < size;i++) {
        real_t c = *(pdf_begin + i);
        if (c <= 0) continue;  // Skip, this cannot be selected
        p += c;
        ret = i;
        if (u <= p)
            break;
    }
    return ret;
}


const real_t SQRT_2 = 1.414213562373095;
const real_t SQRT_2PI = 2.506628274631;

inline PAR_BOTH real_t norm_cdf(double x) { return 0.5 * erfc(-x / SQRT_2); };
inline PAR_BOTH real_t norm_cdf(float x) { return 0.5 * erfcf(-x / SQRT_2); };
inline PAR_BOTH real_t norm_pdf(real_t x) { return std::exp(-0.5*(x*x)) / SQRT_2PI; };

PAR_BOTH real_t norm_cdf_inv(real_t x);

#endif // __COMMON_HPP__
