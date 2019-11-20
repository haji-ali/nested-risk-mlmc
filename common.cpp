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

// Declared in nested_sampler.hpp
PAR_DEVICE_CONST nested_sampler_params gc_sampler_data;
PAR_DEVICE_CONST uint32_t gc_phi_size;    // TODO: Not ideal to have it here
nestedmc_data* g_p_nestedmc_data = 0;


extern "C" PAR_HOST  void init_nestedmc_sampler(uint32_t cores,
                                                const nested_sampler_params *params) {
    PAR_SET_CONST(gc_sampler_data, *params);
    g_p_nestedmc_data = new nestedmc_data(params->fine_moments,
                                          params->diff_moments,
                                          cores, 1);
}

extern "C" PAR_HOST void cleanup_nestedmc_sampler(){
    delete g_p_nestedmc_data;
    g_p_nestedmc_data = 0;
}



//
// Lower tail quantile for standard normal distribution function.
//
// This function returns an approximation of the inverse cumulative
// standard normal distribution function.  I.e., given P, it returns
// an approximation to the X satisfying P = Pr{Z <= X} where Z is a
// random variable from the standard normal distribution.
//
// The algorithm uses a minimax approximation by rational functions
// and the result has a relative error whose absolute value is less
// than 1.15e-9.
//
// Author:      Peter J. Acklam
// Time-stamp:  2003-05-05 05:15:14
// E-mail:      pjacklam@online.no
// WWW URL:     http://home.online.no/~pjacklam
// WWW URL:     https://github.com/rozgo/UE4-DynamicalSystems/blob/master/Source/DynamicalSystems/Private/SignalGenerator.cpp
// An algorithm with a relative error less than 1.15*10-9 in the entire region.
real_t norm_cdf_inv(real_t p)
{
    // Coefficients in rational approximations
    real_t a[] = { -39.696830f, 220.946098f, -275.928510f, 138.357751f, -30.664798f, 2.506628f };
    real_t b[] = { -54.476098f, 161.585836f, -155.698979f, 66.801311f, -13.280681f };
    real_t c[] = { -0.007784894002f, -0.32239645f, -2.400758f, -2.549732f, 4.374664f, 2.938163f };
    real_t d[] = { 0.007784695709f, 0.32246712f, 2.445134f, 3.754408f };

    // Define break-points.
    real_t plow = 0.02425f;
    real_t phigh = 1 - plow;

    // Rational approximation for lower region:
    if ( p < plow ) {
        real_t q = std::sqrt( -2 * std::log( p ) );
        return ( ( ( ( ( c[ 0 ] * q + c[ 1 ] ) * q + c[ 2 ] ) * q + c[ 3 ] ) * q + c[ 4 ] ) * q + c[ 5 ] ) /
        ( ( ( ( d[ 0 ] * q + d[ 1 ] ) * q + d[ 2 ] ) * q + d[ 3 ] ) * q + 1 );
    }

    // Rational approximation for upper region:
    if ( phigh < p ) {
        real_t q = std::sqrt( -2 * std::log( 1 - p ) );
        return -( ( ( ( ( c[ 0 ] * q + c[ 1 ] ) * q + c[ 2 ] ) * q + c[ 3 ] ) * q + c[ 4 ] ) * q + c[ 5 ] ) /
        ( ( ( ( d[ 0 ] * q + d[ 1 ] ) * q + d[ 2 ] ) * q + d[ 3 ] ) * q + 1 );
    }

    // Rational approximation for central region:
    {
        real_t q = p - 0.5f;
        real_t r = q * q;
        return ( ( ( ( ( a[ 0 ] * r + a[ 1 ] ) * r + a[ 2 ] ) * r + a[ 3 ] ) * r + a[ 4 ] ) * r + a[ 5 ] ) * q /
        ( ( ( ( ( b[ 0 ] * r + b[ 1 ] ) * r + b[ 2 ] ) * r + b[ 3 ] ) * r + b[ 4 ] ) * r + 1 );
    }
}
