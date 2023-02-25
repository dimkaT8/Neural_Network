#pragma once

#include "InitScalar.h"

namespace NNE
{

class RNG
{
private:
    const unsigned int _factor;
    const unsigned long _max;
    long _rand;

    inline long long_rand(long kernel)
    {
        unsigned long lo, hi;
        lo = _factor * (long)(kernel & 0xFFFF);
        hi = _factor * (long)((unsigned long)kernel >> 16);
        lo += (hi & 0x7FFF) << 16;

        if (lo > _max)
        {
            lo &= _max;
            ++lo;
        }

        lo += hi >> 15;

        if (lo > _max)
        {
            lo &= _max;
            ++lo;
        }

        return (long)lo;
    }

public:
    RNG(unsigned long init_seed) : _factor(16807), _max(2147483647L),
        _rand(init_seed ? (init_seed & _max) : 1)
    {}

    virtual ~RNG() {}

    virtual void seed(unsigned long seed)
    {
        _rand = (seed ? (seed & _max) : 1);
    }

    virtual Scalar rand()
    {
        _rand = long_rand(_rand);
        return Scalar(_rand) / Scalar(_max);
    }
};

}