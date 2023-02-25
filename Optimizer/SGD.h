#pragma once

#include "Optimizer.h"

namespace  NNE
{
class SGD final : public Optimizer
{
private:
    Scalar _lrate;
    Scalar _decay;
public:
    SGD(const Scalar& lrate = Scalar(0.001), const Scalar& decay = Scalar(0)) :
        _lrate(lrate), _decay(decay)
    {}

    ~SGD() = default;

    void update(ConstAlignedMapVec& dvec, AlignedMapVec& vec) override
    {
        vec.noalias() -= _lrate * (dvec + _decay * vec);
    }
};
}