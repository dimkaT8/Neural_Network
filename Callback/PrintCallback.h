#pragma once

#include <iostream>
#include "Callback.h"

namespace NNE
{


class VerboseCallback final : public Callback
{
    public:
        void post_training_batch(const Network* net, const Matrix& x, const Matrix& y)
        {
            const Scalar loss = net->get_output()->loss();
            std::cout << "[Epoch " << _epoch_id << ", batch " << _batch_id << "] Loss = "
                      << loss << std::endl;
        }

        void post_training_batch(const Network* net, const Matrix& x,const IntegerVector& y)
        {
            Scalar loss = net->get_output()->loss();
            std::cout << "[Epoch " << _epoch_id << ", batch " << _batch_id << "] Loss = "
                      << loss << std::endl;
        }
};


} 