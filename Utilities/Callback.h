#pragma once

#include <iostream>
#include "/home/dimka/Eigen/Core"
#include "InitScalar.h"

namespace NNE
{

class Network;
class  Callback
{
    private:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef Eigen::RowVectorXi IntegerVector;

        int _nbatch{};   // Общее количество партий
        int _batch_id{}; // Индекс текущего мини-пакета (0, 1, ..., _nbatch-1)
        int _nepoch{};   // Общее количество эпох (один прогон на всем наборе данных) в процессе обучения
        int _epoch_id{}; // Индекс текущей эпохи (0, 1, ..., _nepoch-1)

    public:

        Callback();
        ~Callback();

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