#pragma once

#include "/home/dimka/Eigen/Core"
#include "InitScalar.h"

// abstract base class Callback
namespace  NNE
{

class Network;
class Callback
{
protected:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef Eigen::RowVectorXi IntegerVector;

public:
        int _nbatch;   // Общее количество партий
        int _batch_id; // Индекс текущего мини-пакета (0, 1, ..., m_nbatch-1)
        int _nepoch;   // Общее количество эпох (один прогон на всем наборе данных) в процессе обучения
        int _epoch_id; // Индекс текущей эпохи (0, 1, ..., m_nepoch-1)

        Callback() : _nbatch(0), _batch_id(0), _nepoch(0), _epoch_id(0) {}

        virtual ~Callback() {}

        // После обучения мини-партии
        virtual void post_training_batch(const Network* net, const Matrix& x,const Matrix& y) {}
        virtual void post_training_batch(const Network* net, const Matrix& x,const IntegerVector& y) {}
};
}