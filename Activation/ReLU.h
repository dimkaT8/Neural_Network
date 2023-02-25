#pragma once

#include "/home/dimka/Eigen/Core"
#include "InitScalar.h"

namespace NNE
{

class ReLU
{
    private:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;

    public:
        // A =  max(Z, 0)
        static inline void activate(const Matrix& Z, Matrix& A)
        {
            A.array() = Z.array().cwiseMax(Scalar(0));
        }

        // J = d_a / d_z = diag(A > 0)
        // G = (A > 0) * F
        static inline void jacobian(const Matrix& Z, const Matrix& A,
                                          const Matrix& F, Matrix& G)
        {
            G.array() = (A.array() > Scalar(0)).select(F, Scalar(0));
        }

        static std::string return_type()
        {
            return "ReLU";
        }
};


}