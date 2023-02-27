#pragma once

#include "Output.h"
#include "InitScalar.h"

namespace NNE
{


/// Выходной слой регрессии с использованием критерия среднеквадратичной ошибки (MSE)
///
class RegressionMSE final: public Output
{
    private:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

        Matrix m_din;  // Производная входа этого слоя.
                       // Обратите внимание, что вход этого слоя также является выходом предыдущего слоя.

    public:
        void evaluate(const Matrix& prev_layer_data, const Matrix& target)
        {
            // Проверить размер
            const int nobs = prev_layer_data.cols();
            const int nvar = prev_layer_data.rows();

            if ((target.cols() != nobs) || (target.rows() != nvar))
            {
                throw std::invalid_argument("[class RegressionMSE]: Target data have incorrect dimension");
            }

            // Вычислите производную входа этого слоя
            // L = 0.5 * ||yhat - y||^2
            // in = yhat
            // d(L) / d(in) = yhat - y
            m_din.resize(nvar, nobs);
            m_din.noalias() = prev_layer_data - target;
        }

        const Matrix& backprop_data() const
        {
            return m_din;
        }

        Scalar loss() const
        {
            // L = 0.5 * ||yhat - y||^2
            return m_din.squaredNorm() / m_din.cols() * Scalar(0.5);
        }

        std::string output_type() const
        {
            return "RegressionMSE";
        }
};


}

