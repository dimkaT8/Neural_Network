#pragma once

#include "/home/dimka/Eigen/Core"
#include <stdexcept>
#include "InitScalar.h"

namespace NNE
{

/// Интерфейс выходного слоя нейросетевой модели. Выход
/// слой — это специальный слой, который связывает последний скрытый слой с
/// целевая переменная ответа.
///
class Output
{
    protected:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        typedef Eigen::RowVectorXi IntegerVector;

    public:
        virtual ~Output() {}

        // Проверьте формат целевых данных, например. в задачах классификации
        // целевые данные должны быть бинарными (либо 0, либо 1)
        virtual void check_target_data(const Matrix& target) {}

        // Другой тип целевых данных, где каждый элемент является меткой класса.
        // Эта версия может оказаться непригодной для задач регрессии, поэтому по умолчанию
        // мы вызываем исключение
        virtual void check_target_data(const IntegerVector& target)
        {
            throw std::invalid_argument("[class Output]: This output type cannot take class labels as target data");
        }

        // Комбинация прямого этапа и обратного этапа для выходного слоя
        // Вычисленная производная ввода должна храниться в этом слое и может быть извлечена с помощью
        // функция backprop_data()
        virtual void evaluate(const Matrix& prev_layer_data, const Matrix& target) = 0;

        // Другой тип целевых данных, где каждый элемент является меткой класса.
        // Эта версия может оказаться непригодной для задач регрессии, поэтому по умолчанию
        // мы вызываем исключение
        virtual void evaluate(const Matrix& prev_layer_data,
                              const IntegerVector& target)
        {
            throw std::invalid_argument("[class Output]: This output type cannot take class labels as target data");
        }

        // Производная входа этого слоя, которая также является производной
        // вывода предыдущего слоя
        virtual const Matrix& backprop_data() const = 0;

        // Вернуть значение функции потерь после оценки
        // Можно предположить, что эта функция вызывается после вычисления(), так что она может использовать
        // промежуточный результат для экономии вычислений
        virtual Scalar loss() const = 0;

        // Вернуть тип выходного слоя. Он используется для экспорта модели NN.
        virtual std::string output_type() const = 0;
};


} 