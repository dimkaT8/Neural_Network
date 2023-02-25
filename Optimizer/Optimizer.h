#pragma once

#include "/home/dimka/Eigen/Core"
#include "InitScalar.h"
#include <map>

namespace NNE
{
class Optimizer
{
protected:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Vector::ConstAlignedMapType ConstAlignedMapVec;
    typedef Vector::AlignedMapType AlignedMapVec;

public:
    ~Optimizer() = default;

    /// Сбросьте оптимизатор, чтобы очистить всю историческую информацию
    virtual void reset() = 0;

    /// Обновите вектор параметров, используя его градиент
    ///
    /// Предполагается, что адреса памяти `dvec` и `vec` не совпадают
    /// меняются в процессе обучения. Это используется для реализации оптимизации
    /// алгоритмы, которые имеют "память".
    ///
    /// \param dvec Градиент параметра. Только для чтения
    /// \param vec  Ввод ,текущий вектор параметров. На выходе,
    ///             обновленные параметры.
    virtual void update(ConstAlignedMapVec& dvec, AlignedMapVec& vec) = 0;
};
}