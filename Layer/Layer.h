#pragma once

#include "/home/dimka/Eigen/Core"
#include "InitScalar.h"
#include "Utilities/RNG.h"
#include "Optimizer/Optimizer.h"
#include <vector>
#include <map>

namespace NNE
{
class Layer
{
protected:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef std::map<std::string, int> Info;

    const int _in_size;  // Размер входных единиц
    const int _out_size; // Размер выходных единиц
public:
    Layer(const int in_size, const int out_size) :
           _in_size(in_size),_out_size(out_size)
    {}

    virtual ~Layer() {}
    int in_size() const { return _in_size;}
    int out_size() const { return _out_size;}
    /// \param ndm    Среднее нормального распределения.
    /// \param sigma  Стандартное отклонение нормального распределения.
    /// \param rng    Генератор случайных чисел типа RNG.
    virtual void init(const Scalar& ndm, const Scalar& sigma, RNG& rng) = 0;
    virtual void init() = 0;
    /// Вычисляет выходные данные этого слоя.
    virtual void forward(const Matrix& layer_data) = 0;
    /// Выходные значения этого слоя
    virtual const Matrix& output() const = 0;
    /// Вычислить градиенты параметров и входных единиц, используя обратное распространение
    virtual void backprop(const Matrix& prev_layer_data,
                          const Matrix& next_layer_data) = 0;
    /// Градиент входных единиц этого слоя
    virtual const Matrix& backprop_data() const = 0;
    /// Обновить параметры после обратного распространения
    /// \param opt Используемый алгоритм оптимизации.
    virtual void update(Optimizer& opt) = 0;
    /// Получить значения параметров
    virtual std::vector<Scalar> get_parameters() const = 0;
    /// Установите значения параметров слоя
    virtual void set_parameters(const std::vector<Scalar>& param) {};
    /// Получить значения градиента параметров
    virtual std::vector<Scalar> get_derivatives() const = 0;
    virtual std::string layer_type() const = 0;
    virtual std::string activation_type() const = 0;
    virtual void fill_info(Info& map, int index) const = 0;
};

}