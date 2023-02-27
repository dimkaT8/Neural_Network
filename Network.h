#pragma once

#include <vector>
#include <stdexcept>
#include "InitScalar.h"
#include "Utilities/RNG.h"
#include "Layer/Layer.h"
#include "Utilities/Callback.h"
#include "Output/Output.h"

namespace NNE
{
    class Network
    {
     private:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef Eigen::RowVectorXi IntegerVector;
        typedef std::map<std::string, int> MetaInfo;

        RNG                 _default_rng;      // Встроенный ГСЧ
        RNG&                _rng;              // Ссылка на ГСЧ, предоставленная пользователем,
                                               // иначе ссылка на m_default_rng
        std::vector<Layer*> _layers;           // Указатели на скрытые слои
        Output*             _output;           // Выходной слой
        Callback            _default_callback; // Функция обратного вызова по умолчанию
        Callback*           _callback;         // Указывает на предоставленную пользователем функцию обратного вызова,
                                               // иначе указывает на _default_callback

        //Проверьте размеры слоев
        void check_unit_sizes() const{}

        // Пусть каждый слой вычисляет свой вывод
        void forward(const Matrix& input){}

        // Пусть каждый слой вычисляет свои градиенты параметров
        // цель имеет две версии: Matrix and RowVectorXi
        // Версия RowVectorXi используется в задачах классификации, где каждый
        // элемент является меткой класса
        template <typename TargetType>
        void backprop(const Matrix& input, const TargetType& target){}

        // Обновить параметры
        void update(Optimizer& opt){}

        // Получите метаинформацию о сети, используемую для экспорта модели NN.
        MetaInfo get_meta_info() const{}

     public:


    };
}