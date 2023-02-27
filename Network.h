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
        void check_unit_sizes() const;

        // Пусть каждый слой вычисляет свой вывод
        void forward(const Matrix& input)
        {
            const int nlayer = num_layers();

            if (nlayer <= 1) return;

            for (int i = 1; i < nlayer; i++)
                {
                  if (_layers[i]->in_size() != _layers[i - 1]->out_size())
                     {
                        throw std::invalid_argument("[class Network]: Unit sizes do not match ");
                     }
               } 
        }

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
        /// Конструктор по умолчанию, который создает пустую нейронную сеть
        Network() : _default_rng(1), _rng(_default_rng),_output(NULL),
                    _default_callback(),_callback(&_default_callback) {}

        /// Конструктор с предоставленным пользователем генератором случайных чисел
        /// \param rng Предоставленный пользователем объект генератора случайных чисел, который наследует
        ///           из class RNG по умолчанию.
        Network(RNG& rng) : _default_rng(1), _rng(rng), _output(NULL),
                _default_callback(), _callback(&_default_callback) {}

        /// Деструктор, который освобождает добавленные скрытые слои и выходной слой
        ~Network()
        {
            for (int i = 0; i < num_layers(); i++) delete _layers[i];
            if(_output) delete _output;
        }

        /// Добавьте скрытый слой в нейронную сеть
        /// **ПРИМЕЧАНИЕ**: указатель будет обработан и освобожден
        /// в сетевой объект, поэтому не удаляйте его вручную.
        void add_layer(Layer* layer) { _layers.push_back(layer);}

        /// Установите выходной слой нейронной сети
        /// **ПРИМЕЧАНИЕ**: указатель будет обработан и освобожден
        /// сетевой объект, поэтому не удаляйте его вручную.
        void set_output(Output* output)
        {
            if (_output) delete _output;
            _output = output;
        }

        /// Количество скрытых слоев в сети
        int num_layers() const{ return _layers.size();}

        /// Получить список скрытых слоев сети
        std::vector<const Layer*> get_layers() const
        {
            std::vector<const Layer*> layers(num_layers());
            std::copy(_layers.begin(), _layers.end(), layers.begin());
            return layers;
        }

        /// Получить выходной слой
        const Output* get_output() const {return _output;}

        /// Устанавливаем callback-функцию, которую можно вызывать во время подбора модели
        void set_callback(Callback& callback){_callback = &callback;}

        /// Установить функцию тихого обратного вызова по умолчанию
        void set_default_callback(){_callback = &_default_callback;}

        /// Инициализируем параметры слоя в сети, используя нормальное распределение
        /// \param mu    Среднее значение нормального распределения.
        /// \param sigma Стандартное отклонение нормального распределения.
        /// \param seed  Установить случайное начальное число %RNG, если `seed > 0`, иначе
        ///              используем текущее случайное состояние.
        void init(const Scalar& mu = Scalar(0), const Scalar& sigma = Scalar(0.01),
                   int seed = -1)
        {
            check_unit_sizes();
            if (seed > 0) _rng.seed(seed);

            for (int i = 0; i < num_layers(); i++) _layers[i]->init(mu, sigma, _rng);
        }

        /// Получить сериализованные параметры слоя
        std::vector< std::vector<Scalar> > get_parameters() const
        {
            std::vector< std::vector<Scalar> > res;
            res.reserve(num_layers());

            for (int i = 0; i < num_layers(); i++) res.push_back(_layers[i]->get_parameters());

            return res;
        }

        /// Задаем параметры слоя
        void set_parameters(const std::vector< std::vector<Scalar> >& param)
        {
            if (static_cast<int>(param.size()) != num_layers())
                throw std::invalid_argument("[class Network]: Parameter size does not match");

            for (int i = 0; i < num_layers(); i++) _layers[i]->set_parameters(param[i]);
        }

        /// Получить сериализованные производные параметров слоя
        std::vector< std::vector<Scalar> > get_derivatives() const
        {
            std::vector< std::vector<Scalar> > res;
            res.reserve(num_layers());

            for (int i = 0; i < num_layers(); i++) res.push_back(_layers[i]->get_derivatives());
            return res;
        }

        /// Инструмент отладки для проверки градиентов параметров
        template <typename TargetType>
        void check_gradient(const Matrix& input, const TargetType& target, int npoints,
                            int seed = -1)
        {
            if(seed > 0) _rng.seed(seed);

            this->forward(input);
            this->backprop(input, target);
            std::vector< std::vector<Scalar> > param = this->get_parameters();
            std::vector< std::vector<Scalar> > deriv = this->get_derivatives();
            const Scalar eps = 1e-5;
            const int nlayer = deriv.size();

            for (int i = 0; i < npoints; i++)
            {
                //Произвольный выбор слоя
                const int layer_id = int(_rng.rand() * nlayer);
                // Произвольный выбор параметра, обратите внимание, что некоторые слои могут не иметь параметров
                const int nparam = deriv[layer_id].size();

                if (nparam < 1) continue;
                const int param_id = int(_rng.rand() * nparam);
                // Немного турбулизировать параметр
                const Scalar old = param[layer_id][param_id];
                param[layer_id][param_id] -= eps;
                this->set_parameters(param);
                this->forward(input);
                this->backprop(input, target);
                const Scalar loss_pre = m_output->loss();
                param[layer_id][param_id] += eps * 2;
                this->set_parameters(param);
                this->forward(input);
                this->backprop(input, target);
                const Scalar loss_post = _output->loss();
                const Scalar deriv_est = (loss_post - loss_pre) / eps / 2;
                std::cout << "[layer " << layer_id << ", param " << param_id <<
                          "] deriv = " << deriv[layer_id][param_id] << ", est = " << deriv_est <<
                          ", diff = " << deriv_est - deriv[layer_id][param_id] << std::endl;
                param[layer_id][param_id] = old;

            }
            // Восстановить исходные параметры
            this->set_parameters(param);
        }

        /// Собираем модель на основе заданных данных
        /// \param opt        Объект, наследуемый от класса Optimizer, указывающий используемый алгоритм оптимизации.
        /// \param x          Предикторы. Каждый столбец представляет собой наблюдение.
        /// \param y          Переменная ответа. Каждый столбец представляет собой наблюдение.
        /// \param batch_size Размер мини-пакета.
        /// \param epoch      Количество эпох обучения.
        /// \param seed       Установить случайное начальное число %RNG, если `seed > 0`, иначе
        ///                   используем текущее случайное состояние.
        template <typename DerivedX, typename DerivedY>
        bool fit(Optimizer& opt, const Eigen::MatrixBase<DerivedX>& x,
                 const Eigen::MatrixBase<DerivedY>& y,
                 int batch_size, int epoch, int seed = -1)
        {
            // Мы не используем PlainObjectX напрямую, так как он может быть разбит на строки, если x передается как mat.transpose()
            // Мы хотим, чтобы XType и YType были разделены по столбцам
            typedef typename Eigen::MatrixBase<DerivedX>::PlainObject PlainObjectX;
            typedef typename Eigen::MatrixBase<DerivedY>::PlainObject PlainObjectY;
            typedef Eigen::Matrix<typename PlainObjectX::Scalar, PlainObjectX::RowsAtCompileTime, PlainObjectX::ColsAtCompileTime>
            XType;
            typedef Eigen::Matrix<typename PlainObjectY::Scalar, PlainObjectY::RowsAtCompileTime, PlainObjectY::ColsAtCompileTime>
            YType;

            if(num_layers() <= 0) return false;
            // Сброс оптимизатора
            opt.reset();
            // Создаем перетасованные мини-пакеты
            if(seed > 0) _rng.seed(seed);

            std::vector<XType> x_batches;
            std::vector<YType> y_batches;
            const int nbatch = internal::create_shuffled_batches(x, y, batch_size, _rng,
                               x_batches, y_batches);
            // Настройте параметры обратного вызова
            _callback->_nbatch = nbatch;
            _callback->_nepoch = epoch;

            // Итерации по всему набору данных
            for (int k = 0; k < epoch; k++)
            {
                _callback->_epoch_id = k;

                // Тренируйтесь на каждой мини-партии
                for (int i = 0; i < nbatch; i++)
                {
                    _callback->_batch_id = i;
                    _callback->pre_training_batch(this, x_batches[i], y_batches[i]);
                    this->forward(x_batches[i]);
                    this->backprop(x_batches[i], y_batches[i]);
                    this->update(opt);
                    _callback->post_training_batch(this, x_batches[i], y_batches[i]);
                }
            }

            return true;
        }

        /// Используйте подобранную модель, чтобы делать прогнозы
        ///
        /// \param x Предикторы. Каждый столбец представляет собой наблюдение.
        Matrix predict(const Matrix& x)
        {
            if (num_layers() <= 0) return Matrix();

            this->forward(x);
            return _layers[num_layers() - 1]->output();
        }
    };
}