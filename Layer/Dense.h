#include "Layer.h"
#include "Utilities/Random.h"

namespace  NNE
{
template <typename Activation>
class Dense final : public Layer
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Vector::ConstAlignedMapType ConstAlignedMapVec;
    typedef Vector::AlignedMapType AlignedMapVec;

    Matrix _m_weight;  // Весовые параметры, W(in_size -- out_size)
    Vector _v_bias;    // Параметры смещения, b(out_size -- 1)
    Matrix _m_dw;      // Производная весов
    Vector _v_db;      // Производная смещения
    Matrix _m_z;       // Линейный термин, z = W' * in + b
    Matrix _m_a;       // Вывод этого слоя, a = act(z)
    Matrix _m_din;     // Производная входа этого слоя, также является выходом предыдущего слоя.

public:
    Dense(const int in_size, const int out_size) : Layer(in_size,out_size){}

    void init(const Scalar& mu, const Scalar& sigma, RNG& rng)
    {
        init();
        // Установить случайные коэффициенты
        internal::set_normal_random(_m_weight.data(), _m_weight.size(), rng, mu, sigma);
        internal::set_normal_random(_v_bias.data(), _v_bias.size(), rng, mu, sigma);
    }
    // Установить размерность параметра
    void init()
    {
        _m_weight.resize(this->_in_size, this->_out_size);
        _v_bias.resize(this->_out_size);
        _m_dw.resize(this->_in_size, this->_out_size);
        _v_db.resize(this->_out_size);
    }

    // данные предыдущего слоя: in_size x nobs
    void forward(const Matrix& prev_layer_data)
    {
        const int nobs = prev_layer_data.cols();
        // Линейный термин z = W' * in + b
        _m_z.resize(this->_out_size, nobs);
        _m_z.noalias() = _m_weight.transpose() * prev_layer_data;
        _m_z.colwise() += _v_bias;
        // Применить функцию активации
        _m_a.resize(this->_out_size, nobs);
        Activation::activate(_m_z, _m_a);
    }

    const Matrix& backprop_data() const
    {
        return _m_din;
    }

    void update(Optimizer& opt)
    {
        ConstAlignedMapVec dw(_m_dw.data(), _m_dw.size());
        ConstAlignedMapVec db(_v_db.data(), _v_db.size());
        AlignedMapVec      w(_m_weight.data(), _m_weight.size());
        AlignedMapVec      b(_v_bias.data(), _v_bias.size());
        opt.update(dw, w);
        opt.update(db, b);
    }

    std::vector<Scalar> get_parameters() const
    {
        std::vector<Scalar> res(_m_weight.size() + _v_bias.size());
        // Скопируйте данные весов и смещения в длинный вектор
        std::copy(_m_weight.data(), _m_weight.data() + _m_weight.size(), res.begin());
        std::copy(_v_bias.data(), _v_bias.data() + _v_bias.size(),
                  res.begin() + _m_weight.size());
        return res;
    }

    void set_parameters(const std::vector<Scalar>& param)
    {
        if (static_cast<int>(param.size()) != _m_weight.size() + _v_bias.size())
        {
            throw std::invalid_argument("[class Dense]: Размер параметра не соответствует");
        }

        std::copy(param.begin(), param.begin() + _m_weight.size(), _m_weight.data());
        std::copy(param.begin() + _m_weight.size(), param.end(), _v_bias.data());
    }

    std::vector<Scalar> get_derivatives() const
    {
        std::vector<Scalar> res(_m_dw.size() + _v_db.size());
        // Скопируйте данные весов и смещения в длинный вектор
        std::copy(_m_dw.data(), _m_dw.data() + _m_dw.size(), res.begin());
        std::copy(_v_db.data(), _v_db.data() + _v_db.size(), res.begin() + _m_dw.size());
        return res;
    }

    std::string layer_type() const
    {
        return "Dense";
    }

    std::string activation_type() const
    {
        return Activation::return_type();
    }

    void fill_meta_info(Info& map, int index) const
    {
        //std::string ind = internal::to_string(index);
        std::string ind = std::to_string(index);
        map.insert(std::make_pair("Layer" + ind, internal::layer_id(layer_type())));
        map.insert(std::make_pair("Activation" + ind, internal::activation_id(activation_type())));
        map.insert(std::make_pair("in_size" + ind, in_size()));
        map.insert(std::make_pair("out_size" + ind, out_size()));
    }
    ~Dense() = default;
};
}