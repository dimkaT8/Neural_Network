#pragma once

#include <vector>
#include <stdexcept>
#include "InitScalar.h"
#include "Utilities/RNG.h"
#include "Layer/Layer.h"
#include "Utilities/Callback.h"

namespace NNE
{
    class Network
    {
        private:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef Eigen::RowVectorXi IntegerVector;
        typedef std::map<std::string, int> MetaInfo;

    };
}