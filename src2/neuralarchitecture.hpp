#pragma once

#include <memory>
#include "neurallayers.hpp"

class NeuralNetwork {
public:
    vector<std::unique_ptr<Layer>> layers;

    NeuralNetwork();

    float* Predict(const float* inputs, const size_t inputSize) const;
    void Learn();
    void AddLayer(std::unique_ptr<Layer> l);
};