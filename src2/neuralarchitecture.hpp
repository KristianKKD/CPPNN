#pragma once

#include <memory>
#include "neurallayers.hpp"

class NeuralNetwork {
public:
    vector<std::unique_ptr<Layer>> layers;

    NeuralNetwork();

    float* predict(const float* inputs, const size_t inputSize) const;
    void learn();
    void addLayer(std::unique_ptr<Layer> l);
};