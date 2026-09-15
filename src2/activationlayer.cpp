#include "library.hpp"
#include "neurallayers.hpp"
#include <iostream>

inline void activationFunction(float* value, const ActivationLayer::ActivationType type);

ActivationLayer::ActivationLayer(size_t layerSize, ActivationType activationType) {
    if (layerSize < 1)
        throw std::invalid_argument("Layer size must be >= 1");

    this->activationType = activationType; // Which activation function to use
    this->layerSize = layerSize; // Nodes per layer
    this->weightSize = layerSize*layerSize; // Weight count
    this->valueSize = layerSize; // Node count

    // Weights are segmented into layerSize incoming for each node (e.g. 0->4 are the 5 incoming weights to node (layerSize + 1))
    this->weights = vector<float>(weightSize);
    for (size_t weightIndex = 0; weightIndex < weightSize; weightIndex++)
        this->weights[weightIndex] = Library::RandomSignedValue(1);

    // Values are just storage for feed forward values to propogate so no need for values
    this->values = vector<float>(valueSize, 0);
}

void ActivationLayer::activate(const float* inputs, const size_t inputSize) {
    activate(inputs, inputSize, this->values.data(), this->layerSize, this->activationType);
}

void ActivationLayer::activate(const float* inputs, const size_t inputSize, float* values, const size_t layerSize, const ActivationType activationType) const {
    for (size_t valueIndex = 0; valueIndex < layerSize; valueIndex++)
        activationFunction(values + valueIndex, activationType);
}

inline void activationFunction(float* value, const ActivationLayer::ActivationType type) {
    switch (type) {
        case ActivationLayer::ReLU:
            *value = std::fmax(0.0f, *value);
            break;
        case ActivationLayer::Sigmoid:
            *value = 1.0f / (1.0f + exp(-(*value)));
            break;
        case ActivationLayer::Tanh:
            *value = std::tanh(*value);
            break;
    }
}