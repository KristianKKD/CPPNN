#include "neurallayers.hpp"
#include <iostream>

ActivationLayer::ActivationLayer(size_t layerSize, ActivationType activationType) {
    if (layerSize < 1)
        throw std::invalid_argument("Layer size must be >= 1");

    this->layerSize = layerSize;
    this->activationType = activationType;

    // Weights are segmented into layerSize incoming for each node (e.g. 0->4 are the 5 incoming weights to node (layerSize + 1))
    size_t weightSize = layerSize*layerSize;
    this->weights = new float[weightSize];
    for (size_t weightIndex = 0; weightIndex < weightSize; weightIndex++)
        this->weights[weightIndex] = Library::RandomSignedValue(1);

    // Values are just storage for feed forward values to propogate
    size_t valueSize = layerSize;
    this->values = new float[valueSize];
    for (size_t valueIndex = 0; valueIndex < valueSize; valueIndex++)
        this->values[valueIndex] = 0;

    // TODO: BIAS
    this->bias = nullptr;
}

void ActivationLayer::activate(const float* inputs, const size_t inputSize) const {
    activate(inputs, inputSize, this->values, this->layerSize, this->activationType);
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