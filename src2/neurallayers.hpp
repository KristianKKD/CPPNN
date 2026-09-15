#pragma once

#include <vector>

using std::vector;

class Layer { // Abstract class for network components
public:
    size_t layerSize;
    
    size_t weightSize;
    size_t valueSize;

    vector<float> weights;
    vector<float> values;


    virtual void activate(const float* inputs, const size_t inputSize) = 0;

protected:
    //virtual ~Layer() = default;
    //virtual Layer& operator=(const Layer& origin) = 0; 
};

class Dense: public Layer { // Main "Neural Network" layer
public:
    size_t depth;

    size_t biasSize;

    vector<float> bias;


    Dense(const size_t layerSize, const size_t depth);
    ~Dense();
    void activate(const float* inputs, const size_t inputSize) override;
    void activate(const float* inputs, const size_t inputSize, float* weights, float* values, float* bias, const size_t layerSize, const size_t depth);
};

class ActivationLayer: public Layer { // Non-linearity layer
public:
    enum ActivationType {
        ReLU,
        Sigmoid,
        Tanh
    };

    ActivationType activationType = ReLU;

    ActivationLayer(size_t layerSize, ActivationType activationType);
    void activate(const float* inputs, const size_t inputSize) override;
    void activate(const float* inputs, const size_t inputSize, float* values, const size_t layerSize, const ActivationType activationType) const;

};