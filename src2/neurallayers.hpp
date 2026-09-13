#pragma once

class Layer { // Abstract class for network components
public:
    size_t layerSize;

    virtual void activate(const float* inputs, const size_t inputSize) const = 0;
protected:
    float* weights;
    float* values;
    float* bias;

    virtual ~Layer() = default;
    //virtual Layer& operator=(const Layer& origin) = 0; 
};

class Dense: public Layer {
public:
    size_t depth;

    Dense(const size_t layerSize, const size_t depth);
    ~Dense();
    void activate(const float* inputs, const size_t inputSize) const override;
    void activate(const float* inputs, const size_t inputSize, float* weights, float* values, size_t layerSize, size_t depth) const;
};

class ActivationLayer: Layer {
public:
    enum ActivationType {
        ReLU,
        Sigmoid,
        Tanh
    };

    ActivationType activationType = ReLU;

    ActivationLayer(size_t layerSize, ActivationType activationType);
    void activate(const float* inputs, const size_t inputSize) const override;
    void ActivationLayer::activate(const float* inputs, const size_t inputSize, float* values, const size_t layerSize, const ActivationType activationType) const;


};