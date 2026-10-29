#include "neuralarchitecture.hpp"

NeuralNetwork::NeuralNetwork () {
    this->layers = vector<std::unique_ptr<Layer>>();
}

float* NeuralNetwork::predict(const float* inputs, const size_t inputSize) const {
    const float* prev = inputs;
    size_t prevSize = inputSize;

    for (const std::unique_ptr<Layer>& l : this->layers) {
        l->activate(prev, prevSize);
        prev = l->output();   // Output feeds into next layer
        prevSize = l->layerSize;
    }

    return const_cast<float*>(this->layers.back()->output());
}

void NeuralNetwork::learn() {

}

void NeuralNetwork::addLayer(std::unique_ptr<Layer> l) {
    this->layers.push_back(std::move(l));
}
