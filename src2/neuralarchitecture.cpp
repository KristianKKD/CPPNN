#include "neuralarchitecture.hpp"

NeuralNetwork::NeuralNetwork () {
    this->layers = vector<std::unique_ptr<Layer>>();
}

float* NeuralNetwork::Predict(const float* inputs, const size_t inputSize) const {
    for (const std::unique_ptr<Layer>& l : this->layers)
        l->activate(inputs, inputSize);
    return this->layers[layers.size()-1]->values.data();
}

void NeuralNetwork::Learn() {

}

void NeuralNetwork::AddLayer(std::unique_ptr<Layer> l) {
    this->layers.push_back(std::move(l));
}
