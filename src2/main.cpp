#include "neuralarchitecture.hpp"
#include <iostream>

int main() {
    NeuralNetwork net = NeuralNetwork();
    net.AddLayer(std::make_unique<Dense>(4, 4));
    net.AddLayer(std::make_unique<ActivationLayer>(ActivationLayer(4, ActivationLayer::ActivationType::ReLU)));

    vector<float> inputs = {0, 1, 0, 0};

    float* results = net.Predict(inputs.data(), inputs.size());

    for (int i = 0; i < 4; i++) {
        std::cout << results[i] << " ";
    }
    std::cout << "\n";

    return 0;
}