#include <shared.hpp>
#include <neuralnetwork.hpp>
#include <algorithm>
#include <cmath>

static double Library::RandomValue() {
    return Library::distribution(generator);
}

void Library::PrintVector(const vector<double>& vec) {
    for (int i = 0; i < vec.size(); i++)
        std::cout << vec[i] << " ";
    std::cout << std::endl;
}

static double Library::ActivationFunction(double value) {
    return std::clamp(1.0f / (1.0f + std::exp(-value)), minVal, maxVal); //sigmoid
    //return std::max(0.0f, value); //ReLU
}

static double Library::DerActivationFunction(double value) {
    return ActivationFunction(value) * (1.0f - ActivationFunction(value)); //sigmoid
}

double Library::CalculateMSE(const vector<double>& outputs, const vector<double>& targets) {
    if (outputs.size() != targets.size()) {
        Error("Output size(" + to_string(outputs.size()) + ") does not match target size(" + to_string(targets.size()) + ")!");
        return {};
    }

    double error = 0;
    int n = outputs.size();
    for (int i = 0; i < n; i++) { //sum of squared norms (z)
        //z = ||y(x) - a^L(x)||^2
        double diff = targets[i] - outputs[i];
        error += sqrt(diff * diff);
    }

    error /= (2 * n); //average the summed norms

    return error;
}
