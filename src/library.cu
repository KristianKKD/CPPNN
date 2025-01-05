#include <shared.hpp>
#include <library.cuh>
#include <algorithm>
#include <cmath>

std::mt19937 generator(Library::randomSeed);
std::uniform_real_distribution<> distribution(Library::minVal, Library::maxVal);

void Library::PrintVector(const float* array, size_t arraySize) {
    for (int i = 0; i < arraySize; i++)
        std::cout << array[i] << " ";
    std::cout << std::endl;
}

float Library::CalculateMSE(const float* outputs, const size_t outputSize, const float* targets, const size_t targetSize) {
    if (outputSize != targetSize) {
        Error("FeedForward size(" + to_string(outputSize) + ") does not match target size(" + to_string(targetSize) + ")!");
        return {};
    }

    double error = 0;
    for (int i = 0; i < outputSize; i++) { //sum of squared norms (z)
        //z = ||y(x) - a^L(x)||^2
        double diff = targets[i] - outputs[i];
        error += sqrt(diff * diff);
    }

    error /= (2 * int(outputSize)); //average the summed norms

    return error;
}

float Library::RandomValue() {
    return distribution(generator);
}

// float Library::DerActivationFunction(float value) {
//     return ActivationFunction(value) * (1.0f - ActivationFunction(value)); //sigmoid
// }

void Library::ResetArray(float* arr, size_t arrSize) {
    for (int i = 0; i < arrSize; i++) 
        arr[i] = 0;
}