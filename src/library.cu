#include <shared.hpp>
#include <library.cuh>
#include <algorithm>
#include <cmath>

std::mt19937 generator(Library::randomSeed);
std::uniform_real_distribution<> distribution(0, 1);

float Library::RandomValue() { //generate a value between 0 and 1
    return distribution(generator);
}

float Library::MSE(float* preds, float* targets, int arrSize) {
    float sum = 0;
    for (int i = 0; i < arrSize; i++) {
        float diff = (preds[i] - targets[i]);
        sum += diff * diff;
    }

    if (arrSize != 0 && sum != 0)
        sum /= arrSize;

    return sum;
}

float Library::MAE(float* preds, float* targets, int arrSize) {
    float sum = 0;
    for (int i = 0; i < arrSize; i++)
        sum += std::abs(preds[i] - targets[i]);

    if (arrSize != 0 && sum != 0)
        sum /= arrSize;

    return sum;
}

float Library::DerActivationFunction(float activatedValue) {
    return (activatedValue * (1 - activatedValue));
}