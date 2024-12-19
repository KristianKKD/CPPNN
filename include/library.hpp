#include <shared.hpp>
#include <random>

namespace Library {
    const unsigned int randomSeed = 2;
    const float minVal = 0;
    const float maxVal = 1;

    void PrintVector(const float* array, size_t arraySize);
    float CalculateMSE(const float* outputs, const size_t outputSize, const float* targets, const size_t targetSize);
    float RandomValue();
    float ActivationFunction(float value);
    float DerActivationFunction(float value);
};