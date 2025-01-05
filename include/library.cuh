#include <shared.hpp>
#include <random>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace Library {
    const static unsigned int randomSeed = 2;
    const static float minVal = 0;
    const static float maxVal = 1;

    void PrintVector(const float* array, size_t arraySize);
    float CalculateMSE(const float* outputs, const size_t outputSize, const float* targets, const size_t targetSize);
    float RandomValue(); //return value between minValue-maxValue
    // float DerActivationFunction(float value);
    void ResetArray(float* arr, size_t arrSize);

    __host__ __device__ inline void ActivationFunction(float* value, float minVal, float maxVal) {
        //return std::clamp(1.0f / (1.0f + std::exp(-value)) - (Library::maxVal/2), Library::minVal, Library::maxVal); //sigmoid
        if (*value > maxVal)
            *value = maxVal;
        if (*value < minVal)
            *value = minVal;
        *value = std::fmax(0.0f, *value); //ReLU
    }
};