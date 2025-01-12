#include <shared.hpp>
#include <random>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace Library {
    const static unsigned int randomSeed = 2;
    const static unsigned int gpuDevice = 0;

    float RandomValue(); //return value between 0-1

    __host__ __device__ inline void ActivationFunction(float* value) {
        //*value = 1.0f / (1.0f + exp(-*value)) - maxVal/2; //sigmoid
        *value = std::fmax(0.0f, *value); //ReLU
    }
};