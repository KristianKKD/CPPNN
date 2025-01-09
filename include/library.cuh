#include <shared.hpp>
#include <random>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace Library {
    const static unsigned int randomSeed = 2;
    const static float minVal = -1;
    const static float maxVal = 1;

    float RandomValue(); //return value between minValue-maxValue

    __host__ __device__ inline void ActivationFunction(float* value, float minVal, float maxVal) {
        // //clamp
        // if (*value > maxVal)
        //     *value = maxVal;
        // if (*value < minVal)
        //     *value = minVal;
            
        //*value = 1.0f / (1.0f + exp(-*value)) - maxVal/2; //sigmoid
        *value = std::fmax(0.0f, *value); //ReLU
    }
};