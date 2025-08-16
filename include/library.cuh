#pragma once

namespace Library {
#define EPSILON 1e-7 //tiny value to prevent divide by zero errors
    const static unsigned int randomSeed = 2;
    const static unsigned int gpuDevice = 0;

    float RandomValue(const float multiplier = 1); //return value between 0-1 * mult
    float RandomSignedValue(const float multiplier = 1); //return a value between -1-1 * mult
    float MSE(const float* preds, const float* targets, const int arrSize); //return scalar metric for error between two arrays
    float MAE(const float* preds, const float* targets, const int arrSize);
    void Softmax(float* values, const int arrSize);
    int SampleDistribution(const float* probabilities, const int arrSize); //return index of probability chosen, selected based on weighted chance
    void Normalize(float* arr, const int arrSize, const int startingPos = 0); //overwrite the array data with normalized data between the ranges provided
    float SumVector(const float* arr, const int arrSize);
    void CalculateError(float* results, const float* preds, const float* targets, const int arrSize);
};