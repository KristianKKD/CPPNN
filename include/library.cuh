#pragma once

namespace Library {
#define EPSILON 1e-7 //tiny value to prevent divide by zero errors
    const static unsigned int randomSeed = 2;
    const static unsigned int gpuDevice = 0;

    float RandomValue(float multiplier = 1); //return value between 0-1 * mult
    float RandomSignedValue(float multiplier = 1); //return a value between -1-1 * mult
    float MSE(float* preds, float* targets, int arrSize); //return scalar metric for error between two arrays
    float MAE(float* preds, float* targets, int arrSize);
    void Softmax(float* values, int arrSize);
    int SampleDistribution(float* probabilities, int arrSize); //return index of probability chosen, selected based on weighted chance
    void Normalize(float* arr, int arrSize, int startingPos = 0); //overwrite the array data with normalized data between the ranges provided
    float SumVector(float* arr, int arrSize);

};