#include <shared.hpp>
#include <library.cuh>
#include <algorithm>
#include <cmath>
#include <random>

std::mt19937 generator(Library::randomSeed);
std::uniform_real_distribution<> distribution(0, 1);
std::random_device rd;
std::mt19937 gen(rd()); 

float Library::RandomValue(float multiplier) { //generate a value between 0 and 1
    return (distribution(generator) * multiplier);
}

float Library::RandomSignedValue(float multiplier) {
    float rand = Library::RandomValue(multiplier);
    return (Library::RandomValue() < 0.5f) ? -rand : rand;
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
    float sum = 0.0f;
    for (int i = 0; i < arrSize; i++)
        sum += std::abs(preds[i] - targets[i]);

    if (arrSize != 0.0f && sum != 0.0f)
        sum /= arrSize;

    return sum;
}

void Library::Softmax(float* values, int arrSize) {
    float* exp = new float[arrSize];
    float expSum = 0.0f;
    for (int i = 0; i < arrSize; i++) {
        float e = std::exp(values[i]);
        exp[i] = e;
        expSum += e;
    }

    for (int i = 0; i < arrSize; i++)
        values[i] = exp[i] / expSum;

    delete exp;
}

int Library::SampleDistribution(float* probabilities, int arrSize) { //return index of probability chosen, selected based on weighted chance
    std::discrete_distribution<int> dist(probabilities, probabilities + arrSize);
    return dist(gen);
}

void Library::Normalize(float* arr, int arrSize, int startingPos) { //TODO, IMPLEMENT NORMALIZATION ON GPU?
     //calc mean
     float sum = 0.0f;
     for (int sumIndex = 0; sumIndex < arrSize; sumIndex++)
         sum += arr[startingPos + sumIndex];
     if (sum == 0)
         sum += EPSILON;
     float mean = sum/(float)arrSize;

     //calc std
     float variance = 0.0f;
     for (int varIndex = 0; varIndex < arrSize; varIndex++) {
         float val = arr[startingPos + varIndex];
         variance += (val - mean) * (val - mean);
     }
     if (variance == 0.0f)
         variance += EPSILON;
     float std = std::sqrt(variance/(float)arrSize);

     for (int normIndex = 0; normIndex < arrSize; normIndex++) {
         float val = arr[startingPos + normIndex];
         float newVal = val - mean;
         if (newVal == 0.0f)
             newVal += EPSILON;
         
         newVal = (newVal/std);
         arr[startingPos + normIndex] = newVal;
         //Log("Normalized " + to_string(startingPos + normIndex) + " from " + to_string(val) + " -> " + to_string(newVal));
     }
}

float Library::SumVector(float* arr, int arrSize) {
    float sum = 0;
    for (int i = 0; i < arrSize; i++)
        sum += arr[i];

    return sum;
}