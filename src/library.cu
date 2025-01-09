#include <shared.hpp>
#include <library.cuh>
#include <algorithm>
#include <cmath>

std::mt19937 generator(Library::randomSeed);
std::uniform_real_distribution<> distribution(Library::minVal, Library::maxVal);

float Library::RandomValue() {
    return distribution(generator);
}
