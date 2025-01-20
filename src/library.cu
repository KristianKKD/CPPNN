#include <shared.hpp>
#include <library.cuh>
#include <algorithm>
#include <cmath>

std::mt19937 generator(Library::randomSeed);
std::uniform_real_distribution<> distribution(0, 1);

float Library::RandomValue() { //generate a value between 0 and 1
    return distribution(generator);
}

