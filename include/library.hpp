#include <shared.hpp>
#include <random>

namespace Library {
    const static unsigned int randomSeed = 2;
    const static float minVal = 0;
    const static float maxVal = 1;
    static std::mt19937 generator(randomSeed);
    static std::uniform_real_distribution<> distribution(Library::minVal, Library::maxVal);

    void Library::PrintVector(const vector<double>& vec);
    double Library::CalculateMSE(const vector<double>& outputs, const vector<double>& targets);
    static double Library::RandomValue();
    static double Library::ActivationFunction(double value);
    static double Library::DerActivationFunction(double value);
};