#include "shared.hpp"

void StochasticGridWorld();
void GridWorld();
void TestPerformance();
void TestBackPropogation();
void RunGridWorld();

int main() {
    Log("CPPNN - Kristian's neural network framework!");
    RunGridWorld();
    //GridWorld();
    //StochasticGridWorld();
    //TestBackPropogation();
    //TestPerformance();
    
    Log("Finished!");
    return 0;
}