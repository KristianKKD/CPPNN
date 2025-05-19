#include <shared.hpp>

void GridWorld();
void TestPerformance();
void TestBackPropogation();


int main() {
    Log("CPPNN - Kristian's neural network framework!");
    //GridWorld();
    TestBackPropogation();
    //TestPerformance();

    Log("Finished!");
    return 0;
}