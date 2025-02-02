#include <shared.hpp>
#include <neuralnetwork.cuh>
#include <unittests.cuh>

int main() {
    TestBackPropogation();

    // const int inputSize = 8;
    // const int hiddenLayerCount = 4;
    // const int hiddenNodesPerLayer = 4;
    // const int outputSize = 8;

    // NeuralNetwork net = NeuralNetwork(inputSize); //create + input layer
    // for (int i = 0; i < hiddenLayerCount; i++) //hidden layers
    //     net.AddLayer(hiddenNodesPerLayer);
    // net.AddLayer(outputSize); //output layer
    // net.Build(); //finalize
    // Log("Built network with " + to_string(hiddenLayerCount + 2) + " layers");

    // float outputsArr[outputSize];
    // net.FeedForward(outputsArr);

    // for (int i = 0; i < outputSize; i++)
    //     Log(to_string(outputsArr[i]));

    Log("Finished!");
    return 0;
}