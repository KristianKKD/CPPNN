#include <neuralnetwork.cuh>
#include <shared.hpp>
#include <assert.h>

void TestFeedForward() {
    const int inputSize = 3;
    const int hiddenCount = 2;
    const int hiddenSize = 2;
    const int outputSize = 3;

    NeuralNetwork nn = NeuralNetwork(inputSize);
    for (int i = 0; i < hiddenCount; i++)
        nn.AddLayer(hiddenSize);
    nn.AddLayer(outputSize);
    nn.Build(true); //assume that weights = 1 //note that we need to update this when bias added
    
    assert(nn.layerCount == hiddenCount + 2);
    assert(nn.weightCount == (inputSize * hiddenSize + hiddenSize * hiddenSize * (hiddenCount - 1) + outputSize * hiddenSize));
    assert(nn.nodeCount == inputSize + hiddenCount * hiddenSize + outputSize);

    assert(nn.layerSizes[0] == inputSize);
    assert(nn.layerSizes[1] == hiddenSize);
    assert(nn.layerSizes[2] == hiddenSize);
    assert(nn.layerSizes[3] == outputSize);

    float inputsArr[inputSize] = {1, 2, 3};

    float* outputsArr = new float[outputSize];
    nn.FeedForward(inputsArr, outputsArr);

    for (int i = 0; i < nn.nodeCount; i++)
        Log(to_string(i) + " : " + to_string(nn.activatedOutputs[i]));

}