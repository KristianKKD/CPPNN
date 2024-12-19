#include <neuralnetwork.hpp>
#include <library.hpp>
#include <cmath>
#include <algorithm>

using nn = Neural::NeuralNetwork;

nn::NeuralNetwork(int inputCount, int outputCount, int hiddenLayerCount, int hiddenNodesPerLayer) {
    size_t weightSize = (nn::inputCount * hiddenNodesPerLayer) + 
                            int(std::pow(nn::hiddenNodesPerLayer, nn::hiddenLayerCount)) * int(hiddenLayerCount > 1) + //if there are no connections, * 0
                            (nn::outputCount * hiddenNodesPerLayer);
    nn::weights = new float[weightSize]();

    size_t biasSize = (nn::hiddenNodesPerLayer * nn::hiddenLayerCount) + nn::outputCount;
    nn::biases = new float[biasSize]; //no input bias

    for (int i = 0; i < weightSize; i++)
        nn::weights[i] = Library::RandomValue();

    for (int i = 0; i < biasSize; i++)
        nn::biases[i] = Library::RandomValue();
}

nn::~NeuralNetwork() {
    delete[] nn::weights;
    delete[] nn::biases;
}

void nn::Backpropogate() {

}

void nn::FeedForward(const float* inputs, float* outputs) {
    //feed forward input, returns the output layer activated values (a) into the outputs argument

    size_t nodeCount = nn::inputCount + (nn::hiddenNodesPerLayer* nn::hiddenLayerCount) + nn::outputCount;
    float* a = new float[nodeCount];

    std::copy(inputs, inputs + nn::inputCount, a); //test

    //activate the layers using the previous nodes
    for (int layerIndex = (nn::inputCount); layerIndex < nn::hiddenLayerCount + 2; layerIndex++) { //+2 for input/output, don't try to activate the inputs
        int zeroFlag = int((layerIndex - 1) != 0);
       
        for (int nodeIndex = 0; nodeIndex < ((layerIndex < nn::hiddenLayerCount) ? nn::hiddenNodesPerLayer : nn::outputCount); nodeIndex++) { //output layer may have different number of nodes
            
            //initialize sum as bias of node (sum is z, initialized as b)
            float sum = nn::biases[((layerIndex - 1) * nn::hiddenNodesPerLayer) + //no input biases so 0 index is hidden layer 0
                ((layerIndex == nn::hiddenLayerCount + 1) ? 0 : nn::outputCount) //output layer may have different node count
                + nodeIndex];
            
            //get the sum of activations (z = sum of a1w1 + a2w1 + a3w1... + bias), different layers may have different counts of nodes
            for (int preNodeIndex = 0; preNodeIndex < ((layerIndex == 1) ? nn::inputCount : nn::hiddenNodesPerLayer); preNodeIndex++) {
                //if we are on the input layer, some terms should be 0

                //get a
                float relevantVal = a[nn::inputCount * zeroFlag + //account for inputs
                    nn::hiddenNodesPerLayer * zeroFlag * layerIndex + //account for hidden layers
                    preNodeIndex]; //find relevant value

                //get w
                float relevantWeight = nn::weights[nn::inputCount * nn::hiddenNodesPerLayer * zeroFlag + //account for input weights
                    int(std::pow(nn::hiddenNodesPerLayer, layerIndex)) * int(hiddenLayerCount > 1) * int(layerIndex > 1) + //account for hidden layer weights, * 0 if only single hidden layer or on the 0th hidden layer
                    preNodeIndex]; //get relevant weight

                sum += relevantVal * relevantWeight;
            } //end of preNodeIndex

            sum = Library::ActivationFunction(std::clamp(sum, Library::minVal, Library::maxVal));
            a[nn::inputCount * zeroFlag + //account for inputs
                nn::hiddenNodesPerLayer * zeroFlag * layerIndex + //account for hidden layers
                nodeIndex] = sum;
        } //end of nodeIndex
    } //end of layerIndex

    std::copy(a + nodeCount - 1 - nn::outputCount, a + nodeCount - 1, outputs); //test
    delete[] a; //free memory

    return;
}
