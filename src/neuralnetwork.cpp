#include <neuralnetwork.hpp>
#include <library.hpp>
#include <cmath>
#include <algorithm>

using nn = Neural::NeuralNetwork;

nn::NeuralNetwork(int inputCount, int outputCount, int hiddenLayerCount, int hiddenNodesPerLayer) {
    nn::inputCount = inputCount;
    nn::outputCount = outputCount;
    nn::hiddenLayerCount = hiddenLayerCount;
    nn::hiddenNodesPerLayer = hiddenNodesPerLayer;

    size_t weightSize = (nn::inputCount * hiddenNodesPerLayer) + 
                            nn::hiddenNodesPerLayer * nn::hiddenNodesPerLayer * (nn::hiddenLayerCount - 1) * static_cast<int>(hiddenLayerCount > 1) + //if there are no connections, * 0
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
    //node value (z) = sum of 

    //set up the array that will hold the activated values of every node 
    size_t nodeCount = nn::inputCount + (nn::hiddenNodesPerLayer * nn::hiddenLayerCount) + nn::outputCount;
    float* a = new float[nodeCount]; //activated output of node

    //add the inputs to the activated array
    for (int i = 0; i < nn::inputCount; i++)
        a[i] = Library::ActivationFunction(inputs[i]); //bias = 0

    int edgesUsed = 0; //really simple way to use every weight in the correct order, just count the weights as they are already sorted

    for (int nodeIndex = nn::inputCount; nodeIndex < nodeCount; nodeIndex) {
        //find the number of nodes in the previous layer
        size_t lastLayerNodeCount = nn::hiddenNodesPerLayer;
        if (nodeIndex < nn::inputCount + nn::hiddenNodesPerLayer) //input layer was last layer
            lastLayerNodeCount = nn::inputCount;
        
        float sum = nn::biases[nodeIndex - nn::inputCount];
        for (int connectionIndex = 0; connectionIndex < lastLayerNodeCount; connectionIndex++) //iterate over all incoming connections into this current node
            sum += a[nodeIndex - lastLayerNodeCount + connectionIndex] * nn::weights[edgesUsed++];

        a[nodeIndex] = Library::ActivationFunction(std::clamp(sum, Library::minVal, Library::maxVal));
    }

    std::copy(a + nodeCount - 1 - nn::outputCount, a + nodeCount - 1, outputs); //test
    delete[] a; //free memory

    return;
}

void nn::PrintNetwork() {
    int hiddenNodesFlag = static_cast<int>(hiddenLayerCount > 1); //are there any hidden layers with connections to another hidden layer
    
    int edgeSequentialIndex = 0; //to know which edge to display, count +1 every time we display a new one
    for (int layerIndex = 0; layerIndex < nn::hiddenLayerCount + 2; layerIndex++) {
        int notInputFlag = static_cast<int>((layerIndex - 1) != 0); //are we (not) in the input layer
        int insideHiddenFlag = static_cast<int>(layerIndex > 1); //are we past the first hidden layer
       
        std::cout << ((layerIndex == 0) ? "I" : ((layerIndex != 0 && layerIndex != nn::hiddenLayerCount + 2 - 1) ? "H" : "O"));
        std::cout << "L" << layerIndex << std::endl;

        int nodeCount = nn::hiddenNodesPerLayer;
        if (layerIndex == 0) //input layer
            nodeCount = nn::inputCount;
        else if (layerIndex == nn::hiddenLayerCount + 2 - 1) //output layer
            nodeCount = nn::outputCount;

        for (int nodeIndex = 0; nodeIndex < nodeCount; nodeIndex++) {
            std::cout << "  N" << nodeIndex << " - B:" << ((layerIndex == 0) ? 0 : nn::biases[(layerIndex - 1) * nn::hiddenNodesPerLayer + nodeIndex]) << std::endl;

            int edgeCount = nn::hiddenNodesPerLayer;
            if (layerIndex == nn::hiddenLayerCount) //the last hidden layer, layer before the output
                edgeCount = nn::outputCount;
            else if (layerIndex == nn::hiddenLayerCount + 2 - 1)
                edgeCount = 0;

            for (int edgeIndex = 0; edgeIndex < edgeCount; edgeIndex++)
                std::cout << "          E" << edgeIndex << ":" << nn::weights[edgeSequentialIndex++] << std::endl;
        }


    }
}