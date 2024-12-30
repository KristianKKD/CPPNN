#include <neuralnetwork.hpp>
#include <library.hpp>
#include <cmath>
#include <algorithm>

#include <iomanip>

using nn = Neural::NeuralNetwork;

nn::NeuralNetwork(int inputCount, int outputCount, int hiddenLayerCount, int hiddenNodesPerLayer, float learningRate) {
    nn::learningRate = learningRate;
    nn::inputCount = inputCount;
    nn::outputCount = outputCount;
    nn::hiddenLayerCount = hiddenLayerCount;
    nn::hiddenNodesPerLayer = hiddenNodesPerLayer;

    nn::weightSize = (nn::inputCount * hiddenNodesPerLayer) + 
                            nn::hiddenNodesPerLayer * nn::hiddenNodesPerLayer * (nn::hiddenLayerCount - 1) * static_cast<int>(hiddenLayerCount > 1) + //if there are no connections, * 0
                            (nn::outputCount * hiddenNodesPerLayer);

    nn::weights = new float[nn::weightSize]();

    nn::biasSize = (nn::hiddenNodesPerLayer * nn::hiddenLayerCount) + nn::outputCount;
    nn::biases = new float[nn::biasSize]; //no input bias

    for (int i = 0; i < nn::weightSize; i++)
        nn::weights[i] = Library::RandomValue() * nn::learningRate;

    for (int i = 0; i < nn::biasSize; i++)
        nn::biases[i] = Library::RandomValue() * nn::learningRate;
}

nn::~NeuralNetwork() {
    delete[] nn::weights;
    delete[] nn::biases;
}

void nn::CopyWeights(float* newWeights) {
    std::copy(newWeights, newWeights + nn::weightSize, nn::weights);
}

void nn::Backpropogate() {

}

float* nn::StoachasticGradient(const size_t batchLearnSize) {
    //randomly change batchLearnSize weights by learningRate
    
    float* oldWeights = new float[nn::weightSize]();
    std::copy(nn::weights, nn::weights + nn::weightSize, oldWeights);

    for (int i = 0; i < batchLearnSize; i++) {
        //choose a random weight
        int randIndex = static_cast<int>(round((Library::RandomValue() / Library::maxVal) * nn::weightSize)); 

        //choose a direction for the weight change
        int randDir = (((Library::RandomValue() / Library::maxVal) < 0.5) ? -1 : 1); 

        //choose a random size for the change
        float randChange = ((Library::RandomValue() / nn::learningRate) * nn::learningRate); 

        //set new weight
        nn:weights[randIndex] += randChange * randDir;
        nn::weights[randIndex] = std::clamp(nn::weights[randIndex], Library::minVal, Library::maxVal); //clamp
    }

    return oldWeights;
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
        //a[i] = inputs[i];

    //calculate outputs of nodes
    for (int nodeIndex = nn::inputCount; nodeIndex < nodeCount; nodeIndex++) {
        //find the number of nodes in the previous layer
        size_t lastLayerNodeCount = nn::hiddenNodesPerLayer;
        if (nodeIndex < nn::inputCount + nn::hiddenNodesPerLayer) //input layer was last layer
            lastLayerNodeCount = nn::inputCount;
        
        //find info about the current layer this node is in
        int layerId = 1 + std::floor((nodeIndex == nn::inputCount) ? 0 : ((nodeIndex - nn::inputCount) / nn::hiddenNodesPerLayer)); //1 = first hidden layer
        int layerSize = nn::hiddenNodesPerLayer;
        if (nodeIndex >= nodeCount - nn::outputCount) { //this is the output layer
            layerSize = nn::outputCount;
            layerId = nn::hiddenLayerCount + 1;
        }
        int layerSizeOffset = (nodeIndex - 1) % layerSize;

        //calculate the number of weights already seen so we target the correct one
        int pastWeightsOffset = (nn::inputCount * hiddenNodesPerLayer * (layerId > 1)) + //add input weights if we are past hidden layer 1
                            nn::hiddenNodesPerLayer * nn::hiddenNodesPerLayer * std::max(layerId - 2, 0); //add number of layer weights past the first


        float sum = nn::biases[nodeIndex - nn::inputCount]; //will be the sum of incoming connections, start with bias
        // std::cout << std::setprecision(4) << std::fixed;
        // std::cout << "N" << nodeIndex << " | B" << nodeIndex - nn::inputCount << ":" << nn::biases[nodeIndex - nn::inputCount] << std::endl;
        for (int connectionIndex = 0; connectionIndex < lastLayerNodeCount; connectionIndex++) { //iterate over all incoming connections into this current node
            int outputIndex = nodeIndex - lastLayerNodeCount - layerSizeOffset + connectionIndex;
            int weightIndex = pastWeightsOffset + (connectionIndex * layerSize) + layerSizeOffset;

            // std::cout << "  A" << outputIndex << ":" << a[outputIndex];
            // std::cout << "      E" << weightIndex << ":" << nn::weights[weightIndex] << std::endl;
            
            sum += a[outputIndex] * nn::weights[weightIndex];
        }
        a[nodeIndex] = Library::ActivationFunction(sum);
        //std::cout << "                      OUT: " << sum << " -> " << a[nodeIndex] << std::endl;
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