#include <neuralnetwork.hpp>
#include <library.cuh>
#include <cmath>
#include <algorithm>
#include <iomanip>

Neural::NeuralNetwork::NeuralNetwork(int inputSize, int outputSize, int hiddenLayerCount, int hiddenNodesPerLayer, float learningRate) {
    this->learningRate = learningRate;
    this->inputSize = inputSize;
    this->outputSize = outputSize;
    this->hiddenLayerCount = hiddenLayerCount;
    this->hiddenNodesPerLayer = hiddenNodesPerLayer;

    //find the number of values we need to generate
    int weightSize = (inputSize * hiddenNodesPerLayer) + //input connections
                            hiddenNodesPerLayer * hiddenNodesPerLayer * (hiddenLayerCount - 1) * static_cast<int>(hiddenLayerCount > 1) + //hidden connections //if there are no connections, * 0
                            (outputSize * hiddenNodesPerLayer); //output connections
    int biasSize = (hiddenNodesPerLayer * hiddenLayerCount) + outputSize; //no input bias
    int nodeCount = this->inputSize + (this->hiddenNodesPerLayer * this->hiddenLayerCount) + this->outputSize;
    this->weightSize = weightSize;
    this->biasSize = biasSize;
    this->nodeCount = nodeCount;

    //allocate the memory
    cudaError_t error;
    error = cudaMallocManaged(&this->weights, weightSize * sizeof(float));
    if (error != cudaSuccess) {
        Error("Failed to allocate CUDA weights memory");
        return;
    }
    error = cudaMallocManaged(&this->biases, biasSize * sizeof(float));
    if (error != cudaSuccess) {
        Error("Failed to allocate CUDA biases memory");
        return;
    }
    error = cudaMallocManaged(&this->inputs, inputSize * sizeof(float));
    if (error != cudaSuccess) {
        Error("Failed to allocate CUDA memory for inputs");
        return;
    }
    error = cudaMallocManaged(&this->outputs, outputSize * sizeof(float));
    if (error != cudaSuccess) {
        Error("Failed to allocate CUDA memory for outputs");
        return;
    }
    error = cudaMallocManaged(&this->a, nodeCount * sizeof(float));
    if (error != cudaSuccess) {
        Error("Failed to allocate CUDA memory for (a) holding array");
        return;
    }

    //initialize to random values
    for (int i = 0; i < weightSize; i++)
        this->weights[i] = Library::RandomValue() * learningRate;
    for (int i = 0; i < biasSize; i++)
        this->biases[i] = Library::RandomValue() * learningRate;

    //prefetch the data as we will be needing it soon
    int device = 0;
    cudaMemPrefetchAsync(weights, weightSize * sizeof(float), device);
    cudaMemPrefetchAsync(biases, biasSize * sizeof(float), device);
}

Neural::NeuralNetwork::~NeuralNetwork() {
    cudaFree(this->weights);
    cudaFree(this->biases);
    cudaFree(this->inputs);
    cudaFree(this->outputs);
    cudaFree(this->a);
}

void Neural::NeuralNetwork::CopyWeights(float* newWeights) {
    //std::copy(newWeights, newWeights + nn::weightSize, nn::weights);
}

float* Neural::NeuralNetwork::StoachasticGradient(const size_t batchLearnSize) {
    // //randomly change batchLearnSize weights by learningRate
    
    float* oldWeights = new float[this->weightSize]();
    // std::copy(nn::weights, nn::weights + nn::weightSize, oldWeights);

    // for (int i = 0; i < batchLearnSize; i++) {
    //     //choose a random weight
    //     int randIndex = static_cast<int>(round((Library::RandomValue() / Library::maxVal) * nn::weightSize)); 

    //     //choose a direction for the weight change
    //     int randDir = (((Library::RandomValue() / Library::maxVal) < 0.5) ? -1 : 1); 

    //     //choose a random size for the change
    //     float randChange = ((Library::RandomValue() / nn::learningRate) * nn::learningRate); 

    //     //set new weight
    //     nn::weights[randIndex] += randChange * randDir;
    //     nn::weights[randIndex] = std::clamp(nn::weights[randIndex], Library::minVal, Library::maxVal); //clamp
    // }

    return oldWeights;
}

__global__ void ActivateInputs(float* a, const float* inputs, int inputCount, float minVal, float maxVal) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < inputCount) {
        float val = inputs[idx];
        Library::ActivationFunction(&val, minVal, maxVal);
        a[idx] = val;
    }
}

__global__ void Forward(float* a, const float* weights, const float* biases, 
                        int nodeCount, int inputSize, int outputSize, 
                        int hiddenLayerCount, int hiddenNodesPerLayer,
                        float minVal, float maxVal) {
    int nodeIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (nodeIndex >= nodeCount - inputSize)
        return;
    
    //find the number of nodes in the previous layer
    size_t lastLayerNodeCount = hiddenNodesPerLayer;
    if (nodeIndex < inputSize + hiddenNodesPerLayer) //input layer was last layer
        lastLayerNodeCount = inputSize;
    
    //find info about the current layer this node is in
    int layerId = 1 + floorf((nodeIndex == inputSize) ? 0 : ((nodeIndex - inputSize) / hiddenNodesPerLayer)); //1 = first hidden layer
    int layerSize = hiddenNodesPerLayer;
    if (nodeIndex >= nodeCount - outputSize) { //this is the output layer
        layerSize = outputSize;
        layerId = hiddenLayerCount + 1;
    }
    int layerSizeOffset = (nodeIndex - 1) % layerSize;

    //calculate the number of weights already seen so we target the correct one
    int layerIDVal = ((layerId - 2 > 0) ? layerId - 2 : 0);
    int pastWeightsOffset = (inputSize * hiddenNodesPerLayer * (layerId > 1)) + //add input weights if we are past hidden layer 1
                        hiddenNodesPerLayer * hiddenNodesPerLayer * layerIDVal; //add number of layer weights past the first

    float sum = biases[nodeIndex - inputSize]; //will be the sum of incoming connections, start with bias
    for (int connectionIndex = 0; connectionIndex < lastLayerNodeCount; connectionIndex++) { //iterate over all incoming connections into this current node
        int outputIndex = nodeIndex - lastLayerNodeCount - layerSizeOffset + connectionIndex;
        int weightIndex = pastWeightsOffset + (connectionIndex * layerSize) + layerSizeOffset;

        sum += a[outputIndex] * weights[weightIndex];
    }

    Library::ActivationFunction(&sum, minVal, maxVal);
    a[nodeIndex] = sum;
}

void Neural::NeuralNetwork::FeedForward(const float* inputs, float* outputs) {
    //feed forward input, returns the output layer activated values (a) into the outputs argument
    //node value (z) = sum of 

    int threadsPerBlock = 128;
    int blocksForInput = (this->inputSize + threadsPerBlock - 1) / threadsPerBlock;

    //send the input data to the CUDA memory
    cudaMemcpy(this->inputs, inputs, this->inputSize * sizeof(float), cudaMemcpyHostToDevice);
    ActivateInputs<<<blocksForInput, threadsPerBlock>>>(this->a, this->inputs, this->inputSize, Library::minVal, Library::maxVal);

    //calculate outputs of nodes
    for (int nodeIndex = this->inputSize; nodeIndex < nodeCount; nodeIndex++) {
        Forward<<<1, threadsPerBlock>>>(this->a, this->weights, this->biases,
                                        nodeCount, this->inputSize, this->outputSize,
                                        this->hiddenLayerCount, this->hiddenNodesPerLayer,
                                        Library::minVal, Library::maxVal);
    }

    cudaDeviceSynchronize();
    std::copy(this->a + nodeCount - 1 - this->outputSize, this->a + nodeCount - 1, outputs); //test

    return;
}

void Neural::NeuralNetwork::PrintNetwork() {
    // int edgeSequentialIndex = 0; //to know which edge to display, count +1 every time we display a new one
    // for (int layerIndex = 0; layerIndex < nn::hiddenLayerCount + 2; layerIndex++) {
    //     std::cout << ((layerIndex == 0) ? "I" : ((layerIndex != 0 && layerIndex != nn::hiddenLayerCount + 2 - 1) ? "H" : "O"));
    //     std::cout << "L" << layerIndex << std::endl;

    //     int nodeCount = nn::hiddenNodesPerLayer;
    //     if (layerIndex == 0) //input layer
    //         nodeCount = nn::inputSize;
    //     else if (layerIndex == nn::hiddenLayerCount + 2 - 1) //output layer
    //         nodeCount = nn::outputSize;

    //     for (int nodeIndex = 0; nodeIndex < nodeCount; nodeIndex++) {
    //         std::cout << "  N" << nodeIndex << " - B:" << ((layerIndex == 0) ? 0 : nn::biases[(layerIndex - 1) * nn::hiddenNodesPerLayer + nodeIndex]) << std::endl;

    //         int edgeCount = nn::hiddenNodesPerLayer;
    //         if (layerIndex == nn::hiddenLayerCount) //the last hidden layer, layer before the output
    //             edgeCount = nn::outputSize;
    //         else if (layerIndex == nn::hiddenLayerCount + 2 - 1)
    //             edgeCount = 0;

    //         for (int edgeIndex = 0; edgeIndex < edgeCount; edgeIndex++)
    //             std::cout << "          E" << edgeIndex << ":" << nn::weights[edgeSequentialIndex++] << std::endl;
    //     }


    // }
}