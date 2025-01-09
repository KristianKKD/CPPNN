#include "library.cuh"
#include "neuralnetwork.cuh"
  
#define CUDACHECK(call) {                                                        \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at "     \
                      << __FILE__ << ":" << __LINE__ << std::endl;               \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    }


NeuralNetwork::NeuralNetwork(int inputSize) {
    this->weightCount = 0;
    this->nodeCount = 0;
    this->layerCount = 0;
    this->AddLayer(inputSize);
}

NeuralNetwork::~NeuralNetwork() {
    cudaFree(this->weights);
    cudaFree(this->activatedOutputs);
}

void NeuralNetwork::AddLayer(int size) {
    int lastLayerSize = 0;
    if (this->layerCount > 0)
        lastLayerSize = this->layerSizes[this->layerCount - 1];

    int newWeightCount = lastLayerSize * size;
    this->weightCount += newWeightCount;

    this->layerSizes[this->layerCount] = size;
    this->layerCount++;
    this->nodeCount += size;
}

__global__ void Sum(float* activatedOutputs, const float* weights, 
                    const int layerSize, const int nextLayerSize,
                    const int nodeOffset, const int weightOffset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; //this might cause an error if there aren't enough blocks/threads?

    if (i > layerSize * nextLayerSize - 1)
        return;

    int layerSizeOffset = ((i == 0) ? 0 : i/layerSize); //periodic function
    int targetIn = i - (layerSizeOffset * layerSize) + nodeOffset; //periodic in the range between min target node id and max target node id 
    int targetWeight = i + weightOffset;
    int targetNode = nodeOffset + layerSize + layerSizeOffset;

    // printf("Thread %d:      LSO: %d     IN: %d      W: %d      OUT: %d ||| IN: %d      W: %d      OUT: %d\n", 
    // i, layerSizeOffset, targetIn, targetWeight, targetNode,
    // activatedOutputs[targetIn], weights[targetWeight], activatedOutputs[targetNode]);

    printf("%d\n", activatedOutputs[0]);

    activatedOutputs[targetNode] += activatedOutputs[targetIn] * weights[targetWeight];
}

__global__ void ActivateLayer(float* activatedOutputs, const int layerSize, const int nodeOffset, const float minVal, const float maxVal) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i > layerSize)
        return;

    Library::ActivationFunction(&activatedOutputs[nodeOffset + i], minVal, maxVal);
}

void NeuralNetwork::Build(bool debug = false) {
    //memory for the weights array
    CUDACHECK(cudaMallocManaged(&this->weights, this->weightCount * sizeof(float)));

    //randomly initialize weights
    for (int i = 0; i < this->weightCount; i++)
        this->weights[i] = ((debug) ? 1 : Library::RandomValue());

    //weights work as follows in example:
    //3 input size (0,1,2), 2 hidden size (3,4) 
    //0->3 = E0, 1->3 = E1, 2->3 = E2
    //0->4 = E3, 1->3 = E4, 2->3 = E5

    //memory for the node outputs
    CUDACHECK(cudaMallocManaged(&this->activatedOutputs, this->nodeCount * sizeof(float)));

    //some calculations to save time later
    int largestLayerSize = 0;
    int largestLayerWeightCount = 0;
    for (int i = 1; i < this->layerCount; i++) {
        int layerSize = this->layerSizes[i];
        if (layerSize > largestLayerSize)
            largestLayerSize = layerSize;

        int weightCount = layerSize * this->layerSizes[i - 1];
        if (weightCount > largestLayerWeightCount)
            largestLayerWeightCount = weightCount;
    }
    this->largestLayerSize = largestLayerSize;
    this->largestLayerWeightCount = largestLayerWeightCount;
}

void NeuralNetwork::FeedForward(float* inputArr, float* outputArr) {
    //calculate the sizes of the CUDA blocks
    int inputBlocksNeeded = (this->layerSizes[0] + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
    int sumBlocksNeeded = (this->largestLayerWeightCount + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
    int activationBlocksNeeded = (this->largestLayerSize + THREADSPERBLOCK - 1) / THREADSPERBLOCK;

    //copy the input into the relevant array, activate the values
    std::copy(inputArr, inputArr + this->layerSizes[0], this->activatedOutputs);
    CUDACHECK(cudaMemcpy(this->activatedOutputs, inputArr, this->layerSizes[0] * sizeof(float), cudaMemcpyHostToDevice));
    CUDACHECK(cudaDeviceSynchronize());

    for (int i = 0; i < this->layerSizes[0]; i++)
        Log(to_string(i) + " : " + to_string(this->activatedOutputs[i]));

    // ActivateLayer<<<inputBlocksNeeded, THREADSPERBLOCK>>>(this->activatedOutputs, this->layerSizes[0], 0, Library::minVal, Library::maxVal); //activate the input layer
    // error = cudaDeviceSynchronize();
    // if (error != cudaSuccess)
    //     return (void)Error("Failed to activate inputs");

    //iterate over the nodes, saving all of their values into activatedOutputs, fn(sum * weights) + bias to calculate the next layer outs
    int usedNodes = 0;
    int usedWeights = 0;
    for (int i = 0; i < this->layerCount - 1; i++) {

        Log("----------------------------------------------");
        // this->PrintNetwork();

        int layerSize = this->layerSizes[i];
        int nextLayerSize = this->layerSizes[i + 1]; 
        
        //sum all the outputs * weights
        Sum<<<sumBlocksNeeded, THREADSPERBLOCK>>>(this->activatedOutputs, this->weights, layerSize, nextLayerSize, usedNodes, usedWeights); 
        CUDACHECK(cudaDeviceSynchronize());

        usedNodes += layerSize;
        usedWeights += layerSize * nextLayerSize;

        // //activate the next layer's outputs
        // ActivateLayer<<<activationBlocksNeeded, THREADSPERBLOCK>>>(this->activatedOutputs, layerSize, usedNodes, Library::minVal, Library::maxVal); 
        // CUDACHECK(cudaDeviceSynchronize());
    }

    //output by copying the contents of the nodes in the output layer into the arr
    int outputSize = this->layerSizes[this->layerCount];
    for (int i = 0; i < outputSize; i++)
        outputArr[i] = this->activatedOutputs[this->nodeCount - outputSize + i];
}

void NeuralNetwork::PrintNetwork() {
    int seenNodes = 0;
    int seenWeights = 0;
    for (int layerIndex = 0; layerIndex < this->layerCount; layerIndex++) {
        int layerSize = this->layerSizes[layerIndex];

        for (int nodeIndex = 0; nodeIndex < layerSize; nodeIndex++) {
            Log("N" + to_string(seenNodes) + " - " + to_string(this->activatedOutputs[seenNodes]));
            
            if (this->layerCount > layerIndex + 1)
                for (int edgeIndex = 0; edgeIndex < this->layerSizes[layerIndex + 1]; edgeIndex++)
                    Log("   E" + to_string(nodeIndex + (edgeIndex * layerSize) + seenWeights) + " - " + to_string(this->weights[nodeIndex + (edgeIndex * layerSize) + seenWeights]));

            seenNodes++;
        } //nodes

        if (this->layerCount > layerIndex + 1)
            seenWeights += layerSize * this->layerSizes[layerIndex + 1];
    } //layers
}