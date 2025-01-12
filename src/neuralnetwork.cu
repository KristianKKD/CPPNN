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
                        
    //weights work as follows in example:
    //3 input size (0,1,2), 2 hidden size (3,4) 
    //0->3 = E0, 1->3 = E1, 2->3 = E2
    //0->4 = E3, 1->3 = E4, 2->3 = E5

    int i = blockIdx.x * blockDim.x + threadIdx.x; //this might cause an error if there aren't enough blocks/threads?

    if (i > layerSize * nextLayerSize - 1)
        return;

    int targetIn = (i % layerSize) + nodeOffset; //0, 1, 2, 0, 1, 2
    int targetWeight = i + weightOffset; //0, 1, 2, 3, 4, 5
    int targetNode = layerSize + (i / layerSize) + nodeOffset; //3, 3, 3, 4, 4, 4 
    
    atomicAdd(&activatedOutputs[targetNode], activatedOutputs[targetIn] * weights[targetWeight]);

    printf("Thread %d:      IN[%d]:%f      W[%d]:%f      OUT[%d]:%f\n", 
            i, 
            targetIn, activatedOutputs[targetIn],
            targetWeight, weights[targetWeight],
            targetNode, activatedOutputs[targetNode]);
}

__global__ void ActivateLayer(float* activatedOutputs, const int layerSize, const int nodeOffset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i > layerSize)
        return;

    Library::ActivationFunction(&activatedOutputs[nodeOffset + i]);
}

void NeuralNetwork::Build() {
    //initialize cuda device
    cudaSetDevice(Library::gpuDevice); //maybe pointless?

    //memory for the weights array
    CUDACHECK(cudaMallocManaged(&this->weights, this->weightCount * sizeof(float)));

    //randomly initialize weights
    for (int i = 0; i < this->weightCount; i++) {
        float rand = Library::RandomValue();
        if (rand > 0.5)
            rand = 1;
        else
            rand = 0.1;
        this->weights[i] = rand;
    }

    //memory for the bias array
    CUDACHECK(cudaMallocManaged(&this->biases, (this->nodeCount - this->layerSizes[0]) * sizeof(float)));

    //randomly initialize biases
    for (int i = 0; i < this->nodeCount - this->layerSizes[0]; i++) { //no bias for input layer
        // float rand = Library::RandomValue();
        // if (rand > 0.5)
        //     rand = 1;
        // else
        //     rand = 0.1;
        // this->biases[i] = rand;
        this->biases[i] = 0;
    }

    //memory for the node outputs
    CUDACHECK(cudaMallocManaged(&this->activatedOutputs, this->nodeCount * sizeof(float)));

    // //prefetch the data we know we will use soon for some small performance boost
    // CUDACHECK(cudaMemPrefetchAsync(this->activatedOutputs, this->nodeCount * sizeof(float), Library::gpuDevice));
    // CUDACHECK(cudaMemPrefetchAsync(this->weights, this->weightCount * sizeof(float), Library::gpuDevice));
    // CUDACHECK(cudaMemPrefetchAsync(this->biases, (this->nodeCount - this->layerSizes[0]) * sizeof(float), Library::gpuDevice));

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

    CUDACHECK(cudaDeviceSynchronize()); //finish operations
}

void NeuralNetwork::FeedForward(float* inputArr, float* outputArr) {
    //calculate the sizes of the CUDA blocks
    int inputBlocksNeeded = (this->layerSizes[0] + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
    int sumBlocksNeeded = (this->largestLayerWeightCount + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
    int activationBlocksNeeded = (this->largestLayerSize + THREADSPERBLOCK - 1) / THREADSPERBLOCK;

    //copy the input into the outputs array, fill other slots with biases
    CUDACHECK(cudaMemcpy(this->activatedOutputs, inputArr, this->layerSizes[0] * sizeof(float), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(this->activatedOutputs + this->layerSizes[0], this->biases, (this->nodeCount - this->layerSizes[0]) * sizeof(float), cudaMemcpyHostToDevice));
    CUDACHECK(cudaDeviceSynchronize());

    //activate the input layer
    ActivateLayer<<<inputBlocksNeeded, THREADSPERBLOCK>>>(this->activatedOutputs, this->layerSizes[0], 0); 
    CUDACHECK(cudaDeviceSynchronize());

    //iterate over the nodes, saving all of their values into activatedOutputs, fn(sum * weights) + bias to calculate the next layer outs
    int usedNodes = 0;
    int usedWeights = 0;
    int normalizationLayersUsed = 0;
    for (int layerIndex = 0; layerIndex < this->layerCount - 1; layerIndex++) {
        int layerSize = this->layerSizes[layerIndex];
        int nextLayerSize = this->layerSizes[layerIndex + 1]; 
        
        //sum all the outputs * weights
        Sum<<<sumBlocksNeeded, THREADSPERBLOCK>>>(this->activatedOutputs, this->weights, layerSize, nextLayerSize, usedNodes, usedWeights); 
        CUDACHECK(cudaDeviceSynchronize());

        //normalization layer
        if (layerIndex == normalizationLayersUsed) { //TODO, IMPLEMENT NORMALIZATION ON GPU?
            //calc mean
            float sum = 0;
            for (int i = 0; i < layerSize; i++)
                sum += this->activatedOutputs[usedNodes + i];
            if (sum == 0)
                sum += EPSILON;
            float mean = sum/(float)layerSize;

            //calc std
            float variance = 0;
            for (int i = 0; i < layerSize; i++) {
                float val = this->activatedOutputs[usedNodes + i];
                variance += (val - mean) * (val - mean);
            }
            if (variance == 0)
                variance += EPSILON;
            float std = std::sqrt(variance/(float)layerSize);

            for (int i = 0; i < layerSize; i++) {
                float val = this->activatedOutputs[usedNodes + i];
                float newVal = val - mean;
                if (newVal == 0)
                    newVal += EPSILON;
                
                newVal = this->scales[normalizationLayersUsed] * (newVal/std) + this->shifts[normalizationLayersUsed];
            }

            normalizationLayersUsed++;
        }   

        //indexing shortcut shenanigans
        usedNodes += layerSize;
        usedWeights += layerSize * nextLayerSize;

        //activate the next layer's outputs
        ActivateLayer<<<activationBlocksNeeded, THREADSPERBLOCK>>>(this->activatedOutputs, layerSize, usedNodes); 
        CUDACHECK(cudaDeviceSynchronize());
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