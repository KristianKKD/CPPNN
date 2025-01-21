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
    std::fill(this->layerSizes, this->layerSizes + LIMITLAYERCOUNT, 0); //tracking
    std::fill(this->normLayer, this->normLayer + LIMITLAYERCOUNT, false); //bool mask for normalization
    std::fill(this->scales, this->scales + LIMITLAYERCOUNT, 1); //* 1
    std::fill(this->shifts, this->shifts + LIMITLAYERCOUNT, 0); //+ 0
    //weights/biases don't matter because we will change them anyway

    this->weightCount = 0;
    this->nodeCount = 0;
    this->layerCount = 0;
    this->AddLayer(inputSize);
}

NeuralNetwork::~NeuralNetwork() {
    cudaFree(this->weights);
    cudaFree(this->biases);
    cudaFree(this->activatedOutputs);
}

void NeuralNetwork::AddLayer(int size, bool normalized = false) {
    int lastLayerSize = 0;
    if (this->layerCount > 0)
        lastLayerSize = this->layerSizes[this->layerCount - 1];

    int newWeightCount = lastLayerSize * size;
    this->weightCount += newWeightCount;

    this->layerSizes[this->layerCount] = size;
    this->normLayer[this->layerCount] = normalized;
    this->layerCount++;
    this->nodeCount += size;
}

__global__ void Sum(float* activatedOutputs, const float* weights, 
                    const int layerSize, const int nextLayerSize,
                    const long long nodeOffset, const long long weightOffset) {
                        
    //weights work as follows in example:
    //3 input size (0,1,2), 2 hidden size (3,4) 
    //0->3 = E0, 1->3 = E1, 2->3 = E2
    //0->4 = E3, 1->3 = E4, 2->3 = E5

    int i = blockIdx.x * blockDim.x + threadIdx.x; //this might cause an error if there aren't enough blocks/threads?

    if (i > layerSize * nextLayerSize - 1)
        return;

    long long targetIn = (i % layerSize) + nodeOffset; //0, 1, 2, 0, 1, 2
    long long targetWeight = i + weightOffset; //0, 1, 2, 3, 4, 5
    long long targetNode = layerSize + (i / layerSize) + nodeOffset; //3, 3, 3, 4, 4, 4 
    
    float outputVal = activatedOutputs[targetIn] * weights[targetWeight];

    atomicAdd(&activatedOutputs[targetNode], outputVal);

    // printf("Thread %d:      IN[%d]:%f      W[%d]:%f      OUT[%d]:%f     VAL:%f\n", 
    //         i, 
    //         targetIn, activatedOutputs[targetIn],
    //         targetWeight, weights[targetWeight],
    //         targetNode, activatedOutputs[targetNode],
    //         outputVal);
}

__global__ void ActivateLayer(float* activatedOutputs, const int layerSize, const long long nodeOffset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= layerSize)
        return;

    //printf("Activating index %d from value %f\n", nodeOffset + i, activatedOutputs[nodeOffset + i]);
    Library::ActivationFunction(&activatedOutputs[nodeOffset + i]);
}

void NeuralNetwork::Build() {
    if (this->layerCount > LIMITLAYERCOUNT)
        throw std::runtime_error("Too many layers!");

    //initialize cuda device
    cudaSetDevice(Library::gpuDevice); //maybe pointless?

    //memory allocation
    CUDACHECK(cudaMallocManaged(&this->weights, this->weightCount * sizeof(float))); //weights
    CUDACHECK(cudaMallocManaged(&this->biases, (this->nodeCount - this->layerSizes[0]) * sizeof(float))); //biases
    CUDACHECK(cudaMallocManaged(&this->activatedOutputs, this->nodeCount * sizeof(float))); //node outputs

    //randomly initialize weights
    for (long long i = 0; i < this->weightCount; i++) {
        float rand = Library::RandomValue();
        if (rand > 0.5)
            rand = 1;
        else
            rand = 0.1;
        this->weights[i] = rand;
    }

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

    for (int i = 0; i < this->layerCount; i++){
        this->shifts[i] = 0;
        this->scales[i] = 1;
    }

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

        int weightConnections = layerSize * this->layerSizes[i - 1];
        if (weightConnections > largestLayerWeightCount)
            largestLayerWeightCount = weightConnections;
    }
    this->largestLayerSize = largestLayerSize;
    this->largestLayerWeightCount = largestLayerWeightCount;

    CUDACHECK(cudaDeviceSynchronize()); //finish operations
}

void NeuralNetwork::FeedForward(float* inputArr, float* outputArr) {
    //get stats for CUDA so we don't go out of bounds
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, Library::gpuDevice);
    
    //calculate the sizes of the CUDA blocks
    int inputBlocksNeeded = min((this->layerSizes[0] + THREADSPERBLOCK - 1) / THREADSPERBLOCK, properties.maxGridSize[0]);
    int sumBlocksNeeded = min((this->largestLayerWeightCount + THREADSPERBLOCK - 1) / THREADSPERBLOCK, properties.maxGridSize[0]);
    int activationBlocksNeeded = min((this->largestLayerSize + THREADSPERBLOCK - 1) / THREADSPERBLOCK, properties.maxGridSize[0]);

    //copy the input into the outputs array, fill other slots with biases
    CUDACHECK(cudaMemcpy(this->activatedOutputs, inputArr, this->layerSizes[0] * sizeof(float), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(this->activatedOutputs + this->layerSizes[0], this->biases, (this->nodeCount - this->layerSizes[0]) * sizeof(float), cudaMemcpyHostToDevice));
    CUDACHECK(cudaDeviceSynchronize());

    //activate the input layer
    ActivateLayer<<<inputBlocksNeeded, THREADSPERBLOCK>>>(this->activatedOutputs, this->layerSizes[0], 0); 
    CUDACHECK(cudaDeviceSynchronize());

    //iterate over the nodes, saving all of their values into activatedOutputs, fn(sum * weights) + bias to calculate the next layer outs
    long long usedNodes = 0;
    long long usedWeights = 0;
    for (int layerIndex = 0; layerIndex < this->layerCount - 1; layerIndex++) {
        int layerSize = this->layerSizes[layerIndex];
        int nextLayerSize = this->layerSizes[layerIndex + 1]; 
        
        //sum all the outputs * weights
        Sum<<<sumBlocksNeeded, THREADSPERBLOCK>>>(this->activatedOutputs, this->weights, layerSize, nextLayerSize, usedNodes, usedWeights); 
        CUDACHECK(cudaDeviceSynchronize());

        //indexing shortcut shenanigans
        usedNodes += layerSize;
        usedWeights += layerSize * nextLayerSize;

        //normalization layer
        if (this->normLayer[layerIndex]) { //TODO, IMPLEMENT NORMALIZATION ON GPU?
            //calc mean
            float sum = 0;
            for (int sumIndex = 0; sumIndex < nextLayerSize; sumIndex++)
                sum += this->activatedOutputs[usedNodes + sumIndex];
            if (sum == 0)
                sum += EPSILON;
            float mean = sum/(float)nextLayerSize;

            //calc std
            float variance = 0;
            for (int varIndex = 0; varIndex < nextLayerSize; varIndex++) {
                float val = this->activatedOutputs[usedNodes + varIndex];
                variance += (val - mean) * (val - mean);
            }
            if (variance == 0)
                variance += EPSILON;
            float std = std::sqrt(variance/(float)nextLayerSize);

            for (int normIndex = 0; normIndex < nextLayerSize; normIndex++) {
                float val = this->activatedOutputs[usedNodes + normIndex];
                float newVal = val - mean;
                if (newVal == 0)
                    newVal += EPSILON;
                
                newVal = this->scales[layerIndex] * (newVal/std) + this->shifts[layerIndex];
                this->activatedOutputs[usedNodes + normIndex] = newVal;
                //Log("Normalized output " + to_string(usedNodes + normIndex) + " from " + to_string(val) + " -> " + to_string(newVal));
            }

            CUDACHECK(cudaDeviceSynchronize());
        }   

        //activate the next layer's outputs
        ActivateLayer<<<activationBlocksNeeded, THREADSPERBLOCK>>>(this->activatedOutputs, nextLayerSize, usedNodes); 
        CUDACHECK(cudaDeviceSynchronize());
    }

    //output by copying the contents of the nodes in the output layer into the arr
    int outputSize = this->layerSizes[this->layerCount - 1];
    for (int i = 0; i < outputSize; i++) {
        float val = this->activatedOutputs[this->nodeCount - outputSize + i];
        outputArr[i] = val;
    }
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

void NeuralNetwork::SetWeights(const float* hostWeights) {
    CUDACHECK(cudaMemcpy(this->weights, hostWeights, this->weightCount * sizeof(float), cudaMemcpyHostToDevice));
    CUDACHECK(cudaDeviceSynchronize());
}

void NeuralNetwork::SetBiases(const float* hostBiases) {
    CUDACHECK(cudaMemcpy(this->biases, hostBiases, (this->nodeCount - this->layerSizes[0]) * sizeof(float), cudaMemcpyHostToDevice));
    CUDACHECK(cudaDeviceSynchronize());
}

void NeuralNetwork::GradientDescent(int changeCount) {
    //make changeCount changes to a random weight
    for (int i = 0; i < changeCount; i++) {
        long long randIndex = std::round(Library::RandomValue() * (this->weightCount - 1));

        float randChange = Library::RandomValue();
        float randDir = (Library::RandomValue() > 0.5) ? 1 :  -1;
        randChange *= randDir;

        float* val = new float;
        *val = this->weights[randIndex] + randChange;
        CUDACHECK(cudaMemcpy(this->weights + randIndex, val, sizeof(float), cudaMemcpyHostToDevice));
        delete val;
    }
    CUDACHECK(cudaDeviceSynchronize());
}