#include "library.cuh"
#include "neuralnetwork.cuh"
#include "shared.hpp"  

#define CUDACHECK(call) {                                                        \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at "     \
                      << __FILE__ << ":" << __LINE__ << std::endl;               \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    }

NeuralNetwork::NeuralNetwork(int inputSize, OutputType type) {
    this->weightCount = 0;
    this->biasCount = 0;
    this->nodeCount = 0;
    this->layerCount = 0;
    this->AddLayer(inputSize);
    this->outType = type;

    Log("Created neural network with input layer size: " + to_string(inputSize));
}

NeuralNetwork::~NeuralNetwork() {
    cudaFree(this->weights);
    cudaFree(this->biases);
    cudaFree(this->activatedOutputs);

    Log("Destroying neural network!");
}

NeuralNetwork& NeuralNetwork::operator=(const NeuralNetwork& net) {
    if (this == &net)
        return *this;
    
    //copy vals
    this->outType = net.outType;
    this->weightCount = net.weightCount;
    this->biasCount = net.biasCount;
    this->nodeCount = net.nodeCount;
    this->layerCount = net.layerCount;
    this->largestLayerSize = net.largestLayerSize;
    this->largestLayerWeightCount = net.largestLayerWeightCount;
    //can ignore the weight/bias multipliers as they are for init

    //copy layer related stuff
    std::copy(std::begin(net.layerSizes), std::begin(net.layerSizes) + net.layerCount, std::begin(this->layerSizes));
    std::copy(std::begin(net.normLayer), std::begin(net.normLayer) + net.layerCount, std::begin(this->normLayer));

    //reset memory
    if (this->weights)
        cudaFree(this->weights);
    if (this->biases)
        cudaFree(this->biases);
    if (this->activatedOutputs)
        cudaFree(this->activatedOutputs);
    
    //memory allocation
    CUDACHECK(cudaMallocManaged(&this->weights, this->weightCount * sizeof(float))); //weights
    CUDACHECK(cudaMallocManaged(&this->biases, this->biasCount * sizeof(float))); //biases
    CUDACHECK(cudaMallocManaged(&this->activatedOutputs, this->nodeCount * sizeof(float))); //node outputs
    this->preActivatedOutputs.reserve(this->nodeCount);

    //copy memory
    CUDACHECK(cudaMemcpy(this->weights, net.weights, this->weightCount * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDACHECK(cudaMemcpy(this->biases, net.biases, this->biasCount * sizeof(float), cudaMemcpyDeviceToDevice));

    Log("Copying neural network!");
    return *this;
}

void NeuralNetwork::SetInitMultipliers(float weightInitMultiplier, float biasInitMultiplier) {
    this->weightMult = weightInitMultiplier;
    this->biasMult = biasInitMultiplier;
    Log("Applying initialization multipliers to neural network! Weights: " + to_string(weightInitMultiplier) + " | Bias: " + to_string(biasInitMultiplier));
}

void NeuralNetwork::SetGradientRegularization(float gradientMultiplier) {
    this->gradientRegMult = gradientMultiplier;
    Log("Applying gradient regularization to neural network of: " + to_string(gradientMultiplier));
}

void NeuralNetwork::SetGradientClipping(float weightClipping) {
    this->weightClipping = weightClipping;
    Log("Applying gradient clipping to neural network of: " + to_string(weightClipping));
}

void NeuralNetwork::SetActivationFunction(NeuralNetwork::ActivationType t) {
    this->activation = t;
    Log("Changed neural network activation function to type: " + to_string(t));
}

void NeuralNetwork::AddLayer(int size, bool normalized) {
    int lastLayerSize = 0;
    if (this->layerCount > 0)
        lastLayerSize = this->layerSizes[this->layerCount - 1];
        
    if (this->layerCount != 0)
        this->biasCount += size;

    int newWeightCount = lastLayerSize * size;
    this->weightCount += newWeightCount;

    this->layerSizes[this->layerCount] = size;
    this->normLayer[this->layerCount] = normalized;
    this->layerCount++;
    this->nodeCount += size;
    Log("Added neural network layer of size: " + to_string(size));
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

    //TODO, 2 NODES INTO SUB ARRAY FOR ALL TARGETS, COMBINE SUB ARRAYS
}

__global__ void ActivateLayer(float* activatedOutputs, const int layerSize, const long long nodeOffset, NeuralNetwork::ActivationType activationType) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= layerSize)
        return;

    //printf("Activating index %d from value %f\n", nodeOffset + i, activatedOutputs[nodeOffset + i]);
    ActivationFunction(&activatedOutputs[nodeOffset + i], activationType);
}

void NeuralNetwork::Build() {
    if (this->layerCount > LIMITLAYERCOUNT)
        throw std::runtime_error("Too many layers!");

    //initialize cuda device
    cudaSetDevice(Library::gpuDevice); //maybe pointless?

    //memory allocation
    CUDACHECK(cudaMallocManaged(&this->weights, this->weightCount * sizeof(float))); //weights
    CUDACHECK(cudaMallocManaged(&this->biases, this->biasCount * sizeof(float))); //biases
    CUDACHECK(cudaMallocManaged(&this->activatedOutputs, this->nodeCount * sizeof(float))); //node outputs

    this->preActivatedOutputs = vector<float>(this->nodeCount, 0);
    this->weightDeltas = vector<float>(this->weightCount, 0);

    //randomly initialize weights
    for (long long i = 0; i < this->weightCount; i++) {
        float rand = Library::RandomSignedValue(this->weightMult);
        this->weights[i] = rand;
    }
    Library::Normalize(this->weights, this->weightCount);

    //randomly initialize biases
    for (int i = 0; i < this->biasCount; i++) { //no bias for input layer
        float rand = Library::RandomSignedValue(this->biasMult);
        this->biases[i] = rand;
    }
    Library::Normalize(this->biases, this->biasCount);

    // //prefetch the data we know we will use soon for some performance boost
    // CUDACHECK(cudaMemPrefetchAsync(this->activatedOutputs, this->nodeCount * sizeof(float), Library::gpuDevice));
    // CUDACHECK(cudaMemPrefetchAsync(this->weights, this->weightCount * sizeof(float), Library::gpuDevice));
    // CUDACHECK(cudaMemPrefetchAsync(this->biases, this->biasCount * sizeof(float), Library::gpuDevice));

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
    Log("Neural network built and ready for use!");
}

void NeuralNetwork::FeedForward(const float* inputArr, float* outputArr) {
    //get stats for CUDA so we don't go out of bounds
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, Library::gpuDevice);
    
    //calculate the sizes of the CUDA blocks
    int inputBlocksNeeded = min((this->layerSizes[0] + THREADSPERBLOCK - 1) / THREADSPERBLOCK, properties.maxGridSize[0]);
    int sumBlocksNeeded = min((this->largestLayerWeightCount + THREADSPERBLOCK - 1) / THREADSPERBLOCK, properties.maxGridSize[0]);
    int activationBlocksNeeded = min((this->largestLayerSize + THREADSPERBLOCK - 1) / THREADSPERBLOCK, properties.maxGridSize[0]);

    //copy the input into the outputs array, fill other slots with biases
    CUDACHECK(cudaMemcpy(this->activatedOutputs, inputArr, this->layerSizes[0] * sizeof(float), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(this->activatedOutputs + this->layerSizes[0], this->biases, this->biasCount * sizeof(float), cudaMemcpyHostToDevice));
    std::copy(this->activatedOutputs, this->activatedOutputs + this->layerSizes[0], this->preActivatedOutputs.data());
    CUDACHECK(cudaDeviceSynchronize());

    //activate the input layer
    ActivateLayer<<<inputBlocksNeeded, THREADSPERBLOCK>>>(this->activatedOutputs, this->layerSizes[0], 0, this->activation); 
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
        if (this->normLayer[layerIndex]) { 
            Library::Normalize(this->activatedOutputs, nextLayerSize, usedNodes);
            CUDACHECK(cudaDeviceSynchronize());
        }   

        //save the pre-activation values for later use
        std::copy(this->activatedOutputs + usedNodes, this->activatedOutputs + usedNodes + nextLayerSize, this->preActivatedOutputs.data() + usedNodes);
        
        //activate the next layer's outputs
        if (layerIndex == this->layerCount - 2 && this->outType == OutputType::DefaultActivated)
            ActivateLayer<<<activationBlocksNeeded, THREADSPERBLOCK>>>(this->activatedOutputs, nextLayerSize, usedNodes, this->activation); 

        CUDACHECK(cudaDeviceSynchronize());
    }

    //output by copying the contents of the nodes in the output layer into the arr
    int outputSize = this->layerSizes[this->layerCount - 1];
    for (int i = 0; i < outputSize; i++) {
        vector<float> test(this->activatedOutputs, this->activatedOutputs + this->nodeCount);
        float val = this->activatedOutputs[this->nodeCount - outputSize + i];
        outputArr[i] = val;
    }

    if (this->outType == OutputType::Softmax)
        Library::Softmax(outputArr, outputSize);
}

void NeuralNetwork::PrintNetwork() {
    Log("Printing neural network!");
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
    Log("Applied pre-generated weights to neural network!");
}

void NeuralNetwork::SetBiases(const float* hostBiases) {
    CUDACHECK(cudaMemcpy(this->biases, hostBiases, this->biasCount * sizeof(float), cudaMemcpyHostToDevice));
    CUDACHECK(cudaDeviceSynchronize());
    Log("Applied pre-generated biases to neural network!");
}

void NeuralNetwork::RandomGradientDescent(int changeCount) {
    //make changeCount changes to a random weight
    for (int i = 0; i < changeCount; i++) {
        long long randIndex = std::round(Library::RandomValue(this->weightCount - 1));

        float randChange = Library::RandomSignedValue();

        float* val = new float;
        *val = this->weights[randIndex] + randChange;
        CUDACHECK(cudaMemcpy(this->weights + randIndex, val, sizeof(float), cudaMemcpyHostToDevice));
        delete val;
    }
    CUDACHECK(cudaDeviceSynchronize());
}

void NeuralNetwork::ApplyGradients(float learningRate, int batches) {
    if (batches <= 0)
        return (void)Error("Invalid batch count: " + to_string(batches));
    if (learningRate == 0)
        return (void)Error("Learning rate cannot be 0 to update gradients!");

    for (int weightIndex = 0; weightIndex < this->weightCount; weightIndex++) {
        float delta = this->weightDeltas[weightIndex];
        if (delta == 0)
            continue;

        //float weight = this->weights[weightIndex];
        
        float change = (delta * learningRate) / batches;
        if (std::isnan(change))
            return (void)std::runtime_error("fuck");

        if (this->gradientRegMult > 0)
            change = change + this->gradientRegMult * this->weights[weightIndex];;

        this->weights[weightIndex] -= change;
    }

    this->weightDeltas = vector<float>(this->weightCount, 0);
}

void NeuralNetwork::Backpropagate(const float* loss) { //assuming that this is called after FeedForward
    //TODO, UPDATE BIASES

    int outputSize = this->layerSizes[this->layerCount - 1];
    vector<float> nodeError(this->nodeCount, 0);

    //debugging stuff (to see the values more easily)
    vector<float> lossDEBUG(loss, loss + outputSize);
    vector<float> activatedDEBUG(this->activatedOutputs, this->activatedOutputs + this->nodeCount);
    vector<float> weightsDEBUG(this->weights, this->weights + this->weightCount);

    //apply the loss to the output nodes (for hidden node error calculations)
    for (int i = 0; i < outputSize; i++) {
        int index = this->nodeCount - this->layerSizes[this->layerCount - 1] + i;
        nodeError[index] = loss[i];
    }

    int usedNodes = outputSize;
    int usedWeights = 0;
    for (int layerIndex = this->layerCount - 2; layerIndex >= 0; layerIndex--) { //start on the hidden layer behind output layer, move backwards
        int layerSize = this->layerSizes[layerIndex];
        int nextLayerSize = this->layerSizes[layerIndex + 1];
        int layerWeightCount = layerSize * nextLayerSize;

        //calculate hidden node error
        for (int nodeIndex = 0; nodeIndex < layerSize; nodeIndex++) {
            float errorSum = 0;
            for (int weightIndex = 0; weightIndex < nextLayerSize; weightIndex++) {
                int outputIndex = this->nodeCount - usedNodes + weightIndex;
                int targetWeightIndex = this->weightCount - usedWeights - layerWeightCount + nodeIndex + weightIndex * layerSize;

                float outputError = nodeError[outputIndex];
                float weight = this->weights[targetWeightIndex];

                errorSum += outputError * weight;
                if (isnan(errorSum))
                    throw std::runtime_error("Hidden node error sum exploded");

            }
            
            int inputIndex = this->nodeCount - usedNodes - layerSize + nodeIndex;
            float inputVal = this->activatedOutputs[inputIndex];

            float der = inputVal; //have to create a copy as the function modifies the input val
            DerActivationFunction(&der, this->activation);

            float error = errorSum * der;
            nodeError[inputIndex] = error;

            if (isnan(nodeError[inputIndex]))
                throw std::runtime_error("Node error exploded");
        }
        
        //calculate weight loss
        for (int weightIndex = 0; weightIndex < layerWeightCount; weightIndex++) {
            int inputIndex = this->nodeCount - usedNodes - layerSize + (weightIndex % layerSize);
            int targetWeightIndex = this->weightCount - usedWeights - layerWeightCount + weightIndex;
            int outputIndex = this->nodeCount - usedNodes + ((weightIndex == 0) ? 0 : (weightIndex / layerSize));

            float inputVal = this->activatedOutputs[inputIndex];
            float outputError = nodeError[outputIndex]; 

            float delta = inputVal * outputError;

            if (this->weightClipping > 0)
                if (delta > this->weightClipping)
                    delta = this->weightClipping;
                else if (delta < -this->weightClipping)
                    delta = -this->weightClipping;

            
            if (isnan(delta + this->weightDeltas[targetWeightIndex]) || isinf(delta))
                throw std::runtime_error("Weights exploded");

            this->weightDeltas[targetWeightIndex] += delta;
        }

        //indexing shenanigans
        usedNodes += layerSize;
        usedWeights += layerWeightCount;
    }
}