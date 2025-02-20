#pragma once

#include <vector>

#define LIMITLAYERCOUNT 1024 //for now, static limit of 1024 layers
#define THREADSPERBLOCK 256

class NeuralNetwork {
public:
    enum OutputType{
        Raw,
        DefaultActivated,
        Softmax,
    };

    NeuralNetwork(int inputSize, OutputType type = DefaultActivated, float weightInitMultiplier = 1, float biasInitMultiplier = 1);
    ~NeuralNetwork();
    NeuralNetwork& operator=(const NeuralNetwork& net); //copy the weights and biases of the network
    void AddLayer(int size, bool normalized = false); //create a node layer (excluding input)
    void Build(); //initialize all the values needed for training
    void FeedForward(const float* inputArr, float* outputArr);
    void PrintNetwork();
    void RandomGradientDescent(int changeCount);
    void SetWeights(const float* hostWeights);
    void SetBiases(const float* hostBiases);
    void Backpropagate(const float* loss);
    void ApplyGradients(float learningRate, int batches);

    //options
    OutputType outType = DefaultActivated;
    float weightMultiplier = 1;
    float biasMultiplier = 1;

    //counting stuff
    long long weightCount = 0;
    long long biasCount = 0;
    long long nodeCount = 0;
    int layerCount = 1; //assuming input layer = 0
    
    //layer stuff
    int layerSizes[LIMITLAYERCOUNT]{}; //size in node count
    bool normLayer[LIMITLAYERCOUNT]{}; //positions of the normalization layers (one = true, 0 = false)

    //values
    float* weights; //node connection weights
    float* biases; //base node value
    float* activatedOutputs; //activated output values of nodes
    std::vector<float> preActivatedOutputs; //outputs of nodes before activation function
    std::vector<float> weightDeltas; //accumulated changes from backpropogation per weight
   
    //we can save some time by calculating some things early
    int largestLayerSize = 0;
    int largestLayerWeightCount = 0; //used to determine the blocksize for feedforward calculations
    
};