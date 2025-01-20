#define LIMITLAYERCOUNT 1024 //for now, static limit of 1024 layers
#define THREADSPERBLOCK 256
#define EPSILON 1e-7 //tiny value to prevent divide by zero errors

class NeuralNetwork {
public:
    NeuralNetwork(int inputSize);
    ~NeuralNetwork();
    void AddLayer(int size); //create a node layer (excluding input)
    void Build(); //initialize all the values needed for training
    void FeedForward(float* inputArr, float* outputArr);
    void PrintNetwork();
    void GradientDescent(int changeCount);

    void NeuralNetwork::SetWeights(const float* hostWeights);
    void NeuralNetwork::SetBiases(const float* hostBiases);


    //counting stuff
    int weightCount = 0;
    int nodeCount = 0;
    
    //layer stuff
    int layerCount = 1; //assuming input layer = 0
    int layerSizes[LIMITLAYERCOUNT]; //size in node count
    int normLayerIndexes[LIMITLAYERCOUNT]; //positions of the normalization layers

    //values
    float* weights; //node connection weights
    float* biases; //base node value
    float* activatedOutputs; //used in training and debugging

    //normalization params
    float scales[LIMITLAYERCOUNT]; //multiplied on the norm layer
    float shifts[LIMITLAYERCOUNT]; //added on the norm layer

    //we can save some time by calculating some things early
    int largestLayerSize = 0;
    int largestLayerWeightCount = 0; //used to determine the blocksize for feedforward calculations
    
};