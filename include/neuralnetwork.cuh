#define LIMITLAYERCOUNT 128 //for now, static limit of 128 layers
#define THREADSPERBLOCK 128

class NeuralNetwork {
    public:
    int weightCount = 0;
    int nodeCount = 0;
    
    int layerCount = 1; //assuming input layer = 0
    int layerSizes[LIMITLAYERCOUNT]; 

    float* weights; //node connection weights
    float* activatedOutputs; //used in training and debugging

    //we can save some time by calculating some things early
    int largestLayerSize = 0;
    int largestLayerWeightCount = 0; //used to determine the blocksize for feedforward calculations

    NeuralNetwork(int inputSize);
    ~NeuralNetwork();
    void AddLayer(int size); //create a node layer (excluding input)
    void Build(bool debug); //initialize all the values needed for training
    void FeedForward(float* inputArr, float* outputArr);
    void PrintNetwork();
};