#define LIMITLAYERCOUNT 1024 //for now, static limit of 1024 layers
#define THREADSPERBLOCK 256
#define EPSILON 1e-7 //tiny value to prevent divide by zero errors

class NeuralNetwork {
public:
    enum OutputType{
        Raw,
        DefaultActivated,
        Softmax,
    };

    NeuralNetwork(int inputSize, OutputType type=DefaultActivated);
    ~NeuralNetwork();
    void AddLayer(int size, bool normalized = false); //create a node layer (excluding input)
    void Build(); //initialize all the values needed for training
    void FeedForward(float* inputArr, float* outputArr);
    void PrintNetwork();
    void RandomGradientDescent(int changeCount);
    void SetWeights(const float* hostWeights);
    void SetBiases(const float* hostBiases);
    void Backpropogate(float* preds, float* targets);
    void ApplyGradients(float learningRate, int batches);

    //options
    OutputType outType = DefaultActivated;

    //counting stuff
    long long weightCount = 0;
    long long nodeCount = 0;
    
    //layer stuff
    int layerCount = 1; //assuming input layer = 0
    int layerSizes[LIMITLAYERCOUNT]; //size in node count
    bool normLayer[LIMITLAYERCOUNT]; //positions of the normalization layers (one = true, 0 = false)

    //learning stuff
    float* weightDeltas;

    //values
    float* weights; //node connection weights
    float* biases; //base node value
    float* activatedOutputs; //output values of nodes

    //normalization params
    float scales[LIMITLAYERCOUNT]; //multiplied on the norm layer
    float shifts[LIMITLAYERCOUNT]; //added on the norm layer

    //we can save some time by calculating some things early
    int largestLayerSize = 0;
    int largestLayerWeightCount = 0; //used to determine the blocksize for feedforward calculations
    
};