#include <shared.hpp>
#include <neuralnetwork.cuh>
#include <library.cuh>
#include <map>
#include <tuple>
#include <memory>
#include <random>

extern std::mt19937 generator;

string ReadFile(string path);
void SaveEmbeddings(std::map<string, int> wordMap, string path);
//std::map LoadEmbeddings(string path);
bool CheckFileExists(string path);

bool IsDelimiter(char c) {
    const int delimiterCount = 4;
    char delimiters[delimiterCount] = {' ', '\n', '/', ','};

    for (int delimiterIndex = 0; delimiterIndex < delimiterCount; delimiterIndex++)
        if (c == delimiters[delimiterIndex])
            return true;
    
    return false;
}

string GetNextWord(string text) {
    string word = "";
    for (int textIndex = 0; textIndex < text.length(); textIndex++) {
        char c = text[textIndex];

        if (IsDelimiter(c))
            return word;
    
        word += c;
    }

    return word;
}

vector<string> SplitWords(string data) {
    vector<string> words;

    int index = 0;
    while (index < data.size()) {
        //find next word, denoted by delimiter
        string word = GetNextWord(data.substr(index));
        index += word.length();

        words.push_back(word);
    }

    return words;
}


std::map<string, int> EmbedWordsFromVector(vector<string> words) {
    std::map<string, int> wordMap = {};

    for (int i = 0; i < words.size(); i++) {
        string word = words[i];
        //if word isn't in the map, add it
        if (wordMap.count(word) == 0)
            wordMap.try_emplace(word, wordMap.size()); //value is based on the index
    }

    return wordMap;
}

void TrainChatbot(string mapPath, string dataPath) {
    Log("Training chatbot...");

    //load training file
    string data = ReadFile(dataPath);
    Log("Loaded for trianing: " + dataPath);


    //create word mappings from word to value
    Log("Loading word embeddings...");
    vector<string> words;
    std::map<string, int> wordMap = {};

    // if (CheckFileExists(mapPath))
    //     wordMap = LoadEmbeddings(mapPath);
    // else {
        words = SplitWords(data);
        wordMap = EmbedWordsFromVector(words);
        SaveEmbeddings(wordMap, mapPath);
    // }

    int datasetSize = words.size();
    int wordCount = wordMap.size();
    Log("Loaded " + to_string(wordCount) + " words");


    //training hyper params
    const int contextSize = 5; //context of 4 words + 1 for the target word
    const int trainingEpochs = 100;
    const float learningRate = 0.01;
    const float trainTestRatio = 0.8; //80% of data is used to train, 20% to train
    const int epochsPerTesting = 10;
    const int epochsPerBatchLearn = 1;

    //create neural network
    Log("Generating neural network...");
    const int layerCount = 10; //hidden layers
    const int nodesPerLayer = 10; //nodes in each hidden layer
    int inputSize = wordCount * contextSize; //onehot encoded word * context window
    int outputSize = wordCount; //onehot encoded word

    NeuralNetwork net(inputSize);
    for (int i = 0; i < layerCount; i++)
        net.AddLayer(nodesPerLayer, true);
    net.AddLayer(outputSize); //add an output layer for onehot encoded word to use

    //neural network options
    net.SetGradientRegularization(0.01);

    //initialize network
    net.Build();
    Log("Created neural network with " + to_string(layerCount+2) + " layers, input size of " +
        to_string(inputSize) + " and output size of " + to_string(outputSize) + 
        ", with hidden nodes per layer size of " + to_string(nodesPerLayer));


    //create training batches
    Log("Creating training batches...");
    vector<vector<float>> dataBatches; //collection of all batches used for training and testing
    int batchSize = wordCount * contextSize; //(onehot encoded word + context onehot words) per vector
    for (int wordIndex = 0; wordIndex < datasetSize; wordIndex++) {
        vector<float> batch(batchSize, 0); 

        for (int contextIndex = contextSize; contextIndex >= 0; contextIndex--) { //for every word, take the contextSize previous words and place them into the training batches array
            //create onehot array by initializing 0 and setting the word to 1
            vector<float> oneHotEncoded(wordCount, 0);
            if (wordIndex - contextIndex >= 0)
                oneHotEncoded[wordMap[words[wordIndex - contextIndex]]] = 1;

            //move onehot data into the batches
            std::move(oneHotEncoded.begin(), oneHotEncoded.end(), batch.begin() + (contextIndex * wordCount));
        }

        dataBatches.push_back(batch);
    }
    
    //create the train/test batches
    int trainBatchCount = (int)std::round(datasetSize * trainTestRatio);
    int testBatchCount = datasetSize - trainBatchCount;
    vector<float> trainBatches(trainBatchCount * (wordCount * contextSize), 0);
    vector<float> testBatches(testBatchCount * (wordCount * contextSize), 0);

    //assign randomly the data
    std::shuffle(dataBatches.begin(), dataBatches.end(), generator);
    for (int batchIndex = 0; batchIndex < trainBatchCount; batchIndex++)
        std::copy(dataBatches[batchIndex].begin(), dataBatches[batchIndex].end(), trainBatches.begin() + batchSize * batchIndex);
    for (int batchIndex = trainBatchCount; batchIndex < datasetSize; batchIndex++)
        std::copy(dataBatches[batchIndex].begin(), dataBatches[batchIndex].end(), testBatches.begin() + batchSize * batchIndex);

    Log("Created " + to_string(trainBatchCount) + " training batches and " + to_string(testBatchCount) +
        " test batches of a total of " + to_string(datasetSize) + " batches!");


    //train the model
    Log("Starting training...");
    //create the required memory
    vector<float> inputsArr(inputSize, 0);
    vector<float> outputsArr(outputSize, 0);
    vector<float> targetsArr(outputSize, 0);
    vector<float> error(outputSize, 0); //used between prediction and backpropogation
    vector<float> trainingError; //captures the average error in the tests every epochsPerTesting learning iterations

    for (int epoch = 0; epoch < trainingEpochs; epoch++) {
        Log("Training epoch " + to_string(epoch) + "/" + to_string(trainingEpochs));

        //train over the dataset
        for (int trainBatchIndex = 0; trainBatchIndex < trainBatchCount; trainBatchIndex++) {
            std::copy(trainBatches.begin(), trainBatches.begin() + batchSize - wordCount, inputsArr);
            std::copy(trainBatches.begin(), trainBatches.begin() + batchSize, targetsArr);
            outputsArr.assign(outputSize, 0);

            net.FeedForward(inputsArr.data(), outputsArr.data());

            Library::CalculateError(error.data(), outputsArr.data(), targetsArr.data(), outputSize);

            net.Backpropagate(error.data());
        }

        if (epoch % epochsPerBatchLearn == 0)
            net.ApplyGradients(learningRate, epochsPerBatchLearn);

        //test the model
        if (epoch % epochsPerTesting == 0)
            for (int testBatchIndex = 0; testBatchIndex < testBatchCount; testBatchIndex++) {
                std::copy(testBatches.begin(), testBatches.begin() + batchSize - wordCount, inputsArr);
                std::copy(testBatches.begin(), testBatches.begin() + batchSize, targetsArr);
                outputsArr.assign(outputSize, 0);

                net.FeedForward(inputsArr.data(), outputsArr.data());

                trainingError.push_back(Library::MSE(outputsArr.data(), targetsArr.data(), outputSize));
            }
    }
}