#include "shared.hpp"
#include <map>
#include <unordered_set>
#include <neuralnetwork.hpp>
#include <random>

string ReadFile(string path);
void ToLower(string& input);
void RemoveChar(string& text, char c);
string ReplaceAll(string input, string oldstring, string newstring);

using namespace std;

bool IsDelimiter(char c) {
    static const std::unordered_set<char> delimiters = {
        ' ', '\t', '\n', '\r', '\v', '\f',   // Whitespace characters
        ';', ',', '.', '!', '?', ':',        // Punctuation
        '"', '(', ')', '[', ']', '{', '}', // Brackets and quotes
        '_', '=', '+', '/', '\\',       // Arithmetic and separator characters
        '<', '>', '@', '#', '$', '%', '^', '&', '*', // Symbols
        '~', '`'  // Miscellaneous symbols
    };
    return delimiters.find(c) != delimiters.end();
}

map<string, int> IndexWords(string text) {
    map<string, int> wordHashmap;
    int mapSize = 0;
    
    int lastStop = 0;
    for (int i = 0; i < text.size(); i++) {
        char c = text[i];

        if (IsDelimiter(c) || std::isdigit(c) || i == text.size()) {
            //get word
            string word = text.substr(lastStop, i-lastStop);
            lastStop = i + 1;
            
            if (word.size() == 0 || word.back() == '-' || word.front() == '-' || word.back() == '\'' || word.front() == '\'') //invalid word
                continue;
            //valid word

            //add to map
            auto pos = wordHashmap.find(word);
            if (pos == wordHashmap.end()) //doesn't exist in the map yet
                wordHashmap[word] = mapSize++; //just add the word
        }
    }

    return wordHashmap;
}

vector<string> GetNextNWords(string text, int pos, int n) {
    //we won't know if we are placed within a word
    //it's not worth going back to find the word so we need to find the next word after the next delimiter
    int lastStop = pos; 

    vector<string> words;

    for (int i = pos; i < text.size(); i++) {
        char c = text[i];

        if (IsDelimiter(c) || std::isdigit(c) || i == text.size()) {
            //get word
            string word = text.substr(lastStop, i-lastStop);
            lastStop = i + 1;
            
            if (word.size() == 0 || word.back() == '-' || word.back() == '\'' || word.front() == '\'') //invalid word
                continue;
            //valid word

            words.push_back(word);
            if (n > 0 && words.size() >= n)
                break;
        }
    }

    return words;
}

int main() {
    //read text, minor formatting
    string path = "C:/Users/KrabGor/OneDrive/Programming/C++NN/shakespeare.txt";
    string text = ReadFile(path);
    ToLower(text);
    text = ReplaceAll(text, "--", " ");

    if (text.size() <= 0)
        return Error("Failed to read " + path);

    //find all words
    //convert them into an index (i.e. 1=the, 2=man)
    //replace the text with the indexed words (the,man = 1,2)
    map<string, int> indexedWords = IndexWords(text);

    //convert the words to an easier to use format
    vector<string> words;
    for (const auto& pair : indexedWords)
        words.push_back(pair.first);

    int wordCount = words.size(); //0-1 in float is within this range
    Log("Found " + to_string(wordCount) + " unique words!");

    //ATTEMPT 2
    //setup training
    int batchSize = 4; //sentence size
    int epochCount = 1;
    
    //collect data for further processing
    vector<string> allDataWords = GetNextNWords(text, 0, -1); //get all words
    int datasetSize = allDataWords.size() - (allDataWords.size() % batchSize); //divisible by batchSize for batches
    Log("Dataset is " + to_string(datasetSize) + " words long!");

    //convert words to useable values for model
    vector<double> allDataConverted;
    for (int i = 0; i < datasetSize; i++)
        allDataConverted.push_back((double)indexedWords[allDataWords[i]]/wordCount);
   
   //setup batches for training
    double trainingPercentage = 0.8;
    int trainingBatchCount = std::round((datasetSize / batchSize) * trainingPercentage);
    
    //collect the data into batches
    Log("Creating batches...");
    vector<vector<double>> batches;
    for (int i = 0; i < datasetSize;) {
        vector<double> subset(allDataConverted.begin() + i, allDataConverted.begin() + i + batchSize);
        batches.push_back(subset);
        i += batchSize;
    }
    Log("Finished creating batches...");
    
    //shuffle the batches
    mt19937 g(Library::randomSeed);  // Mersenne Twister random number generator
    shuffle(batches.begin(), batches.end(), g);

    //split the batches into training and verification
    vector<vector<double>> trainingBatches;
    vector<vector<double>> verificationBatches;
    trainingBatches.assign(batches.begin(), batches.begin() + trainingBatchCount);
    verificationBatches.assign(batches.begin() + trainingBatchCount, batches.end());

    //build network
    NeuralNetwork* net = new NeuralNetwork(batchSize);
    net->AddLayers(10, 10);
    net->Build(batchSize);
    Log("Built network with " + to_string(net->layers.size()) + " layers!");
   
   //train
    // for (int epoch = 0; epoch < epochCount; epoch++) {
    //     Log("Epoch: " + to_string(epoch));

    //     for (int batchIndex = 0; batchIndex < batches.size(); batchIndex++) { 
    //         vector<double> targets = batches[batchIndex];
    //         vector<double> inputs(batchSize, 0);

    //         int randInputCount = std::max(1, (int)round(Library::RandomValue() * batchSize) - 1);
    //         for (int i = 0; i < randInputCount; i++)
    //             inputs[i] = targets[i];
            
    //         vector<double> outputs = net->Output(inputs);
    //         net->BackpropogateLearn(outputs, targets);
    //     }
    // }
    
    // double trainingErrorVal = Library::CalculateMSE(outputs, targets);
    // if (epochCount > 10 && epoch % (int)std::round(epochCount/10.0) == 0 && batchIndex == 0)
    //     Log("Epoch: " + to_string(epoch) + " - training error: " + to_string(trainingErrorVal));

    //vector<double> trainingError = Train(text, indexedWords, net, wordCount, epochCount, sentenceSize);
    //Log("Finished training, starting error: " + to_string(trainingError[0]) + ", new error: " + to_string(trainingError.back()));

    string line = "";
    while (line != "exit") {
        Log("Enter input:");
        getline(cin, line);

        //collect inputs into a useable format
        vector<string> uiWords = GetNextNWords(line, 0, -1);
        vector<double> uiInputVals(batchSize, 0);
        for (int i = 0; i < min((int)uiWords.size(), (int)batchSize); i++) {
            Log(uiWords[i] + " - " + to_string((double)indexedWords[uiWords[i]]/wordCount));
            uiInputVals[i] = (double)indexedWords[uiWords[i]] / wordCount;
        }

        vector<double> uiOutputs = net->Output(uiInputVals);

        for (int i = uiWords.size(); i < batchSize; i++) 
            std::cout << words[std::round(uiOutputs[i] * wordCount)] << " ";
        std::cout << endl;

    }   

    return 0;
}

vector<double> Train(string text, map<string, int> indexedWords, NeuralNetwork* net, int wordCount, int epochCount, int sentenceSize) {
    vector<double> trainingError;
     for (int epoch = 0; epoch < epochCount; epoch++) {
        //get random set of words, treat it as a target sentence
        double randPos = Library::RandomValue();
        vector<string> targetWords = GetNextNWords(text, round(randPos * text.size()), sentenceSize);
        vector<double> targetWordVals;
        for (int i = 0; i < sentenceSize; i++)
            targetWordVals.push_back((double)indexedWords[targetWords[i]]/wordCount);

        //give a random subset of the sentence as input, the model must complete the sentence
        int randCount = round((double)Library::RandomValue() * (sentenceSize - 1));
        vector<double> inputs(sentenceSize, 0);
        for (int i = 0; i < randCount; i++)
            inputs[i] = targetWordVals[i];

        //predict the rest of the sentence
        vector<double> outputs;
        for (int sentenceIndex = randCount; sentenceIndex < sentenceSize; sentenceIndex++) {
            outputs = net->Output(inputs);
            
            net->BackpropogateLearn(outputs, targetWordVals);
            //net->RandomMutate(10, net->learningRate, outputs, targetWordVals, &Library::CalculateMSE);

            inputs[sentenceIndex] = targetWordVals[sentenceIndex];
        }
        
        double trainingErrorVal = Library::CalculateMSE(outputs, targetWordVals);
        trainingError.push_back(trainingErrorVal);

        if (epochCount > 10 && epoch % (int)std::round(epochCount/10.0) == 0)
            Log("Epoch: " + to_string(epoch) + " - training error: " + to_string(trainingErrorVal));
    }

    return trainingError;
}