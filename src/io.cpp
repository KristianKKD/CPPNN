#include "shared.hpp"
#include <neuralnetwork.cuh>
#include <fstream>
#include <filesystem>
#include <map>

namespace fs = std::filesystem;

bool OpenAndCheckFile(std::fstream& f, const string path, std::filesystem::perms permissions) {
    if (!f.is_open() || !f.good()) {
        Error("Failed to open file!");
        return false;
    }

    if ((fs::status(path).permissions() & permissions) == fs::perms::none) {
        Error("Do not have required permissions for file!");
        return false;
    }

    return true;
}

bool CheckFileExists(string path) {
    std::fstream f(path);
    return OpenAndCheckFile(f, path, fs::perms::owner_read);
}

string ReadFile(const string path) {
    std::fstream f(path);
    string output = "";

    Log("Reading " + path);
    if (OpenAndCheckFile(f, path, fs::perms::owner_read)) {
        return output;
    } else
        Log(path + " opened successfully!");

    string s = "";
    while (getline(f, s)) {
        if (f.fail()){
            Error("Failed to get line in:" + path);
            break;
        }

        output += s + "\n";
    }

    if (output.size() > 0)
        Log(path + " was read successfully with size: " + to_string(output.size()));
    else
        Error("Failed to read " + path);

    return output;
}

void SaveNetwork(const string path, const NeuralNetwork* net) {
    std::fstream f(path);

    Log("Saving neural network to " + path);
    if (OpenAndCheckFile(f, path, fs::perms::owner_write)) {
        return;
    } else
        Log(path + " opened successfully!");

    long long version = 3;

    f << version << "\n";

    f << net->weightCount << "\n";
    f << net->nodeCount << "\n";
    f << net->layerCount << "\n";

    for (size_t i = 0; i < net->weightCount; i++)
        f << net->weights[i] << ((i + 1 == net->weightCount) ? "\n" : ",");
    
    for (size_t i = 0; i < net->nodeCount - net->layerSizes[0]; i++)
        f << net->biases[i] << ((i + 1 == net->nodeCount - net->layerSizes[0]) ? "\n" : ",");
    
    for (size_t i = 0; i < LIMITLAYERCOUNT; i++)
        f << net->normLayer[i] << ((i + 1 == LIMITLAYERCOUNT) ? "\n" : ",");
}

void SaveEmbeddings(const std::map<string, int> wordMap, const string path) {
    std::fstream f(path);

    long long version = 1;

    Log("Saving neural network to " + path);
    if (OpenAndCheckFile(f, path, fs::perms::owner_write)) {
        return;
    } else
        Log(path + " opened successfully!");

    for (auto const& [key, val] : wordMap)
        f << key << "\n"; //the index in the map should correspond to the value
}

std::map<string, int> LoadEmbeddings(const string path) {
    std::fstream f(path);
    std::map<string, int> wordMap = {};

    Log("Loading embeddings from " + path);
    if (OpenAndCheckFile(f, path, fs::perms::owner_read)) {
        return wordMap;
    } else
        Log(path + " opened successfully!");


    int lineNum = 0;
    string s = "";
    while (getline(f, s)) {
        if (lineNum++ < 1) //skip version
            continue;

        if (f.fail()){
            Error("Failed to get line (" + to_string(lineNum) + ") in:" + path);
            break;
        }

        wordMap.emplace(s, wordMap.size());
    }

    return wordMap;
}

float* LoadWeights(string path) {
    return NULL;
}