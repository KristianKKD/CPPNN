#include "shared.hpp"
#include <neuralnetwork.cuh>
#include <fstream>
#include <filesystem>
#include <map>

bool CheckFileExists(string path) {
    std::fstream f(path);
    bool exists = OpenAndCheckFile(f, fs::perms::owner_read);

    return exists;
}

bool OpenAndCheckFile(std::fstream& f, std::filesystem::perms permissions) {
    namespace fs = std::filesystem;

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

string ReadFile(string path) {
    std::fstream f(path);

    Log("Reading " + path);
    if (OpenAndCheckFile(f, fs::perms::owner_read)) {
        return output;
    } else
        Log(path + " opened successfully!");

    string output = "";
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

void SaveNetwork(string path, NeuralNetwork* net) {
    std::fstream f(path);

    Log("Saving neural network to " + path);
    if (OpenAndCheckFile(f, fs::perms::owner_write)) {
        return;
    } else
        Log(path + " opened successfully!");

    long long version = 2;

    std::ofstream f(path);
    f.open(path);

    f << version << "\n";
    f << checksum << "\n";

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

void SaveEmbeddings(std::map<string, int> wordMap, string path) {
     std::fstream f(path);

    long long version = 1;

    Log("Saving neural network to " + path);
    if (OpenAndCheckFile(f, fs::perms::owner_write)) {
        return;
    } else
        Log(path + " opened successfully!");

    for (auto const& [key, val] : table)
        f << key << "\n"; //the index in the map should correspond to the value
}

std::map LoadEmbeddings(string path) {
    string data = ReadFile(path);

    std::map<string, int> wordMap = {};

    std::fstream f(path);

    Log("Loading embeddings from " + path);
    if (OpenAndCheckFile(f, fs::perms::owner_read)) {
        return output;
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