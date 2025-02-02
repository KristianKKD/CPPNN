#include "shared.hpp"
#include <fstream>
#include <filesystem>
#include <neuralnetwork.cuh>

string ReadFile(string path) {
    namespace fs = std::filesystem;

    string output = "";
    try {
        std::ifstream f(path);

        if (!f.is_open() || !f.good())
            Error("Error opening " + path);
        else {
            if ((fs::status(path).permissions() & fs::perms::owner_read) == fs::perms::none) {
                Error("Cannot read file, no permissions");
                return output;
            }

            string s = "";
            while (getline(f, s)) {
                if (f.fail()){
                    Error("Failed to get line in:" + path);
                    break;
                }

                output += s + "\n";
            }

            f.close();
        }
    } catch (const std::exception& ex) {
        Error("ERROR:" + (string)ex.what());
    }

    if (output.size() > 0)
        Log(path + " was read successfully with size: " + to_string(output.size()));
    else
        Error("Failed to read " + path);

    return output;
}

void SaveNetwork(string path, NeuralNetwork* net) {
    namespace fs = std::filesystem;

    //version, checksum, weightCount, nodeCount, layerCount, weights, biases, normLayers, scales, shifts

    try {
        std::ifstream f(path);

        if (!f.is_open() || !f.good())
            Error("Error opening " + path);
        else {
            if ((fs::status(path).permissions() & fs::perms::owner_write) == fs::perms::none)
                return (void)Error("Cannot write to " + path + ", no permissions");

            long long version = 1;
            long long checksum = 2; //static for now until i implement

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
                
            for (size_t i = 0; i < LIMITLAYERCOUNT; i++)
                f << net->scales[i] << ((i + 1 == LIMITLAYERCOUNT) ? "\n" : ",");

            for (size_t i = 0; i < LIMITLAYERCOUNT; i++)
                f << net->shifts[i] << ((i + 1 == LIMITLAYERCOUNT) ? "\n" : ",");

            f.close();
        }
    } catch (const std::exception& ex) {
        Error("ERROR:" + (string)ex.what());
    }
}

float* LoadWeights(string path) {
    return NULL;
}