#include <shared.hpp>
#include <random>
#include <algorithm>

class Library {
public:
    const static unsigned int randomSeed;
    const static double minVal;
    const static double maxVal;

    static std::mt19937 generator;
    static std::uniform_real_distribution<> distribution;

    static double RandomValue() {
        return distribution(generator);
    }

    static void PrintVector(const vector<double>& vec) {
        for (int i = 0; i < vec.size(); i++)
            std::cout << vec[i] << " ";
        std::cout << std::endl;
    }

    static double ActivationFunction(double value) {
        return std::clamp(1.0f / (1.0f + std::exp(-value)), minVal, maxVal); //sigmoid
        //return std::max(0.0f, value); //ReLU
    }

    static double DerActivationFunc(double value) {
        return value * (1.0f - value); //sigmoid
    }

    static double CalculateMSE(const vector<double>& outputs, const vector<double>& targets) {
        if (outputs.size() != targets.size())
            return Error("Output size(" + to_string(outputs.size()) + ") does not match target size(" + to_string(targets.size()) + ")!");
        
        double sum = 0;
        for (int i = 0; i < targets.size(); i++) {
            double error = targets[i] - outputs[i];
            sum += error * error;
        }

        return sum / (double)targets.size();
    }
};
const unsigned int Library::randomSeed = 2;
const double Library::minVal = 0.0;
const double Library::maxVal = 1.0;

std::mt19937 Library::generator(randomSeed); 
std::uniform_real_distribution<> Library::distribution(Library::minVal, Library::maxVal);


class Edge {
public:
    double weight;

    Edge() {
        weight = Library::RandomValue();
    }
};

class Node {
public:
    int id;
    double value;
    double bias;
    std::vector<Edge> outgoingEdges;

    Node(int id, bool hasBias = true) {
        this->id = id;
        this->value = 0.0f;
        if (hasBias)
            this->bias = Library::RandomValue();
        else
            this->bias = 0.0f;
    }

    void AddEdges(int count) {
        for (int i = 0; i < count; i++)
            outgoingEdges.emplace_back();
    }

    double Activate(const std::vector<Node>& previousLayer) {
        value = this->bias;
        for (size_t i = 0; i < previousLayer.size(); i++) {
            const Node& n = previousLayer[i];
            value += n.value * n.outgoingEdges[this->id].weight;
        }
        value = Library::ActivationFunction(value);
        return value;
    }
};

class Layer {
public:
    int id;
    std::vector<Node> nodes;

    Layer(int layerID, int nodeCount, bool isInputLayer = false) {
        this->id = layerID;
        for (int i = 0; i < nodeCount; i++)
            nodes.emplace_back(i, !isInputLayer);
    }

    void Input(const std::vector<double>& inputs) {
        for (size_t i = 0; i < inputs.size(); i++)
            nodes[i].value = inputs[i];
    }
};

class NeuralNetwork {
public:
    std::vector<Layer> layers;
    double learningRate;


    NeuralNetwork(int inputCount, double lr = 0.01f) {
        this->layers = {Layer(0, inputCount, true)}; //input layer
        this->learningRate = lr;
    }

    void AddLayers(int layerCount, int nodeCount) {
        for (int i = 0; i < layerCount; i++)
            layers.emplace_back(layers.size() - 1, nodeCount);
    }

    void Build(int outputCount) {
        layers.emplace_back(layers.size() - 1, outputCount); //output layer

        for (size_t i = 0; i < layers.size() - 1; i++)
            for (size_t j = 0; j < layers[i].nodes.size(); j++)
                layers[i].nodes[j].AddEdges(layers[i + 1].nodes.size());
    }

    std::vector<double> Output(const std::vector<double>& inputs) {
        std::vector<double> outputs;
        layers[0].Input(inputs);

        for (size_t i = 1; i < layers.size(); i++) //activate to generate the values, ignore the input layer
            for (size_t j = 0; j < layers[i].nodes.size(); j++)
                layers[i].nodes[j].Activate(layers[i - 1].nodes);

        for (size_t i = 0; i < layers.back().nodes.size(); i++) //collect the values
            outputs.push_back(layers.back().nodes[i].value);

        return outputs;
    }

    void RandomMutate(int modificationCount, double learningRate, const vector<double>& outputs, const vector<double>& targets,
            double (*func)(const vector<double>&, const vector<double>&)) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> layerDist(1, static_cast<int>(layers.size()) - 2);
        std::uniform_int_distribution<int> nodeDist(0, static_cast<int>(layers[1].nodes.size()) - 1);
        std::uniform_int_distribution<int> edgeDist(0, static_cast<int>(layers[1].nodes[0].outgoingEdges.size()) - 1);
        std::uniform_real_distribution<double> weightDist(-learningRate, learningRate);

        for (int i = 0; i < modificationCount; i++) {
            double oldScore = func(outputs, targets); //less is better

            int layerIndex = layerDist(gen);
            int nodeIndex = nodeDist(gen);
            int edgeIndex = edgeDist(gen);

            double oldVal = layers[layerIndex].nodes[nodeIndex].outgoingEdges[edgeIndex].weight;
            layers[layerIndex].nodes[nodeIndex].outgoingEdges[edgeIndex].weight += weightDist(gen);
            layers[layerIndex].nodes[nodeIndex].outgoingEdges[edgeIndex].weight = std::clamp(layers[layerIndex].nodes[nodeIndex].outgoingEdges[edgeIndex].weight, Library::minVal, Library::maxVal);
        
            double newScore = func(outputs, targets);

            if (oldScore < newScore) //less is better
                layers[layerIndex].nodes[nodeIndex].outgoingEdges[edgeIndex].weight = oldVal;
        }
    }

    void BackpropogateLearn(const vector<double>& outputs, const vector<double>& targets) {
        // calculate deltas
        // output layer

        vector<double> outputDeltas;
        for (size_t i = 0; i < outputs.size(); i++) {
            double delta = (targets[i] - outputs[i]) * Library::DerActivationFunc(outputs[i]);
            outputDeltas.push_back(delta);
        }

        // hidden layers
        vector<vector<double>> layerDeltas = {outputDeltas};
        for (int layerIndex = static_cast<int>(layers.size()) - 2; layerIndex >= 0; layerIndex--) {
            const Layer& currentLayer = layers[layerIndex];
            const Layer& nextLayer = layers[layerIndex + 1];
            vector<double> currentLayerDeltas;

            for (size_t currentLayerNodeIndex = 0; currentLayerNodeIndex < currentLayer.nodes.size(); currentLayerNodeIndex++) {
                double nodeError = 0.0f;

                for (size_t nextLayerNodeIndex = 0; nextLayerNodeIndex < nextLayer.nodes.size(); nextLayerNodeIndex++) {
                    double delta = layerDeltas[layers.size() - 2 - layerIndex][nextLayerNodeIndex]; // 0 is used as we insert into pos 0 when updated
                    if (currentLayer.nodes[currentLayerNodeIndex].outgoingEdges.size() > nextLayerNodeIndex)
                        nodeError += currentLayer.nodes[currentLayerNodeIndex].outgoingEdges[nextLayerNodeIndex].weight * delta; // how much this node connection contributed to the total error
                }

                double nodeDelta = nodeError * Library::DerActivationFunc(currentLayer.nodes[currentLayerNodeIndex].value);
                currentLayerDeltas.push_back(nodeDelta);
            }

            layerDeltas.insert(layerDeltas.begin(), currentLayerDeltas); // add the deltas to the front of the list as the further into the current loop, the further back we are in the layers
        }

        // update weights and biases
        for (size_t layerIndex = 1; layerIndex < layers.size(); layerIndex++) { // ignore the input layer bias
            Layer& currentLayer = layers[layerIndex];
            Layer& lastLayer = layers[layerIndex - 1];

            for (size_t currentLayerNodeIndex = 0; currentLayerNodeIndex < currentLayer.nodes.size(); currentLayerNodeIndex++) {
                Node& n = currentLayer.nodes[currentLayerNodeIndex];
                double nodeDelta = layerDeltas[layerIndex - 1][currentLayerNodeIndex];

                n.bias += learningRate * nodeDelta;

                for (size_t lastLayerNodeIndex = 0; lastLayerNodeIndex < lastLayer.nodes.size(); lastLayerNodeIndex++)
                    lastLayer.nodes[lastLayerNodeIndex].outgoingEdges[currentLayerNodeIndex].weight +=
                        lastLayer.nodes[lastLayerNodeIndex].value * nodeDelta * learningRate; // modify connections from last layer nodes to the target node
            }
        }
    }
};

