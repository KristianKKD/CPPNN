#include <shared.hpp>
#include <random>

class Library {
public:
    const static float minVal;
    const static float maxVal;

    static float RandomValue() {
        // Create a random device and a Mersenne Twister random number generator
        std::random_device rd;  // Obtain a random number from hardware
        std::mt19937 gen(rd()); // Seed the generator
        // Define the distribution for integers
        std::uniform_int_distribution<> distr(minVal, maxVal);

        float val = distr(gen);
        return val;
    }

    static float ActivationFunction(float value) {
        return std::clamp(1.0f / (1.0f + std::exp(-value)), minVal, maxVal); //sigmoid
        //return std::max(0.0f, value); //ReLU
    }

    static float DerActivationFunc(float value) {
        return value * (1.0f - value); //sigmoid
    }
};
const float Library::minVal = 0.0f;
const float Library::maxVal = 1.0f;

class Edge {
public:
    float weight;

    Edge() {
        weight = Library::RandomValue();
    }
};

class Node {
public:
    int id;
    float value;
    float bias;
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

    float Activate(const std::vector<Node>& previousLayer) {
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

    void Input(const std::vector<float>& inputs) {
        for (size_t i = 0; i < inputs.size(); i++)
            nodes[i].value = inputs[i];
    }
};

class NeuralNetwork {
public:
    std::vector<Layer> layers;
    float learningRate;

    NeuralNetwork(int inputCount, float lr = 0.01f) {
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

    std::vector<float> Output(const std::vector<float>& inputs) {
        std::vector<float> outputs;
        layers[0].Input(inputs);

        for (size_t i = 1; i < layers.size(); i++) //activate to generate the values, ignore the input layer
            for (size_t j = 0; j < layers[i].nodes.size(); j++)
                layers[i].nodes[j].Activate(layers[i - 1].nodes);

        for (size_t i = 0; i < layers.back().nodes.size(); i++) //collect the values
            outputs.push_back(layers.back().nodes[i].value);

        return outputs;
    }

    void RandomMutate(int learnCount, float learningRate) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> layerDist(1, static_cast<int>(layers.size()) - 2);
        std::uniform_int_distribution<int> nodeDist(0, static_cast<int>(layers[1].nodes.size()) - 1);
        std::uniform_int_distribution<int> edgeDist(0, static_cast<int>(layers[1].nodes[0].outgoingEdges.size()) - 1);
        std::uniform_real_distribution<float> weightDist(-learningRate, learningRate);

        for (int i = 0; i < learnCount; i++) {
            int layerIndex = layerDist(gen);
            int nodeIndex = nodeDist(gen);
            int edgeIndex = edgeDist(gen);
            layers[layerIndex].nodes[nodeIndex].outgoingEdges[edgeIndex].weight += weightDist(gen);
            layers[layerIndex].nodes[nodeIndex].outgoingEdges[edgeIndex].weight = std::clamp(layers[layerIndex].nodes[nodeIndex].outgoingEdges[edgeIndex].weight, Library::minVal, Library::maxVal);
        }
    }

    void BackpropogateLearn(const std::vector<float>& outputs, const std::vector<float>& targets) {
        // calculate deltas
        // output layer
        std::vector<float> outputDeltas;
        for (size_t i = 0; i < outputs.size(); i++) {
            float delta = (targets[i] - outputs[i]) * Library::DerActivationFunc(outputs[i]);
            outputDeltas.push_back(delta);
        }

        // hidden layers
        std::vector<std::vector<float>> layerDeltas = {outputDeltas};
        for (int layerIndex = static_cast<int>(layers.size()) - 2; layerIndex >= 0; layerIndex--) {
            const Layer& currentLayer = layers[layerIndex];
            const Layer& nextLayer = layers[layerIndex + 1];
            std::vector<float> currentLayerDeltas;

            for (size_t currentLayerNodeIndex = 0; currentLayerNodeIndex < currentLayer.nodes.size(); currentLayerNodeIndex++) {
                float nodeError = 0.0f;

                for (size_t nextLayerNodeIndex = 0; nextLayerNodeIndex < nextLayer.nodes.size(); nextLayerNodeIndex++) {
                    float delta = layerDeltas[0][nextLayerNodeIndex]; // 0 is used as we insert into pos 0 when updated
                    nodeError += currentLayer.nodes[currentLayerNodeIndex].outgoingEdges[nextLayerNodeIndex].weight * delta; // how much this node connection contributed to the total error
                }

                float nodeDelta = nodeError * Library::DerActivationFunc(currentLayer.nodes[currentLayerNodeIndex].value);
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
                float nodeDelta = layerDeltas[layerIndex][currentLayerNodeIndex];

                n.bias += learningRate * nodeDelta;

                for (size_t lastLayerNodeIndex = 0; lastLayerNodeIndex < lastLayer.nodes.size(); lastLayerNodeIndex++)
                    lastLayer.nodes[lastLayerNodeIndex].outgoingEdges[currentLayerNodeIndex].weight +=
                        lastLayer.nodes[lastLayerNodeIndex].value * nodeDelta * learningRate; // modify connections from last layer nodes to the target node
            }
        }
    }
};