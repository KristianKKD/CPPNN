#include "neurallayers.hpp"
#include "library.hpp"
#include <stdexcept>

Dense::Dense(const size_t layerSize, const size_t depth) {
    if (layerSize < 1)
        throw std::invalid_argument("Layer size must be >= 1");
    if (depth < 1)
        throw std::invalid_argument("Depth (layer count) must be >= 1");

    this->layerSize = layerSize; // Nodes per layer
    this->depth = depth; // Layer count
    this->weightSize = layerSize*layerSize*depth; // Weight count
    this->valueSize = layerSize*depth; // Node count
    this->biasSize = layerSize*depth; // Node count

    // Weights are segmented into layerSize incoming for each node (e.g. 0->4 are the 5 incoming weights to node (layerSize + 1))
    this->weights = vector<float>(weightSize);
    for (size_t weightIndex = 0; weightIndex < weightSize; weightIndex++)
        this->weights[weightIndex] = Library::RandomSignedValue(1);

    // Values are just storage for feed forward values to propogate so no need for values
    this->values = vector<float>(valueSize, 0);
    
    // Bias is another 'weight' for each node
    this->bias = vector<float>(biasSize);
    for (size_t biasIndex = 0; biasIndex < biasSize; biasIndex++)
        this->bias[biasIndex] = Library::RandomSignedValue(1);
}

Dense::~Dense() {
}

void Dense::activate(const float* inputs, const size_t inputSize) {
    activate(inputs, inputSize, this->weights.data(), this->values.data(), this->bias.data(), this->layerSize, this->depth);
}

void Dense::activate(const float* inputs, const size_t inputSize, float* weights, float* values, float* bias, const size_t layerSize, const size_t depth) {
    if (inputSize != layerSize)
        throw std::invalid_argument("Input size must equal layerSize");

    for (size_t layerIndex = 0; layerIndex < depth; layerIndex++) {
        for (size_t outNodeIndex = 0; outNodeIndex < layerSize; outNodeIndex++) {

            float sum = 0;
            for (size_t inNodeIndex = 0; inNodeIndex < layerSize; inNodeIndex++) {
                size_t prevLayerNodeIndex = (layerIndex != 0)
                    ? layerSize * (layerIndex - 1) + inNodeIndex // this can overflow to max size, but shouldn't happen due to conditional above
                    : inNodeIndex;

                // Value of nodes in the previous layer (use input arr if first layer)
                const float nodeVal = (layerIndex != 0)
                    ? values[prevLayerNodeIndex]
                    : inputs[prevLayerNodeIndex];

                // Bias of nodes in previous layer (use none if first layer)
                const float biasVal = (layerIndex != 0)
                    ? bias[prevLayerNodeIndex]
                    : 0;

                // Weights are blocks of layerSize pointing to output node of same index as block
                // E.g. 3x3 nodes*layers, weights are: block 0 = 0->2 for node 0, block 1 = 3->5 for node 1...
                const float weightVal = weights[layerSize*layerSize*(layerIndex) + outNodeIndex * layerSize + inNodeIndex];

                sum += nodeVal * weightVal + biasVal;
            }


            values[layerSize * layerIndex + outNodeIndex] = sum;
        }
    }
    
}
