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

    // Weights are segmented into layerSize incoming for each node (e.g. 0->4 are the 5 incoming weights to node (layerSize + 1))
    const size_t weightSize = layerSize*layerSize*depth;
    this->weights = new float[weightSize];
    for (size_t weightIndex = 0; weightIndex < weightSize; weightIndex++)
        this->weights[weightIndex] = Library::RandomSignedValue(1);

    // Values are just storage for feed forward values to propogate
    const size_t valueSize = layerSize*depth;
    this->values = new float[valueSize];
    for (size_t valueIndex = 0; valueIndex < valueSize; valueIndex++)
        this->values[valueIndex] = 0;
    
    // TODO: BIAS
    this->bias = nullptr;
}

Dense::~Dense() {
    delete[] weights;
    delete[] values;
    delete[] bias;
}

void Dense::activate(const float* inputs, const size_t inputSize) const {
    activate(inputs, inputSize, this->weights, this->values, this->layerSize, this->depth);
}

void Dense::activate(const float* inputs, const size_t inputSize, float* weights, float* values, size_t layerSize, const size_t depth) const {
    if (inputSize != layerSize)
        throw std::invalid_argument("Input size must equal layerSize");

    for (size_t layerIndex = 0; layerIndex < depth; layerIndex++) {
        for (size_t outNodeIndex = 0; outNodeIndex < layerSize; outNodeIndex++) {

            float sum = 0;
            for (size_t inNodeIndex = 0; inNodeIndex < layerSize; inNodeIndex++) {
                // Value of nodes in the previous layer (use input arr if first layer)
                const float nodeVal = (layerIndex != 0)
                    ? values[layerSize * (layerIndex - 1) + inNodeIndex]
                    : inputs[inNodeIndex];

                // Weights are blocks of layerSize pointing to output node of same index as block
                // E.g. 3x3 nodes*layers, weights are: block 0 = 0->2 for node 0, block 1 = 3->5 for node 1...
                const float weightVal = weights[layerSize*layerSize*(layerIndex) + outNodeIndex * layerSize + inNodeIndex];

                sum += nodeVal * weightVal;
            }


            values[layerSize * layerIndex + outNodeIndex] = sum;
        }
    }
    
}
