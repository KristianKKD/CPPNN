#include <shared.hpp>

namespace Neural {
    class NeuralNetwork {
        private:
            float learningRate;

            int hiddenLayerCount;
            int hiddenNodesPerLayer;

            int inputSize;
            int outputSize;
            int weightSize;
            int biasSize;
            int nodeCount;

            float* weights;
            float* biases;
            float* inputs;
            float* outputs;
            float* a; //activated outputs of nodes
        public:
            NeuralNetwork(int inputSize, int outputSize, int hiddenLayerCount, int hiddenNodesPerLayer, float learningRate);
            ~NeuralNetwork();
            void CopyWeights(float* newWeights);
            float* StoachasticGradient(const size_t batchLearnSize);
            void FeedForward(const float* inputs, float* outputs);
            void PrintNetwork();
    };
}