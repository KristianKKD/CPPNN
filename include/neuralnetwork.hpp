#include <shared.hpp>

namespace Neural {
    class NeuralNetwork {
        private:
            float learningRate;

            int inputCount;
            int outputCount;

            int hiddenLayerCount;
            int hiddenNodesPerLayer;

            size_t weightSize;
            size_t biasSize;

            float* weights;
            float* biases;
        public:
            NeuralNetwork(int inputCount, int outputCount, int hiddenLayerCount, int hiddenNodesPerLayer, float learningRate);
            ~NeuralNetwork();
            void CopyWeights(float* newWeights);
            void Backpropogate();
            float* StoachasticGradient(const size_t batchLearnSize);
            void FeedForward(const float* inputs, float* outputs);
            void PrintNetwork();
    };
}