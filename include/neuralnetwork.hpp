#include <shared.hpp>

namespace Neural {
    class NeuralNetwork {
        private:
            int inputCount;
            int outputCount;

            int hiddenLayerCount;
            int hiddenNodesPerLayer;

            float* weights;
            float* biases;
        public:
            NeuralNetwork(int inputCount, int outputCount, int hiddenLayerCount, int hiddenNodesPerLayer);
            ~NeuralNetwork();
            void Backpropogate();
            void FeedForward(const float* inputs, float* outputs);
            void PrintNetwork();
    };
}