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
            NeuralNetwork(int inputCOunt, int outputCount, int hiddenLayerCount, int hiddenNodesPerLayer);
            ~NeuralNetwork();
            void Build();
            void Backpropogate();

            float* Output(float* inputCount);
    };
}