#include <neuralnetwork.cuh>
#include <shared.hpp>
#include <library.cuh>
#include <assert.h>
#include <iostream>
#include <cmath>
#include <cstring>

// Utility function to compute sigmoid
__host__ __device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

void TestNormalize() {
    int n = 100;

    vector<float> pre;
    pre.reserve(n);

    for (int i = 0; i < n; i++)
        pre.push_back(Library::RandomSignedValue() * 100);
    Library::Normalize(pre.data(), n);

    return;
}

void TestFeedForward() {
    std::cout << "Starting TestFeedForward..." << std::endl;

    const int inputSize = 3;
    const int hiddenCount = 2;
    const int hiddenSize = 2;
    const int outputSize = 3;

    NeuralNetwork nn(inputSize);
    for (int i = 0; i < hiddenCount; i++)
        nn.AddLayer(hiddenSize);
    nn.AddLayer(outputSize);
    nn.Build(); 

    assert(nn.layerCount == hiddenCount + 2);
    assert(nn.weightCount == (inputSize * hiddenSize + hiddenSize * hiddenSize * (hiddenCount - 1) + outputSize * hiddenSize));
    assert(nn.nodeCount == inputSize + hiddenCount * hiddenSize + outputSize);

    assert(nn.layerSizes[0] == inputSize);
    assert(nn.layerSizes[1] == hiddenSize);
    assert(nn.layerSizes[2] == hiddenSize);
    assert(nn.layerSizes[3] == outputSize);

    int totalBiases = nn.nodeCount - nn.layerSizes[0];

    float* hostWeights = new float[nn.weightCount];
    float* hostBiases = new float[totalBiases];


    for (int i = 0; i < nn.weightCount; i++)
        hostWeights[i] = (i % 2 == 0) ? 1.0f : 0.1f;
   
    for (int i = 0; i < totalBiases; i++) {
        if (i % 2 == 0)
            hostBiases[i] = 0.1f * (i / 2 + 1); // 0.1, 0.3, 0.5, 0.7
        else
            hostBiases[i] = -0.1f * ((i + 1) / 2 + 1); // -0.2, -0.4, -0.6
    }

    nn.SetWeights(hostWeights);
    nn.SetBiases(hostBiases);

    float inputsArr[inputSize] = { 1.0f, 2.0f, 3.0f };
    float* outputsArr = new float[outputSize];
    memset(outputsArr, 0, outputSize * sizeof(float));
    nn.FeedForward(inputsArr, outputsArr);

    // Manually compute expected outputs (ChatGPT)

    /*
    Network Architecture:
    - Input Layer: 3 nodes (0, 1, 2)
    - Hidden Layer 1: 2 nodes (3, 4)
    - Hidden Layer 2: 2 nodes (5, 6)
    - Output Layer: 3 nodes (7, 8, 9)

    Weight Indexing:
    - Weights 0-5: Input to Hidden Layer 1
    - Weights 6-9: Hidden Layer 1 to Hidden Layer 2
    - Weights 10-15: Hidden Layer 2 to Output Layer
    */

    // Step 1: Activate Inputs
    float activated_input0 = sigmoid(inputsArr[0]); // sigmoid(1.0) ≈ 0.731059
    float activated_input1 = sigmoid(inputsArr[1]); // sigmoid(2.0) ≈ 0.880797
    float activated_input2 = sigmoid(inputsArr[2]); // sigmoid(3.0) ≈ 0.952574

    // Step 2: Layer 1 (Hidden Layer 1)
    // Node 3:
    // sum3 = (activated_input0 * w0) + (activated_input1 * w1) + (activated_input2 * w2) + bias0
    //       = (0.731059 * 1.0) + (0.880797 * 0.1) + (0.952574 * 1.0) + 0.1
    //       = 0.731059 + 0.0880797 + 0.952574 + 0.1 = 1.8717127
    float sum3 = (activated_input0 * hostWeights[0]) + (activated_input1 * hostWeights[1]) + (activated_input2 * hostWeights[2]) + hostBiases[0];
    float out3 = sigmoid(sum3); // sigmoid(1.8717127) ≈ 0.8697

    // Node 4:
    // sum4 = (activated_input0 * w3) + (activated_input1 * w4) + (activated_input2 * w5) + bias1
    //       = (0.731059 * 0.1) + (0.880797 * 1.0) + (0.952574 * 0.1) + (-0.2)
    //       = 0.0731059 + 0.880797 + 0.0952574 - 0.2 = 0.8491603
    float sum4 = (activated_input0 * hostWeights[3]) + (activated_input1 * hostWeights[4]) + (activated_input2 * hostWeights[5]) + hostBiases[1];
    float out4 = sigmoid(sum4); // sigmoid(0.8491603) ≈ 0.7001

    // Step 3: Layer 2 (Hidden Layer 2)
    // Node 5:
    // sum5 = (out3 * w6) + (out4 * w7) + bias2
    //       = (0.8697 * 1.0) + (0.7001 * 0.1) + 0.3
    //       = 0.8697 + 0.07001 + 0.3 = 1.23971
    float sum5 = (out3 * hostWeights[6]) + (out4 * hostWeights[7]) + hostBiases[2];
    float out5 = sigmoid(sum5); // sigmoid(1.23971) ≈ 0.7771

    // Node 6:
    // sum6 = (out3 * w8) + (out4 * w9) + bias3
    //       = (0.8697 * 1.0) + (0.7001 * 0.1) + (-0.4)
    //       = 0.8697 + 0.07001 - 0.4 = 0.53971
    float sum6 = (out3 * hostWeights[8]) + (out4 * hostWeights[9]) + hostBiases[3];
    float out6 = sigmoid(sum6); // sigmoid(0.53971) ≈ 0.6321

    // Step 4: Layer 3 (Output Layer)
    // Node 7:
    // sum7 = (out5 * w10) + (out6 * w11) + bias4
    //       = (0.7771 * 1.0) + (0.6321 * 0.1) + 0.5
    //       = 0.7771 + 0.06321 + 0.5 = 1.3403
    float sum7 = (out5 * hostWeights[10]) + (out6 * hostWeights[11]) + hostBiases[4];
    float out7 = sigmoid(sum7); // sigmoid(1.3403) ≈ 0.7937

    // Node 8:
    // sum8 = (out5 * w12) + (out6 * w13) + bias5
    //       = (0.7771 * 1.0) + (0.6321 * 0.1) + (-0.6)
    //       = 0.7771 + 0.06321 - 0.6 = 0.2403
    float sum8 = (out5 * hostWeights[12]) + (out6 * hostWeights[13]) + hostBiases[5];
    float out8 = sigmoid(sum8); // sigmoid(0.2403) ≈ 0.5597

    // Node 9:
    // sum9 = (out5 * w14) + (out6 * w15) + bias6
    //       = (0.7771 * 1.0) + (0.6321 * 0.1) + 0.7
    //       = 0.7771 + 0.06321 + 0.7 = 1.5403
    float sum9 = (out5 * hostWeights[14]) + (out6 * hostWeights[15]) + hostBiases[6];
    float out9 = sigmoid(sum9); // sigmoid(1.5403) ≈ 0.8246

    float expectedOutputs[outputSize] = { out7, out8, out9 };
    for (int i = 0; i < outputSize; i++) {
        float diff = std::abs(outputsArr[i] - expectedOutputs[i]);
        if (diff > EPSILON) {
            std::cerr << "TestFeedForward FAILED at output index " << i
                      << ". Expected: " << expectedOutputs[i]
                      << ", Got: " << outputsArr[i] << std::endl;
            assert(false);
        }
    }

    std::cout << "TestFeedForward PASSED!" << std::endl;

    //clean up
    delete[] outputsArr;
    delete[] hostWeights;
    delete[] hostBiases;
}

void TestPerformance() {
    const int inputSize = 100;
    const int hiddenCount = 500;
    const int hiddenSize = 500;
    const int outputSize = 100;

    NeuralNetwork nn(inputSize);
    for (int i = 0; i < hiddenCount; i++)
        nn.AddLayer(hiddenSize, true);
    nn.AddLayer(outputSize);
    nn.SetGradientClipping(.1);
    nn.SetGradientRegularization(0.01);
    nn.SetInitMultipliers(0.1, 0.1);
    nn.Build();

    Log("Weight count: " + to_string(nn.weightCount));

    float inputsArr[inputSize];
    memset(inputsArr, .1, inputSize * sizeof(float));
    float outputsArr[outputSize];
    memset(outputsArr, 0, outputSize * sizeof(float));

    StartTimer();
    nn.FeedForward(inputsArr, outputsArr);
    StopTimer("Feedforward");

    float loss[outputSize];
    memset(loss, .1, outputSize * sizeof(float));

    StartTimer();
    nn.Backpropagate(loss);
    StopTimer("Loss");
}

void TestBackPropogation() {
    //params
    const int inputSize = 10;
    const int hiddenCount = 8;
    const int hiddenSize = 8;
    const int outputSize = 10;
    const int learningIterations = 2000;
    const float learningRate = 0.05;

    //create network
    NeuralNetwork nn(inputSize, NeuralNetwork::OutputType::DefaultActivated);
    nn.SetGradientClipping(1);
    nn.SetGradientRegularization(0.01);
    for (int i = 0; i < hiddenCount; i++)
        nn.AddLayer(hiddenSize, true);
    nn.AddLayer(outputSize);
    nn.Build();

    //inputs
    float inputsArr[inputSize];
    for (int i = 0; i < inputSize; i++)
        inputsArr[i] = Library::RandomValue();
    Library::Normalize(inputsArr, inputSize);

    //outputs
    float outputsArr[outputSize];
    memset(outputsArr, 0, outputSize * sizeof(float));

    //targets
    float targets[outputSize];
    for (int i = 0; i < inputSize; i++)
        targets[i] = Library::RandomValue();

    nn.FeedForward(inputsArr, outputsArr);
    float initialScore = Library::MSE(outputsArr, targets, outputSize);
    Log("Initial score: " + to_string(initialScore));

    for (int i = 0; i < learningIterations; i++) {
        nn.FeedForward(inputsArr, outputsArr);

        //manual loss calculation
        vector<float> loss(outputSize, 0);
        for (int j = 0; j < outputSize; j++) {
            float error = (outputsArr[j] - targets[j]);
            loss[j] = error;
        }

        nn.Backpropagate(loss.data());
        nn.ApplyGradients(learningRate, 1);
        float newScore = Library::MSE(outputsArr, targets, outputSize);
        if (i % 100 == 0)
            Log("Iteration " + to_string(i) + ": " + to_string(newScore));
    }
}

