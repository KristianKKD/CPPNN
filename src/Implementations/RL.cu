#include "shared.hpp"
#include "neuralnetwork.cuh"

#include "GridWorld2.cpp"

//Planner
//Simulator
//Evaluator

// Plan steps (choose an action)
// Simulate steps (imagine the environment after the actions)
// Estimate score (guess the total score from the past actions cumulatively)
// Replan steps, can signal 'finished' from Planner

// Planning trains on optimizing score
// Simulation trains on reality vs sim
// Evaluator trains on score accuracy

void RL() {
    vector<char> grid = {
        '|', '|', '|', '|', '|',
        '|', '0', '-', '-', '|',
        '|', '-', '|', '-', '|',
        '|', '-', '-', 'X', '|',
        '|', '|', '|', '|', '|'
    };
    
    int gridWidth = 5;
    int gridHeight = 5;
    GridWorld world = GridWorld(grid, gridWidth, gridHeight);
    int gridCellCount = gridWidth*gridHeight;

    //greed
    const float greedChanceStart = 0.3;
    float greedChance = greedChanceStart;
    float greedStep = 0.02; //make it more greedy over time so we choose the better move more often

    //learning hyper params
    const int learningIterations = 5000;
    const float learningRate = 0.002;
    const int timeCutoff = 15; //max steps per try

    //architecture params
    const int predictedStepCount = 1;
    const int possibleStatesPerCell = 4;

    //create Evaluator network - predict output score
    const int eInputs = gridCellCount * predictedStepCount;
    const int eHiddenLayers = 4;
    const int eHiddenSize = 4;
    const int eOutputSize = 8; //8 bit binary representation of score
    NeuralNetwork evalNet(eInputs);
    for (int i = 0; i < eHiddenLayers; i++)
        evalNet.AddLayer(eHiddenSize, true);
    evalNet.AddLayer(eOutputSize);
    evalNet.SetActivationFunction(NeuralNetwork::ActivationType::Tanh);
    evalNet.SetGradientClipping(.1);
    evalNet.SetGradientRegularization(0.1);
    evalNet.SetInitMultipliers(0.1, 0.1);
    evalNet.Build();
    vector<float> eOutputsArr(eOutputSize, 0);

    //create Planner network - selects the move to make
    const int pInputs = gridCellCount + eOutputSize;
    const int pHiddenLayers = 4;
    const int pHiddenSize = 4;
    const int pOutputSize = possibleStatesPerCell * predictedStepCount + 1; //(left, right, up, down) * N, FINISH
    NeuralNetwork planNet(pInputs, NeuralNetwork::OutputType::Softmax); //softmax for probability of selection of move
    for (int i = 0; i < pHiddenLayers; i++)
        planNet.AddLayer(pHiddenSize, true);
    planNet.AddLayer(pOutputSize);
    planNet.SetActivationFunction(NeuralNetwork::ActivationType::Tanh);
    planNet.SetGradientClipping(.1);
    planNet.SetGradientRegularization(0.1);
    planNet.SetInitMultipliers(0.1, 0.1);
    planNet.Build();
    vector<float> pOutputsArr(pOutputSize, 0);

    //create Simulator network - predict what next env will look like
    const int sInputs = gridCellCount * (predictedStepCount + 1); //number of envs as input
    const int sHiddenLayers = 4;
    const int sHiddenSize = 8;
    const int sOutputSize = gridCellCount * possibleStatesPerCell * predictedStepCount; //each cell * all possible cell types * n
    NeuralNetwork simNet(sInputs);
    for (int i = 0; i < sHiddenLayers; i++)
        simNet.AddLayer(sHiddenSize, true);
    simNet.AddLayer(sOutputSize);
    simNet.SetActivationFunction(NeuralNetwork::ActivationType::Tanh);
    simNet.SetGradientClipping(.1);
    simNet.SetGradientRegularization(0.1);
    simNet.SetInitMultipliers(0.1, 0.1);
    simNet.Build();
    vector<float> sOutputsArr(sOutputSize, 0);

}