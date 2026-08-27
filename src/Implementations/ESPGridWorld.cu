#include "shared.hpp"
#include "neuralnetwork.cuh"

#include "GridWorld2.cpp"

//Planner
//Simulator
//Estimator

// Plan steps (choose an action)
// Simulate steps (imagine the environment after the actions)
// Estimate score (guess the total score from the past actions cumulatively)
// Replan steps, can signal 'finished' from Planner

// Planning trains on optimizing score
// Simulation trains on reality vs sim
// Estimator trains on score accuracy

void RL() {
    // Environment
    vector<char> grid = {
        '|', '|', '|', '|', '|',
        '|', '0', '-', '-', '|',
        '|', '-', '|', '-', '|',
        '|', '-', '-', 'X', '|',
        '|', '|', '|', '|', '|'
    };
    
    const int gridWidth = 5;
    const int gridHeight = 5;
    GridWorld world = GridWorld(grid, gridWidth, gridHeight);
    const int gridCellCount = gridWidth * gridHeight;
    
    // Learning hyperparams
    const int learningIterations = 5000;
    const float learningRate = 0.002;
    const int timeCutoff = 15; // Max steps per try of epoch
    //greed
    const float greedChanceStart = 0.3;
    float greedChance = greedChanceStart;
    float greedStep = 0.02; // Make it more greedy over time so we choose the better move more often

    // Architecture params
    const int possibleActions = 4;
    const int possibleStatesPerCell = 4;

    // Create Estimator network - predict output score
    const int eInputSize = gridCellCount;
    const int eHiddenLayers = 4;
    const int eHiddenSize = 4;
    const int eOutputSize = 8; // 8 bit binary representation of score
    NeuralNetwork estNet(eInputSize);
    for (int i = 0; i < eHiddenLayers; i++)
        estNet.AddLayer(eHiddenSize, true);
    estNet.AddLayer(eOutputSize);
    estNet.SetActivationFunction(NeuralNetwork::ActivationType::Tanh);
    estNet.SetGradientClipping(.1);
    estNet.SetGradientRegularization(0.1);
    estNet.SetInitMultipliers(0.1, 0.1);
    estNet.Build();
    vector<float> eOutputsArr(eOutputSize, 0);

    // Create Planner network - selects the move to make
    const int pInputSize = gridCellCount + eOutputSize;
    const int pHiddenLayers = 4;
    const int pHiddenSize = 4;
    const int pOutputSize = possibleActions + 1; //(up, down, left, right), FINISH
    NeuralNetwork planNet(pInputSize, NeuralNetwork::OutputType::Activated); 
    for (int i = 0; i < pHiddenLayers; i++)
        planNet.AddLayer(pHiddenSize, true);
    planNet.AddLayer(pOutputSize);
    planNet.SetActivationFunction(NeuralNetwork::ActivationType::Tanh);
    planNet.SetGradientClipping(.1);
    planNet.SetGradientRegularization(0.1);
    planNet.SetInitMultipliers(0.1, 0.1);
    planNet.Build();
    vector<float> pOutputsArr(pOutputSize, 0);

    // Create Simulator network - predict what next env will look like
    const int sInputSize = gridCellCount + pOutputSize; // Initial env + predicted step
    const int sHiddenLayers = 4;
    const int sHiddenSize = 8;
    const int sOutputSize = gridCellCount * possibleStatesPerCell; // Each cell * all possible cell types
    NeuralNetwork simNet(sInputSize);
    for (int i = 0; i < sHiddenLayers; i++)
        simNet.AddLayer(sHiddenSize, true);
    simNet.AddLayer(sOutputSize);
    simNet.SetActivationFunction(NeuralNetwork::ActivationType::Tanh);
    simNet.SetGradientClipping(.1);
    simNet.SetGradientRegularization(0.1);
    simNet.SetInitMultipliers(0.1, 0.1);
    simNet.Build();
    vector<float> sOutputsArr(sOutputSize, 0);

    // Training
    for (int epoch = 0; epoch < learningIterations; epoch++) {
        // Reset environment
        world.Reset();

        // Normalize environment
        vector<float> encodedGrid = world.EncodedGrid(); // TODO: IMPLEMENT ENCODED WORLD GRID

        // Prediction
        // Planner prediction
        // Plan steps (choose an action)
        planNet.FeedForward(encodedGrid.data() + eOutputsArr.data(), pOutputsArr.data());
        // Copy outputs (so we can learn from raw output due to limation of our backprop)
        vector<float> planOutput = pOutputsArr;

        // Softmax output for probability of selection of move
        Library::Softmax(planOutput, pOutputSize);
        int chosenMove = Library::SampleDistribution(pOutputsArr.data(), pOutputSize);

        // One-hot-encode the output
        vector<float> oneHotEncodedPlan(pOutputSize, 0);
        oneHotEncodedPlan[chosenMove] = 1;
        ///////


        // Simulator prediction
        // Aggregate inputs
        vector<float> simInput = encodedGrid.data() + oneHotEncodedPlan.data();

        // Simulate steps (imagine the environment after the actions)
        simNet.FeedForward(simInput.data(), sOutputsArr.data());
        
        // Copy outputs (so we can learn from raw output due to limation of our backprop)
        vector<float> simOutput = sOutputsArr;

        // Threshold the array so we get encoded outputs
        Library::Threshold(sOutputArr.data(), 0.5);
        ///////


        // Estimator prediction
        // Estimate score (guess the total score from the past simulated actions cumulatively)
        estNet.FeedForward(sOutputArr.data(), eOutputsArr.data());
        
        // Copy outputs (so we can learn from raw output due to limation of our backprop)
        vector<float> estOutput = eOutputsArr;

        // Threshold the array so we get encoded outputs
        Library::Threshold(eOutputsArr.data(), 0.5);
        ///////


        // Learning
        // Planner learning
        // Planning trains on optimizing score
        ///////

        // Simulator learning
        // Simulation trains on reality vs sim
        // Find reality
        switch (chosenMove) {
            case 0: // Up
                world.MoveAgent(0, 1);
                break;
            case 1: // Down
                world.MoveAgent(0, -1);
                break;
            case 2: // Left
                world.MoveAgent(-1, 0);
                break;
            case 4: // Right
                world.MoveAgent(1, 0);
                break;
        }


        // Encode reality
        vector<float> resultantGrid = world.EncodeGrid();

        // Compare reality to prediction
        vector<float> simLoss(sOutputSize, 0);
        for (int sOutIndex = 0; sOutIndex < sOutputSize; sOutIndex++) {
            simLoss[sOutIndex] = sOutputArr[sOutIndex] - resultantGrid[sOutIndex];
        }

        // Learn
        simNet.Backpropagate(simLoss.data());
        ///////


    }


    // Replan steps, can signal 'finished' from Planner

    // Estimator trains on score accuracy

    }
}