#include <shared.hpp>
#include <neuralnetwork.cuh>
#include <library.cuh>
#include <random>

void DrawGrid(const vector<float> grid, int rows, int columns, int agentPos, int loseVal, int winVal) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            int index = i * columns + j;
            if (index == agentPos)
                std::cout << "0"; // Agent position
            else if (grid[index] == winVal)
                std::cout << "+"; // Win condition
            else if (grid[index] == loseVal)
                std::cout << "X"; // Edge
            else
                std::cout << "-"; // Regular space
        }
        std::cout << std::endl;
    }
}

void GenerateGrid(vector<float>& grid, int rows, int columns, int agentPos, float loseVal, float winVal, float generalVal) {
    int gridSize = rows * columns;
    //generate the base grid
    for (int i = 0; i < gridSize; ++i) {
        if (i <= columns || i >= gridSize - columns || //top and bottom
                i % columns == 0 || (i + 1) % columns == 0) //left and right
            grid[i] = loseVal; //edges are death
        else
            grid[i] = generalVal; //fill the rest of the grid with small negatives to punish lots of moving
    }

    grid[agentPos] = 0; //label agent as '0' to indicate no reward for moving to itself
    grid[gridSize - columns - 2] = winVal; //look for best reward which is placed away from starting pos
}

float RewardFunction(const vector<float> grid, int agentPos, int time, float timeVal, int rows, int columns, float winValue) {
    float reward = 0;
    
    reward += grid[agentPos];
    
    //distance to goal
    int goalPos = grid.size() - columns - 2;
    int goalX = goalPos % columns;
    int goalY = goalPos / columns;
    int agentX = agentPos % columns;
    int agentY = agentPos / columns;
    float distance = abs(goalX - agentX) + abs(goalY - agentY);
    reward += winValue / (winValue + distance);
    
    return reward;
}

void GridWorld() {
    //environment params, generated per iteration
    const int rows = 6;
    const int columns = 6;
    const int gridSize = rows * columns;
    const float loseVal = -50;
    const float winVal = 25;
    const float generalVal = 0;
    const float timeVal = 1;
    const int drawDelay = -100; //draw the grid every n training iterations

    vector<float> grid(gridSize);

    //policy hyper params
    const int pInputs = gridSize; //6x6 cells
    const int pHiddenLayers = 4;
    const int pHiddenSize = 4;
    const int pOutputs = 4; //left, right, up, down
    const int pBatchSize = 5; //iterations between updating the gradients

    //value hyper params
    const int vInputs = gridSize; //6x6 cells
    const int vHiddenLayers = 4;
    const int vHiddenSize = 4;
    const int vOutputs = 1; //estimated reward
    const int vBatchSize = 5; //iterations between updating the gradients

    //greed
    const float greedChanceStart = 0.3;
    float greedChance = greedChanceStart;
    float greedStep = 0.025;

    //learning hyper params
    const int learningIterations = 50000;
    const float learningRate = 0.0001;
    const int timeCutoff = 15; //max steps per try 

    //create policy network
    NeuralNetwork policyNet(pInputs, NeuralNetwork::OutputType::Softmax); //softmax for probability of selection of move
    policyNet.SetActivationFunction(NeuralNetwork::ActivationType::Tanh);
    policyNet.SetGradientClipping(.1);
    policyNet.SetGradientRegularization(0.01);
    policyNet.SetInitMultipliers(0.1, 0.1);
    for (int i = 0; i < pHiddenLayers; i++)
        policyNet.AddLayer(pHiddenSize, true);
    policyNet.AddLayer(pOutputs);
    policyNet.Build();

    //create value network
    NeuralNetwork valueNet(vInputs, NeuralNetwork::OutputType::Raw);
    for (int i = 0; i < vHiddenLayers; i++)
        valueNet.AddLayer(vHiddenSize, false);
    valueNet.SetActivationFunction(NeuralNetwork::ActivationType::Tanh);
    valueNet.SetGradientClipping(.1);
    valueNet.SetGradientRegularization(0.01);
    valueNet.SetInitMultipliers(0.1, 0.1);
    valueNet.AddLayer(vOutputs);
    valueNet.Build();

    //create policy network output arr
    vector<float> pOutputsArr(pOutputs);

    //create value network output arr
    vector<float> vOutputsArr(vOutputs);

    NeuralNetwork oldPolicy = policyNet;

    for (int epoch = 0; epoch < learningIterations; epoch++) {
        if (epoch % pBatchSize == 0)
            policyNet.ApplyGradients(learningRate, pBatchSize);

        int agentPos = columns + 1; //set agent pos to the top left corner next to edges
        
        //create the grid
        GenerateGrid(grid, rows, columns, agentPos, loseVal, winVal, generalVal);

        //draw initial grid
        if (epoch % drawDelay == 0  && drawDelay > 0)
            DrawGrid(grid, rows, columns, agentPos, loseVal, winVal);

        //vecs to save data about the path
        vector<vector<float>> states;
        vector<float> rewards;
        vector<float> chosenProbability;
        vector<int> chosenOptionIndex;

        //reset
        int time = 0;
        greedChance = greedChanceStart; 

        for (; time < timeCutoff; time++) {
            //save current state
            vector<float> state(grid);
            states.push_back(state);

            //normalize input state
            vector<float> normalizedState(state);
            Library::Normalize(normalizedState.data(), gridSize);

            //get probability distribution of moves
            policyNet.FeedForward(normalizedState.data(), pOutputsArr.data());
            valueNet.FeedForward(normalizedState.data(), vOutputsArr.data());

            //select a move based on the probabilities
            int chosenMove = Library::SampleDistribution(pOutputsArr.data(), pOutputs);

            //greedy choice
            if (rand() / float(RAND_MAX) < greedChance)
                chosenMove = rand() % pOutputs;
            greedChance -= greedStep;

            //save this choice
            chosenProbability.push_back(pOutputsArr[chosenMove]);
            chosenOptionIndex.push_back(chosenMove);

            //map the move
            int newPos = 0;
            switch(chosenMove) { 
                case 0: //left
                    newPos = agentPos - 1;
                    break;
                case 1: //right
                    newPos = agentPos + 1;
                    break;
                case 2: //up
                    newPos = agentPos - columns; 
                    break;
                case 3: //down
                    newPos = agentPos + columns;
                    break;
            }
            
            //reset the old pos to nothing
            grid[agentPos] = generalVal;
            agentPos = newPos; //it shouldn't be possible to go out of bounds of the array
            
            //draw for visual representation
            if (epoch % drawDelay == 0 && drawDelay > 0)
                DrawGrid(grid, rows, columns, agentPos, loseVal, winVal);

            //save the reward
            float reward = RewardFunction(grid, newPos, time, timeVal, rows, columns, winVal);
            rewards.push_back(reward);

            //train the value network
            float output = vOutputsArr[0];
            float loss = output - reward;
            valueNet.Backpropagate(&loss);
            if (time % vBatchSize == 0)
                valueNet.ApplyGradients(learningRate, vBatchSize);
            if (epoch % 200 == 0 && (grid[newPos] == loseVal || grid[newPos] == winVal))
                Log("Epoch:" + to_string(epoch) + "/" + to_string(learningIterations) +
                ", Time:" + to_string(time) + "/" + to_string(timeCutoff) + ", ValueLoss:" + to_string(loss) +
                ", PolicyReward:" + to_string(Library::SumVector(rewards.data(), rewards.size())));

            //find out if agent crashed into edge or touched the win
            if (grid[newPos] == loseVal || grid[newPos] == winVal)
                break;
        }
        
        //calculate cumulative loss
        vector<float> loss(pOutputs, 0);
        for (int i = 0; i < time; i++) {
            //normalize input
            vector<float> normalizedState(states[i]);
            Library::Normalize(normalizedState.data(), gridSize);

            //get the predicted loss
            valueNet.FeedForward(normalizedState.data(), vOutputsArr.data());

            //calculate advantage
            float predicatedValue = vOutputsArr[0];
            float advantage = rewards[i] - predicatedValue;

            //compare old policy prediction
            oldPolicy.FeedForward(normalizedState.data(), pOutputsArr.data());
            float oldProbability = pOutputsArr[chosenOptionIndex[i]];
            float newProbability = chosenProbability[i];
            float ratio = newProbability/oldProbability;

            float clippedLoss = std::min(ratio * advantage, std::clamp(ratio, 1- learningRate, 1 + learningRate) * advantage);
            loss[chosenOptionIndex[i]] += clippedLoss;
        }

        //average loss
        if (time > 0) {
            for (int i = 0; i < pOutputs; i++)
                loss[i] = ((loss[i] != 0) ? (loss[i] / time) : 0);

            policyNet.Backpropagate(loss.data());
        }

        if (epoch % pBatchSize == 0)
            policyNet.ApplyGradients(learningRate, pBatchSize);

        //save this policy as the old policy
        if (time > 0)
            oldPolicy = policyNet;
    }


    //watch it do what it has learnt
    int agentPos = columns + 1; //set agent pos to the top left corner next to edges
    
    //create the grid
    GenerateGrid(grid, rows, columns, agentPos, loseVal, winVal, generalVal);

    //draw initial grid
    DrawGrid(grid, rows, columns, agentPos, loseVal, winVal);

    for (int time = 0; time < timeCutoff; time++) {
        //save current state
        vector<float> state(grid);

        vector<float> normalizedState(state);
        Library::Normalize(normalizedState.data(), gridSize);

        //get probability distribution of moves
        policyNet.FeedForward(normalizedState.data(), pOutputsArr.data());

        //select a move based on the probabilities
        int chosenMove = Library::SampleDistribution(pOutputsArr.data(), pOutputs);

        //map the move
        int newPos = 0;
        switch(chosenMove) { 
            case 0: //left
                newPos = agentPos - 1;
                break;
            case 1: //right
                newPos = agentPos + 1;
                break;
            case 2: //up
                newPos = agentPos - columns; 
                break;
            case 3: //down
                newPos = agentPos + columns;
                break;
        }
        
        //reset the old pos to nothing
        grid[agentPos] = generalVal;
        agentPos = newPos; //it shouldn't be possible to go out of bounds of the array
        
        //draw for visual representation
        DrawGrid(grid, rows, columns, agentPos, loseVal, winVal);

        //find out if agent crashed into edge or touched the win
        if (grid[newPos] == loseVal || grid[newPos] == winVal)
            break;
    }


    return;
}