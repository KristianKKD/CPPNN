#include <shared.hpp>
#include <neuralnetwork.cuh>
#include <unittests.cuh>
#include <library.cuh>

void DrawGrid(const float* grid, int rows, int columns, int agentPos, int loseVal, int winVal) {
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

void GenerateGrid(float* grid, int rows, int columns, int agentPos, int loseVal, int winVal) {
    int gridSize = rows * columns;
    //generate the base grid
    for (int i = 0; i < gridSize; ++i) {
        if (i <= columns || i >= gridSize - columns || //top and bottom
                i % columns == 0 || (i + 1) % columns == 0) //left and right
            grid[i] = loseVal; //edges are death
        else
            grid[i] = -1; //fill the rest of the grid with small negatives to punish lots of moving
    }

    grid[agentPos] = 0; //label agent as '0' to indicate no reward for moving to itself
    grid[gridSize - columns - 2] = winVal; //look for best reward which is placed away from starting pos
}

float RewardFunction(const float* grid, int agentPos, int time, float timePunish) {
    float reward = 0;

    reward += grid[agentPos];
    reward -= time * timePunish;

    return reward;
}

int main() {
    //environment params, generated per iteration
    const int rows = 6;
    const int columns = 6;
    const int gridSize = rows*columns;
    const int loseVal = -1000;
    const int winVal = 1000;
    const int timePunish = 1;

    float* grid = new float[gridSize];

    //policy hyper params
    const int pInputs = gridSize; //6x6 cells
    const int pHiddenLayers = 5;
    const int pHiddenSize = 5;
    const int pOutputs = 4; //left, right, up, down
    const int pBatchSize = 1; //iterations between updating the gradients

    //value hyper params
    const int vInputs = gridSize; //6x6 cells
    const int vHiddenLayers = 2;
    const int vHiddenSize = 5;
    const int vOutputs = 1; //estimated reward
    const int vBatchSize = 1; //iterations between updating the gradients

    //learning hyper params
    const int learningIterations = 2000;
    const float learningRate = 0.02;
    const int timeCutoff = 100; //max steps per try 

    //create policy network
    NeuralNetwork policyNet(pInputs, NeuralNetwork::OutputType::Softmax); //softmax for probability of selection of move
    for (int i = 0; i < pHiddenLayers; i++)
        policyNet.AddLayer(pHiddenSize, true);
    policyNet.AddLayer(pOutputs);
    policyNet.Build();

    //create value network
    NeuralNetwork valueNet(vInputs);
    for (int i = 0; i < vHiddenLayers; i++)
        valueNet.AddLayer(vHiddenSize, false); //don't normalize the value net
    valueNet.AddLayer(vOutputs);
    valueNet.Build();

    //create policy network output arr
    float pOutputsArr[pOutputs];
    memset(pOutputsArr, 0, pOutputs * sizeof(float));

    //create value network output arr
    float vOutputsArr[vOutputs];
    memset(vOutputsArr, 0, vOutputs * sizeof(float));

    NeuralNetwork oldPolicy = policyNet;

    for (int epoch = 0; epoch < learningIterations; epoch++) {
        if (epoch % pBatchSize == 0)
            policyNet.ApplyGradients(learningRate, pBatchSize);

        int agentPos = columns + 1; //set agent pos to the top left corner next to edges
        
        //create the grid
        GenerateGrid(grid, rows, columns, agentPos, loseVal, winVal);

        //draw initial grid
        DrawGrid(grid, rows, columns, agentPos, loseVal, winVal);

        vector<vector<float>> states;
        vector<float> rewards;
        vector<float> chosenProbability;
        vector<int> chosenOptionIndex;

        int time = 0;

        for (; time < timeCutoff; time++) {
            //save current state
            vector<float> state(grid, grid + gridSize);
            states.push_back(state);

            //get probability distribution of moves
            std::copy(grid, grid + gridSize, state.data()); //copy state to network
            policyNet.FeedForward(state.data(), pOutputsArr);
            valueNet.FeedForward(state.data(), vOutputsArr);

            //print probability dist
            // for (int i = 0; i < outputSize; i++)
            //     std::cout << pOutputsArr[i] << " ";
            // std::cout << std::endl;

            //select a move based on the probabilities
            int chosenMove = Library::SampleDistribution(pOutputsArr, pOutputs);

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
            grid[agentPos] = -1;
            agentPos = newPos; //it shouldn't be possible to go out of bounds of the array
            
            //draw for visual representation
            DrawGrid(grid, rows, columns, agentPos, loseVal, winVal);

            //save the reward
            float reward = RewardFunction(grid, newPos, time, timePunish);
            rewards[time] = reward;

            //train the value network
            float loss = Library::MAE(vOutputsArr, &reward, vOutputs);
            valueNet.Backpropogate(&loss);
            if (time % vBatchSize == 0)
                valueNet.ApplyGradients(learningRate, vBatchSize);

            //find out if agent crashed into edge or touched the win
            if (grid[newPos] == loseVal || grid[newPos] == winVal)
                break;
        }
        
        //calculate cumulative loss
        vector<float> loss(pOutputs, 0);
        for (int i = 0; i < time; i++) {
            valueNet.FeedForward(states[i].data(), vOutputsArr);
            float predicatedValue = vOutputsArr[0];

            float advantage = rewards[i] - predicatedValue;

            oldPolicy.FeedForward(states[i].data(), pOutputsArr);
            float oldProbability = pOutputsArr[chosenOptionIndex[i]];
            float newProbability = chosenProbability[i];

            float ratio = newProbability/oldProbability;

            float clippedLoss = std::min(ratio * advantage, std::clamp(ratio, 1- learningRate, 1 + learningRate) * advantage);
            loss[chosenOptionIndex[i]] += clippedLoss;
        }

        //average loss
        for (int i = 0; i < pOutputs; i++)
            loss[i] = loss[i] / time;

        policyNet.Backpropogate(loss.data());

        if (epoch % pBatchSize == 0)
            policyNet.ApplyGradients(learningRate, pBatchSize);

        //save this policy as the old policy
        oldPolicy = policyNet;
    }

    delete[] grid;

    Log("Finished!");
    return 0;
}