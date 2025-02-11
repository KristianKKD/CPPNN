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

int main() {
    //environment params, generated per iteration
    const int rows = 6;
    const int columns = 6;
    const int gridSize = rows*columns;
    const int loseVal = -1000;
    const int winVal = 1000;
    float* grid = new float[gridSize];

    //params
    const int inputSize = gridSize; //6x6 cells
    const int hiddenCount = 5;
    const int hiddenSize = 5;
    const int outputSize = 4; //left, right, up, down
    const int learningIterations = 2000;
    const float learningRate = 0.02;
    const int timeCutoff = 100; //max steps per try 

    //create policy network
    NeuralNetwork nn(inputSize, NeuralNetwork::OutputType::Softmax);
    for (int i = 0; i < hiddenCount; i++)
        nn.AddLayer(hiddenSize, true);
    nn.AddLayer(outputSize);
    nn.Build();

    //create input arr
    float inputsArr[inputSize];

    //create output arr
    float outputsArr[outputSize];
    memset(outputsArr, 0, outputSize * sizeof(float));

    for (int epoch = 0; epoch < learningIterations; epoch++) {
        int agentPos = columns + 1; //set agent pos to the top left corner next to edges
        
        //create the grid
        GenerateGrid(grid, rows, columns, agentPos, loseVal, winVal);

        //draw initial grid
        DrawGrid(grid, rows, columns, agentPos, loseVal, winVal);

        float reward = 0;
        int time = 0;
        for (; time < timeCutoff; time++) {
            //get probability distribution of moves
            std::copy(grid, grid + gridSize, inputsArr); //copy state to network
            nn.FeedForward(inputsArr, outputsArr);

            //print probability dist
            // for (int i = 0; i < outputSize; i++)
            //     std::cout << outputsArr[i] << " ";
            // std::cout << std::endl;

            //select a move based on the probabilities
            int chosenMove = Library::SampleDistribution(outputsArr, outputSize);

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

            //accumulate the reward
            reward += grid[newPos];

            //find out if agent crashed into edge or touched the win
            if (grid[newPos] == loseVal || grid[newPos] == winVal)
                break;
        }

    }

    delete[] grid;

    Log("Finished!");
    return 0;
}