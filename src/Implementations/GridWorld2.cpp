#include "shared.hpp"
#include <iomanip>

class GridWorld {
private:
    int width;
    int height;
    vector<char> grid;
    int agentX, agentY;
    int targetX, targetY;

public:
    GridWorld(const vector<char>& initialGrid, int w, int h):
        width(w), height(h), grid(initialGrid), agentX(-1), agentY(-1), targetX(-1), targetY(-1) {
        
        if (grid.size() != width * height)
            Error("GridWorld: Grid size doesn't match dimensions");
        
        // Find agent and target positions
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++) {
                char cell = grid[y * width + x];
                if (cell == '0') {
                    agentX = x;
                    agentY = y;
                } else if (cell == 'X') {
                    targetX = x;
                    targetY = y;
                }
        }
        
        if (agentX == -1 || agentY == -1)
            Error("GridWorld: No agent found in grid");
        if (targetX == -1 || targetY == -1)
            Error("GridWorld: No target found in grid");
    }
    
    void print() const {
        std::cout << "\n";
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                char cell = grid[y * width + x];
                std::cout << cell << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    
    bool moveAgent(int dx, int dy) {
        int newX = agentX + dx;
        int newY = agentY + dy;
        
        // Check bounds
        if (newX < 0 || newX >= width || newY < 0 || newY >= height)
            return false;
        
        // Check for wall
        char targetCell = grid[newY * width + newX];
        if (targetCell == '|')
            return false;
        
        // Update grid
        grid[agentY * width + agentX] = '-';
        agentX = newX;
        agentY = newY;
        
        if (targetCell == 'X')
            grid[agentY * width + agentX] = '0'; // Keep both agent and target visible
        else
            grid[agentY * width + agentX] = '0';
        
        return true;
    }
    
    bool isGoalReached() const {
        return agentX == targetX && agentY == targetY;
    }
    
    int getWidth() const { return width; }
    int getHeight() const { return height; }
    int getAgentX() const { return agentX; }
    int getAgentY() const { return agentY; }
    int getTargetX() const { return targetX; }
    int getTargetY() const { return targetY; }
};

void RunGridWorld() {
    vector<char> grid = {
        '|', '|', '|', '|', '|',
        '|', '0', '-', '-', '|',
        '|', '-', '|', '-', '|',
        '|', '-', '-', 'X', '|',
        '|', '|', '|', '|', '|'
    };
    
    GridWorld world = GridWorld(grid, 5, 5);
    
    Log("Initial GridWorld:");
    world.print();
    
    // Example moves
    world.moveAgent(1, 0);  // Move right
    world.print();
    
    world.moveAgent(0, 1);  // Move down
    world.print();
    
    world.moveAgent(1, 0);  // Move right
    world.print();
    
    if (world.isGoalReached()) {
        Log("Goal reached!");
    }
}