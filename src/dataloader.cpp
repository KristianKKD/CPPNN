#include "shared.hpp"
#include <fstream>

using namespace std;

string ReadFile(string path) {
    string output = "";

    try {
        ifstream f(path);

        if (!f.is_open() || !f.good()) {
            Error("Error opening " + path);
        } else {
            string s = "";
            while (getline(f, s))
                output += s + "\n";

            f.close();
        }
    } catch (const std::exception& ex) {
        std::cout << "ERROR" << endl;
        Error((string)ex.what());
    }

    Log(path + " was read successfully with size: " + to_string(output.size()));
    return output;
}