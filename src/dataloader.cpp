#include "shared.hpp"
#include <fstream>

using namespace std;

string ReadFile(string path) {
    ifstream f(path);
    string output = "";

    if (!f.is_open() || !f.good()) {
        Error("Error opening " + path);
    } else {
        string s = "";
        while (getline(f, s))
            output += s + "\n";

        f.close();
    }

    return output;
}