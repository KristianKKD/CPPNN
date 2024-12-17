#include "shared.hpp"
#include <fstream>
#include <filesystem>

using namespace std;

string ReadFile(string path) {
    namespace fs = std::filesystem;

    string output = "";
    try {
        ifstream f(path);

        if (!f.is_open() || !f.good()) {
            Error("Error opening " + path);
        } else {
            auto perms = fs::status(path).permissions();
            
            if ((perms & fs::perms::owner_read) == fs::perms::none) {
                Error("Cannot read file, no permissions");
                return output;
            }

            string s = "";
            while (getline(f, s)) {
                if (f.fail()){
                    Error("Failed to get line in:" + path);
                    break;
                }

                output += s + "\n";
            }

            f.close();
        }
    } catch (const std::exception& ex) {
        std::cout << "ERROR" << endl;
        Error((string)ex.what());
    }

    if (output.size() > 0)
        Log(path + " was read successfully with size: " + to_string(output.size()));
    else
        Error("Failed to read " + path);

    return output;
}