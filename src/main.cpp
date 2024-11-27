#include "shared.hpp"
#include <map>
#include <unordered_set>

//#include <neuralnetwork.hpp>

string ReadFile(string path);
void ToLower(string& input);
void RemoveChar(string& text, char c);
string ReplaceAll(string input, string oldstring, string newstring);

using namespace std;

bool IsDelimiter(char c) {
    static const std::unordered_set<char> delimiters = {
        ' ', '\t', '\n', '\r', '\v', '\f',   // Whitespace characters
        ';', ',', '.', '!', '?', ':',        // Punctuation
        '"', '(', ')', '[', ']', '{', '}', // Brackets and quotes
        '_', '=', '+', '/', '\\',       // Arithmetic and separator characters
        '<', '>', '@', '#', '$', '%', '^', '&', '*', // Symbols
        '~', '`'  // Miscellaneous symbols
    };
    return delimiters.find(c) != delimiters.end();
}

map<string, int> IndexWords(string text) {
    map<string, int> wordHashmap;
    int mapSize = 0;
    
    int lastStop = 0;
    for (int i = 0; i < text.size(); i++) {
        char c = text[i];

        if (IsDelimiter(c)) {
            //get word
            string word = text.substr(lastStop, i-lastStop);
            lastStop = i + 1;
            
            if (word.size() == 0 || word.back() == '-' || word.back() == '\'' || word.front() == '\'') //invalid word
                continue;

            //add to map
            auto pos = wordHashmap.find(word);
            if (pos == wordHashmap.end()) //doesn't exist in the map yet
                wordHashmap[word] = mapSize++;
        }
    }

    return wordHashmap;
}

int main() {
    //NeuralNetwork* net = new NeuralNetwork(3);
    //vector<float> inputs;

    string text = ReadFile("../../shakespeare.txt");
    ToLower(text);
    text = ReplaceAll(text, "--", " ");

    //find all words
    //convert them into an index (i.e. 1=the, 2=man)
    //replace the text with the indexed words (the,man = 1,2)

    map<string, int> indexedWords = IndexWords(text);

    for (const auto& pair : indexedWords) {
        std::cout << pair.first << ": " << pair.second << std::endl;
    }

    return 0;
}
