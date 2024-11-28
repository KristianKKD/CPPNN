#include "shared.hpp"
#include <map>
#include <unordered_set>
#include <neuralnetwork.hpp>
#include <random>

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

        if (IsDelimiter(c) || std::isdigit(c)) {
            //get word
            string word = text.substr(lastStop, i-lastStop);
            lastStop = i + 1;
            
            if (word.size() == 0 || word.back() == '-' || word.back() == '\'' || word.front() == '\'') //invalid word
                continue;
            //valid word

            //add to map
            auto pos = wordHashmap.find(word);
            if (pos == wordHashmap.end()) //doesn't exist in the map yet
                wordHashmap[word] = mapSize++; //just add the word
        }
    }

    return wordHashmap;
}

string GetNextWord(string text, int pos) {
    //we won't know if we are placed within a word
    //it's not worth going back to find the word so we need to find the next word after the next delimiter
    int lastStop = -1; 

    for (int i = pos; i < text.size(); i++) {
        char c = text[i];

        if (IsDelimiter(c) || std::isdigit(c)) {
            if (lastStop == -1) { //first word
                lastStop = i + 1;
                continue;
            }

            //get word
            string word = text.substr(lastStop, i-lastStop);
            lastStop = i + 1;
            
            if (word.size() == 0 || word.back() == '-' || word.back() == '\'' || word.front() == '\'') //invalid word
                continue;
            //valid word

            return word;
        }
    }

    return std::string();
}

int main() {
    //read text, minor formatting
    string text = ReadFile("C:/Users/KrabGor/OneDrive/Programming/C++NN/shakespeare.txt");
    ToLower(text);
    text = ReplaceAll(text, "--", " ");
    
    //find all words
    //convert them into an index (i.e. 1=the, 2=man)
    //replace the text with the indexed words (the,man = 1,2)
    map<string, int> indexedWords = IndexWords(text);
    
    //convert the words to an easier to use format
    vector<string> words;
    for (const auto& pair : indexedWords)
        words.push_back(pair.first);

    int wordCount = words.size(); //0-1 in float is within this range

    //setup training
    int sentenceSize = 16; //16 word sentences max
    int epochCount = 1;
    
    //build network
    NeuralNetwork* net = new NeuralNetwork(sentenceSize);
    net->AddLayers(10, 10);
    net->Build(wordCount);

    for (int epoch = 0; epoch < epochCount; epoch++) {
        //get a 'sentence'
        //give model starting word
        //ask to predict next word
        //penalize for different words 

        //get first word
        double randPos = Library::RandomValue();
        string startingWord = GetNextWord(text, std::round(randPos * text.size()));
        // std::cout << startingWord << endl;
        double wordVal = (double)indexedWords[startingWord]/wordCount; //CHECK IF THIS MATCHES WITH THE BELOW LOOP INDEXING

        vector<double> inputs(sentenceSize, 0);
        inputs[0] = wordVal;

        for (int sentenceIndex = 1; sentenceIndex < sentenceSize; sentenceIndex++) {
            vector<double> outputs = net->Output(inputs);
            double outputVal = outputs[sentenceIndex];
            inputs[sentenceIndex] = outputVal; //remember the word

            // string nextWord = words[std::round(outputVal * wordCount)]; 
            // std::cout << " " << outputVal;
        }
        
        // std::cout << endl;
        // for (int i = 0; i < sentenceSize; i++) {
        //     std::cout << words[std::round(inputs[i] * wordCount)] << " ";
        // }
        // std::cout << endl;
    }


    return 0;
}
