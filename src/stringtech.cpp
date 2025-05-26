#include "shared.hpp"
#include <algorithm>

void ToUpper(string& input) {
	std::transform(input.begin(), input.end(), input.begin(), [](unsigned char c) { return std::toupper(c); });
}

void ToLower(string& input) {
	std::transform(input.begin(), input.end(), input.begin(), [](unsigned char c) { return std::tolower(c); });
}

void RemoveChar(string& text, char c) {
    text.erase(std::remove(text.begin(), text.end(), c), text.end());
}

void ReplaceAll(string& input, const vector<string> findTexts, const string newText) {
	for (const string text : findTexts) {
		size_t pos = 0;
		while ((pos = input.find(text, pos)) != string::npos) {
			input.replace(pos, text.length(), newText);
			pos += newText.length();
		}
	}
}