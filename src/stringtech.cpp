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

string ReplaceAll(string input, string oldstring, string newstring) {
	string rString = input;
	size_t pos = 0;

	while ((pos = rString.find(oldstring, pos)) != string::npos) {
		rString.replace(pos, oldstring.length(), newstring);
		pos += newstring.length();
	}

	return rString;
}