#include "shared.hpp"
#include "windows.h"
#include <chrono>

void ReplaceAll(string& input, const vector<string> findTexts, const string newText);

void Log(const string msg) {
	string msgCopy = msg;
	ReplaceAll(msgCopy, {"\n", "\r"}, " "); //remove common characters which would disrupt the logging

	std::cout << msgCopy << std::endl;
}

int Error(const string msg) {
	Log("ERROR: " + msg);
    return 0;
}

void SetConsoleColor(const int color) {
    SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), color);
}

void ColourPrint(const string msg, const int colour, const bool amplify) {
	int printColour = colour | (FOREGROUND_INTENSITY * amplify);
	SetConsoleColor(printColour);
	std::cout << msg;
	SetConsoleColor(7); //reset console to default
}

std::chrono::high_resolution_clock::time_point startTime;
void StopTimer(const string printText) {
	std::chrono::high_resolution_clock::time_point currentTime = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - startTime);

	Log(string(printText + "|	Time elapsed: " + to_string(duration.count()) + "ms"));
}

void StartTimer() {
	startTime = std::chrono::high_resolution_clock::now();
	Log("Started timer!");
}

void SetConsoleColor(WORD color) {
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    SetConsoleTextAttribute(hConsole, color);
}
