#include "shared.hpp"
#include <chrono>

string ReplaceAll(string input, string find, string replaceWith);

string defaultValue = "-123456"; //used as a catch all error code for string related content
int defaultInt = std::stoi(defaultValue);

void Log(string msg) {
	msg = ReplaceAll(msg, "\n", " "); //remove common characters which would disrupt the logging
	msg = ReplaceAll(msg, "\r", " ");

	std::cout << msg << std::endl;
}

int Error(string msg) {
    msg = "ERROR: " + msg;
	Log(msg);
    return 0;
}

std::chrono::high_resolution_clock::time_point startTime;
void StopTimer(string printText) {
	std::chrono::high_resolution_clock::time_point currentTime = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - startTime);

	Log(string(printText + "|	Time elapsed: " + to_string(duration.count()) + "ms"));
}

void StartTimer() {
	startTime = std::chrono::high_resolution_clock::now();
	Log("Started timer!");
}
