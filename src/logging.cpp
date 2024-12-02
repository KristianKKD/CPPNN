#include "shared.hpp"
#include <chrono>

string ReplaceAll(string input, string find, string replaceWith);

string defaultValue = "-123456"; //used as a catch all error code for string related content
int defaultInt = std::stoi(defaultValue);

int Error(string msg) {
    msg = "ERROR: " + msg;
	Log(msg);
    return 0;
}

void Log(string msg) {
	msg = ReplaceAll(msg, "\n", " "); //remove common characters which would disrupt the logging
	msg = ReplaceAll(msg, "\r", " ");

	std::cout << msg << std::endl;
}

std::chrono::high_resolution_clock::time_point startTime, lastTime;
void CheckTimer() {
	std::chrono::high_resolution_clock::time_point currentTime = std::chrono::high_resolution_clock::now();

	auto fullDuration = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - startTime);
	auto lastDuration = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - lastTime);

	std::cout << ReplaceAll(string("Total time: " + to_string(fullDuration.count()) + " ms|Since last: " + to_string(lastDuration.count()) + "ms"), "\n", "") << "\n";

	lastTime = currentTime;
}

void RestartTimer() {
	startTime = std::chrono::high_resolution_clock::now();
}
