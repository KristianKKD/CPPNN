#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <cmath>
#include <stdlib.h>

using std::string;
using std::to_string;
using std::vector;

int Error(const string msg);
void Log(const string msg);
void StartTimer();
void StopTimer(const string printText);