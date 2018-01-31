#pragma once
#include <string>

class GraphicException {
public:
	std::string msg;

	GraphicException(const std::string & message) {
		msg = message;
	}
};