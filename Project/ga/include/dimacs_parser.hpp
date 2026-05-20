#pragma once

#include <string>

#include "graph.hpp"

Graph parse_dimacs_col(const std::string& filepath);