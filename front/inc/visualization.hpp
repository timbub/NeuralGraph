#pragma once
#include "graph.hpp"
#include <string>

namespace GraphVisual {
    void visualizate(const Graph::Graph& graph, const std::string& filename);
    bool is_intermediate(const Graph::Tensor* t);
}
