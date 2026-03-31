// src/visualization.cpp
#include "visualization.hpp"
#include <fstream>
#include <iostream>

namespace GraphVisual {

    void visualizate(const Graph::Graph& graph, const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << filename << "\n";
            return;
        }

        file << "digraph G {\n";
        file << "  rankdir = TB;\n";
        file << "  node [fontname=\"Arial\"];\n\n";

        file << "  node [shape=ellipse, style=filled, fillcolor=lightyellow];\n";
        for (const auto& tensor : graph.get_tensors()) {
            if(is_intermediate(tensor.get())) continue;
            file << "  \"" << tensor->get_name() << "\" [label=\"" << tensor->get_name() << "\"];\n";
        }

        file << "  node [shape=box, style=filled, fillcolor=lightblue];\n";
        for (auto& node : graph.get_nodes()) {
            file << "  \"" << node->get_name() << "\" [label=\"" << node->get_op_type() << "\"];\n";

            for (auto* in_tensor : node->get_inputs()) {
                if (!is_intermediate(in_tensor)) {
                    file << "  \"" << in_tensor->get_name() << "\" -> \"" << node->get_name() << "\";\n";
                }
            }

            for (auto* out_tensor : node->get_outputs()) {
                if (is_intermediate(out_tensor)) {
                    for (auto* consumer_node : out_tensor->get_consumers()) {
                        file << "  \"" << node->get_name() << "\" -> \"" << consumer_node->get_name() << "\";\n";
                    }
                } else {
                    file << "  \"" << node->get_name() << "\" -> \"" << out_tensor->get_name() << "\";\n";
                }
            }
        }
    file << "}\n";
}
    bool is_intermediate(const Graph::Tensor* t) { return (t->get_producer() && t->get_consumers().size() > 0); }
}
