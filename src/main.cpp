#include <iostream>
#include <fstream>
#include "onnx.pb.h"
#include "graph.hpp"
#include "parser.hpp"
#include "visualization.hpp"

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Write: <input.onnx> <output.dot>\n";
    }

    std::string input_onnx = argv[1];
    std::string output_dot = argv[2];

    onnx::ModelProto model;
    std::ifstream input_stream(input_onnx, std::ios::binary);

    if (!input_stream.is_open()) {
        std::cerr << "Error: Cannot open file " << input_onnx << "\n";
    }

    if (!model.ParseFromIstream(&input_stream)) {
        std::cerr << "Error: Failed to parse ONNX model." << "\n";
    }

    Graph::Graph graph;
    parse_graph(graph, model.graph());

    GraphVisual::visualizate(graph, output_dot);

    return 0;
}
