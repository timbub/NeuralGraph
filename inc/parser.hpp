#pragma once
#include "graph.hpp"
#include "onnx.pb.h"
#include <unordered_map>

void process_value_info(const google::protobuf::RepeatedPtrField<onnx::ValueInfoProto>& info_list,
                        Graph::Graph& graph,
                        std::unordered_map<std::string, Graph::Tensor*>& map_tensor);

void processing_tensors(const onnx::GraphProto& onnx_graph,
                        Graph::Graph& graph,
                        std::unordered_map<std::string, Graph::Tensor*>& map_tensor);

void processing_nodes(const onnx::GraphProto& onnx_graph,
                      Graph::Graph& graph,
                      std::unordered_map<std::string, Graph::Tensor*>& map_tensor);

void parse_graph(Graph::Graph& graph, const onnx::GraphProto& onnx_graph);
