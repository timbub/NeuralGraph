#pragma once
#include "graph.hpp"
#include "onnx.pb.h"
#include <unordered_map>

void process_value_info(const google::protobuf::RepeatedPtrField<onnx::ValueInfoProto>& info_list,
                        Graph::Graph& graph,
                        std::unordered_map<std::string, Graph::Tensor*>& map_tensor);

void process_attributes(const ::onnx::NodeProto& proto_node, Graph::Node* node);

void parse_graph(Graph::Graph& graph, const onnx::GraphProto& onnx_graph);
