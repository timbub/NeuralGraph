#include "parser.hpp"
#include <unordered_map>
#include <iostream>

void parse_graph(Graph::Graph& graph, const onnx::GraphProto& onnx_graph) {
    std::unordered_map<std::string, Graph::Tensor*> map_tensor;

    for (const auto& proto_tensor : onnx_graph.initializer()) {
        std::vector<int64_t> dims_vec(proto_tensor.dims().begin(), proto_tensor.dims().end());
        auto tensor = std::make_unique<Graph::Tensor>(
            proto_tensor.name(),
            static_cast<Graph::Tensor::DataType>(proto_tensor.data_type()),
            dims_vec
        );
        map_tensor[tensor->get_name()] = tensor.get();
        graph.add_tensor(std::move(tensor));
    }

    process_value_info(onnx_graph.input(), graph, map_tensor);
    process_value_info(onnx_graph.output(), graph, map_tensor);
    process_value_info(onnx_graph.value_info(), graph, map_tensor);

    for (const auto& proto_node : onnx_graph.node()) {
        auto node = std::make_unique<Graph::Node>(proto_node.op_type(), proto_node.name());

        for (const auto& input_name : proto_node.input()) {
            if (input_name.empty()) continue;

            if (map_tensor.find(input_name) == map_tensor.end()) {
                auto t = std::make_unique<Graph::Tensor>(input_name, Graph::Tensor::DataType::UNDEFINED, std::vector<int64_t>{});
                map_tensor[input_name] = t.get();
                graph.add_tensor(std::move(t));
            }
            Graph::Tensor* t = map_tensor[input_name];
            node->add_input(t);
        }

        for (const auto& output_name : proto_node.output()) {
            if (output_name.empty()) continue;

            if (map_tensor.find(output_name) == map_tensor.end()) {
                auto t = std::make_unique<Graph::Tensor>(output_name, Graph::Tensor::DataType::UNDEFINED, std::vector<int64_t>{});
                map_tensor[output_name] = t.get();
                graph.add_tensor(std::move(t));
            }
            Graph::Tensor* t = map_tensor[output_name];
            node->add_output(t);
        }
        graph.add_node(std::move(node));
    }
}

void process_value_info(const google::protobuf::RepeatedPtrField<onnx::ValueInfoProto>& info_list,
                        Graph::Graph& graph,
                        std::unordered_map<std::string, Graph::Tensor*>& map_tensor) {
    for (const auto& info : info_list) {
        if (map_tensor.count(info.name())) continue;

        std::vector<int64_t> dims;
        auto type = Graph::Tensor::DataType::UNDEFINED;

        if (info.has_type() && info.type().has_tensor_type()) {
            const auto& tensor_type = info.type().tensor_type();
            type = static_cast<Graph::Tensor::DataType>(tensor_type.elem_type());

            if (tensor_type.has_shape()) {
                for (const auto& dim : tensor_type.shape().dim()) {
                    dims.push_back(dim.has_dim_value() ? dim.dim_value() : -1);
                }
            }
        }
        auto tensor = std::make_unique<Graph::Tensor>(info.name(), type, dims);
        map_tensor[tensor->get_name()] = tensor.get();
        graph.add_tensor(std::move(tensor));
    }
}
