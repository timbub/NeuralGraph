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
        process_attributes(proto_node, node.get());
        graph.add_node(std::move(node));
    }
}

void process_attributes(const ::onnx::NodeProto& proto_node, Graph::Node* node) {
    for(auto& proto_atr: proto_node.attribute()) {
    //TODO: create atr after processing< сначала получит value
        Graph::Attribute atr(proto_atr.name(), static_cast<Graph::Attribute::AttributeType>(proto_atr.type()));
        switch(atr.get_type()) {
        case Graph::Attribute::AttributeType::INT: {
            atr.value_ = proto_atr.i();
            break;
        }
        case Graph::Attribute::AttributeType::INTS: {
            std::vector<int> value(proto_atr.ints().begin(), proto_atr.ints().end());
            atr.value_ = std::move(value);
            break;
        }
        case Graph::Attribute::AttributeType::FLOAT: {
            atr.value_ = proto_atr.f();
            break;
        }
        case Graph::Attribute::AttributeType::FLOATS: {
            std::vector<float> value(proto_atr.floats().begin(), proto_atr.floats().end());
            atr.value_ = std::move(value);
            break;
        }
        case Graph::Attribute::AttributeType::STRING: {
            atr.value_ = proto_atr.s();
            break;
        }
        case Graph::Attribute::AttributeType::STRINGS: {
            std::vector<std::string> value(proto_atr.strings().begin(), proto_atr.strings().end());
            atr.value_ = std::move(value);
            break;
        }
        default: {
            break;
        }
        }
        node->add_attribute(std::move(atr));
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
