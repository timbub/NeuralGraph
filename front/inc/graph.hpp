#pragma once
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <variant>

namespace Graph {

    class Node;
    class Attribute;

    class Tensor {
    public:
        enum class DataType {
            UNDEFINED = 0,
            FLOAT = 1,
            UINT8 = 2,
            INT8 = 3,
            UINT16 = 4,
            INT16 = 5,
            INT32 = 6,
            INT64 = 7,
            STRING = 8,
            BOOL = 9,
            FLOAT16 = 10,
            DOUBLE = 11,
            UINT32 = 12,
            UINT64 = 13,
            COMPLEX64 = 14,
            COMPLEX128 = 15,
            BFLOAT16 = 16,
            FLOAT8E4M3FN = 17,
            FLOAT8E4M3FNUZ = 18,
            FLOAT8E5M2 = 19,
            FLOAT8E5M2FNUZ = 20,
            UINT4 = 21,
            INT4 = 22,
            FLOAT4E2M1 = 23,
            FLOAT8E8M0 = 24,
            UINT2 = 25,
            INT2 = 26,
        };
    private:
        std::string name_;
        DataType type_;
        std::vector<int64_t> dims_;

        Node* producer_ = nullptr;
        std::vector<Node*> consumers_;
    public:

        Tensor(std::string name, DataType type, std::vector<int64_t> dims)
            : name_(std::move(name)), type_(type), dims_(std::move(dims)) {}

        const std::string& get_name() const { return name_; }
        DataType get_type() const { return type_; }
        const std::vector<int64_t>& get_dims() const { return dims_; }
        const std::vector<Node*>& get_consumers() const { return consumers_; }
        const Node* get_producer() const { return producer_; }


        void set_producer(Node* node) { producer_ = node; }
        void add_consumer(Node* node) { consumers_.push_back(node); }
    };
    class Attribute {
    public:
        using Value = std::variant<int64_t,
            std::vector<int64_t>,
            float, std::vector<float>,
            std::string,
            std::vector<std::string>
            >;
        enum AttributeType {
          UNDEFINED = 0,
          FLOAT = 1,
          INT = 2,
          STRING = 3,
          FLOATS = 4,
          INTS = 5,
          STRINGS = 6
        };
    private:
        std::string   name_;
        AttributeType type_;
        Value value_;
    public:
        Attribute(std::string name, AttributeType type, Value value) : name_(name), type_(type), value_(value) {}
        std::string   get_name() { return name_; }
        AttributeType get_type() { return type_; }
        Value get_value() { return value_; }
        //TODO: getter for value_
    };

    class Node {
    private:
        std::string op_type_;
        std::string name_;
        std::vector<Tensor*> input_;
        std::vector<Tensor*> output_;
        std::unordered_map<std::string, Attribute> attributes_;
    public:
        Node(std::string op_type, std::string name)
            : op_type_(std::move(op_type)), name_(std::move(name)) {}

        const std::string& get_name() const { return name_; }
        const std::string& get_op_type() const { return op_type_; }
        const std::vector<Tensor*>& get_inputs() const { return input_; }
        const std::vector<Tensor*>& get_outputs() const { return output_; }

        void add_input(Tensor* t) {
            input_.push_back(t);
            t->add_consumer(this);
        }
        void add_output(Tensor* t) {
            output_.push_back(t);
            t->set_producer(this);
        }

        void add_attribute(Attribute&& atr) {
            attributes_.insert({atr.get_name(), std::move(atr)});
        }
    };

    class Graph {
    private:
        std::vector<std::unique_ptr<Tensor>> tensors_;
        std::vector<std::unique_ptr<Node>> nodes_;
    public:
        void add_tensor(std::unique_ptr<Tensor> t) { tensors_.push_back(std::move(t)); }
        void add_node(std::unique_ptr<Node> n) { nodes_.push_back(std::move(n)); }

        const std::vector<std::unique_ptr<Tensor>>& get_tensors() const { return tensors_; }
        const std::vector<std::unique_ptr<Node>>& get_nodes() const { return nodes_; }
    };

}
