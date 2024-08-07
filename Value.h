#ifndef VALUE_H
#define VALUE_H

#include <iostream>
#include <memory>
#include <set>
#include <functional>
#include <cassert>
#include <cmath>

class Value : public std::enable_shared_from_this<Value> {
public:
    double data;
    double grad;
    std::function<void()> _backward;
    std::set<std::shared_ptr<Value>> _prev;
    std::string _op;

    Value(double data, std::set<std::shared_ptr<Value>> _children = {}, std::string _op = "")
        : data(data), grad(0), _backward([]{}), _prev(_children), _op(_op) {}

    std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& other) {
        auto out = std::make_shared<Value>(data + other->data, std::set<std::shared_ptr<Value>>{shared_from_this(), other}, "+");
        out->_backward = [this, other, out] {
            this->grad += out->grad;
            other->grad += out->grad;
        };
        return out;
    }

    std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& other) {
        auto out = std::make_shared<Value>(data * other->data, std::set<std::shared_ptr<Value>>{shared_from_this(), other}, "*");
        out->_backward = [this, other, out] {
            this->grad += other->data * out->grad;
            other->grad += this->data * out->grad;
        };
        return out;
    }

    std::shared_ptr<Value> pow(double exponent) {
        assert(exponent == static_cast<int>(exponent) || exponent == static_cast<float>(exponent) || exponent == static_cast<double>(exponent));
        auto out = std::make_shared<Value>(std::pow(data, exponent), std::set<std::shared_ptr<Value>>{shared_from_this()}, "**" + std::to_string(exponent));
        out->_backward = [this, exponent, out] {
            this->grad += (exponent * std::pow(data, exponent - 1)) * out->grad;
        };
        return out;
    }

    std::shared_ptr<Value> relu() {
        auto out = std::make_shared<Value>(data < 0 ? 0 : data, std::set<std::shared_ptr<Value>>{shared_from_this()}, "ReLU");
        out->_backward = [this, out] {
            this->grad += (out->data > 0) * out->grad;
        };
        return out;
    }

    void backward() {
        std::vector<std::shared_ptr<Value>> topo;
        std::set<std::shared_ptr<Value>> visited;
        auto build_topo = [&](auto&& self, const std::shared_ptr<Value>& v) -> void {
            if (visited.find(v) == visited.end()) {
                visited.insert(v);
                for (const auto& child : v->_prev) {
                    self(self, child);
                }
                topo.push_back(v);
            }
        };
        build_topo(build_topo, shared_from_this());
        grad = 1;
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            (*it)->_backward();
        }
    }

    std::shared_ptr<Value> operator-() {
        return (*this) * std::make_shared<Value>(-1);
    }

    std::shared_ptr<Value> operator+(double other) {
        return (*this) + std::make_shared<Value>(other);
    }

    std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& other) {
        return (*this) + (-other);
    }

    std::shared_ptr<Value> operator-(double other) {
        return (*this) + std::make_shared<Value>(-other);
    }

    std::shared_ptr<Value> operator*(double other) {
        return (*this) * std::make_shared<Value>(other);
    }

    std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& other) {
        return (*this) * other->pow(-1);
    }

    std::shared_ptr<Value> operator/(double other) {
        return (*this) * std::make_shared<Value>(1.0 / other);
    }

    // Overload the operators for shared_ptr<Value>
    friend std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& l) {
        return l->operator-();
    }

    friend std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs) {
        return lhs->operator+(rhs);
    }

    friend std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs) {
        return lhs->operator-(rhs);
    }

    friend std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs) {
        return lhs->operator*(rhs);
    }

    friend std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs) {
        return lhs->operator/(rhs);
    }

    friend std::shared_ptr<Value> operator+(double lhs, const std::shared_ptr<Value>& rhs) {
        return std::make_shared<Value>(lhs) + rhs;
    }

    friend std::shared_ptr<Value> operator-(double lhs, const std::shared_ptr<Value>& rhs) {
        return std::make_shared<Value>(lhs) - rhs;
    }

    friend std::shared_ptr<Value> operator*(double lhs, const std::shared_ptr<Value>& rhs) {
        return std::make_shared<Value>(lhs) * rhs;
    }

    friend std::shared_ptr<Value> operator/(double lhs, const std::shared_ptr<Value>& rhs) {
        return std::make_shared<Value>(lhs) / rhs;
    }

    friend std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& lhs, double rhs) {
        return lhs + std::make_shared<Value>(rhs);
    }

    friend std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& lhs, double rhs) {
        return lhs - std::make_shared<Value>(rhs);
    }

    friend std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& lhs, double rhs) {
        return lhs * std::make_shared<Value>(rhs);
    }

    friend std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& lhs, double rhs) {
        return lhs / std::make_shared<Value>(rhs);
    }

    friend std::ostream& operator<<(std::ostream& os, const Value& v) {
        os << "Value(data=" << v.data << ", grad=" << v.grad << ")" << std::endl;
        return os;
    }

    friend std::ostream& operator<<(std::ostream& os, const std::shared_ptr<Value>& v) {
        os << "Value(data=" << v->data << ", grad=" << v->grad << ")" << std::endl;
        return os;
    }
};

#endif // VALUE_H
