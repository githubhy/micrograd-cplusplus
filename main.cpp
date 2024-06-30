#include "Module.h"
#include <iostream>

int main() {
    MLP mlp(3, {4, 4, 1});

    std::vector<std::shared_ptr<Value>> x = {
        std::make_shared<Value>(1.0),
        std::make_shared<Value>(-2.0),
        std::make_shared<Value>(3.0)
    };

    auto output = mlp(x);
    output[0]->backward();

    std::cout << "Output: ";
    for (const auto& val : output) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    mlp.zero_grad();
    std::cout << "After zero_grad():" << std::endl;
    for (const auto& param : mlp.parameters()) {
        std::cout << param << " ";
    }
    std::cout << std::endl;

    return 0;
}