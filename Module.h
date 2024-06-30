#ifndef MODULE_H
#define MODULE_H

#include <vector>
#include <variant>
#include <memory>
#include <random>
#include <iostream>
#include "Value.h"

class Module {
public:
    virtual void zero_grad() {
        for (auto& p : parameters()) {
            p->grad = 0;
        }
    }

    virtual std::vector<std::shared_ptr<Value>> parameters() {
        return {};
    }
};

class Neuron : public Module {
public:
    std::vector<std::shared_ptr<Value>> w;
    std::shared_ptr<Value> b;
    bool nonlin;

    Neuron(int nin, bool nonlin = true) 
        : b(std::make_shared<Value>(0)), nonlin(nonlin) {
        std::uniform_real_distribution<double> unif(-1, 1);
        std::default_random_engine re;
        for (int i = 0; i < nin; ++i) {
            w.push_back(std::make_shared<Value>(unif(re)));
        }
    }

    std::shared_ptr<Value> operator()(const std::vector<std::shared_ptr<Value>>& x) {
        auto act = b;
        for (size_t i = 0; i < w.size(); ++i) {
            act = act + (w[i] * x[i]);
        }
        return nonlin ? act->relu() : act;
    }

    std::vector<std::shared_ptr<Value>> parameters() override {
        auto params = w;
        params.push_back(b);
        return params;
    }

    friend std::ostream& operator<<(std::ostream& os, const Neuron& neuron) {
        os << (neuron.nonlin ? "ReLU" : "Linear") << "Neuron(" << neuron.w.size() << ")";
        return os;
    }
};


class Layer : public Module {
public:
    std::vector<std::shared_ptr<Neuron>> neurons;

    Layer(int nin, int nout, bool nonlin = true) {
        for (int i = 0; i < nout; ++i) {
            neurons.push_back(std::make_shared<Neuron>(nin, nonlin));
        }
    }

    std::vector<std::shared_ptr<Value>> operator()(const std::vector<std::shared_ptr<Value>>& x) {
        std::vector<std::shared_ptr<Value>> out;
        for (auto& neuron : neurons) {
            out.push_back((*neuron)(x));
        }
        return out;
    }

    std::vector<std::shared_ptr<Value>> parameters() override {
        std::vector<std::shared_ptr<Value>> params;
        for (auto& neuron : neurons) {
            auto neuron_params = neuron->parameters();
            params.insert(params.end(), neuron_params.begin(), neuron_params.end());
        }
        return params;
    }

    friend std::ostream& operator<<(std::ostream& os, const Layer& layer) {
        os << "Layer of [";
        for (size_t i = 0; i < layer.neurons.size(); ++i) {
            os << *(layer.neurons[i]);
            if (i != layer.neurons.size() - 1) os << ", ";
        }
        os << "]";
        return os;
    }
};


class MLP : public Module {
public:
    std::vector<std::shared_ptr<Layer>> layers;

    MLP(int nin, const std::vector<int>& nouts) {
        int sz = nin;
        for (size_t i = 0; i < nouts.size(); ++i) {
            layers.push_back(std::make_shared<Layer>(sz, nouts[i], i != nouts.size() - 1));
            sz = nouts[i];
        }
    }

    std::vector<std::shared_ptr<Value>> operator()(const std::vector<std::shared_ptr<Value>>& x) {
        auto output = x;
        for (auto& layer : layers) {
            output = (*layer)(output);
        }
        return output;
    }

    std::vector<std::shared_ptr<Value>> parameters() override {
        std::vector<std::shared_ptr<Value>> params;
        for (auto& layer : layers) {
            auto layer_params = layer->parameters();
            params.insert(params.end(), layer_params.begin(), layer_params.end());
        }
        return params;
    }

    friend std::ostream& operator<<(std::ostream& os, const MLP& mlp) {
        os << "MLP of [";
        for (size_t i = 0; i < mlp.layers.size(); ++i) {
            os << *(mlp.layers[i]);
            if (i != mlp.layers.size() - 1) os << ", ";
        }
        os << "]";
        return os;
    }
};

#endif // MODULE_H
