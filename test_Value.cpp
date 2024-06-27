#include "Value.h"

void test_sanity_check() {
    auto x = std::make_shared<Value>(-4.0);
    auto z = 2 * x + 2 + x;
    auto q = z->relu() + z * x;
    auto h = z->pow(2)->relu();
    auto y = h + q + q * x;
    y->backward();
    auto xmg = x, ymg = y;

    std::cout << x << z << q << h << y << std::endl;
    assert(ymg->data == -34.0);
    assert(xmg->grad == -46.0);
}

void test_more_ops() {
    auto a = std::make_shared<Value>(-4.0);
    auto b = std::make_shared<Value>(2.0);
    auto c = a + b;
    auto d = a * b + b->pow(3);
    c = c + c + 1;
    c = c + 1 + c + (-a);
    d = d + d * 2 + (b + a)->relu();
    d = d + 3 * d + (b - a)->relu();
    auto e = c - d;
    auto f = e->pow(2);
    auto g = f / 2.0;
    g = g + 10.0 / f;
    g->backward();
    auto amg = a, bmg = b, gmg = g;

    double tol = 1e-6;
    assert(std::abs(gmg->data - 24.70408163265306) < tol);
    assert(std::abs(amg->grad - 138.8338192419825) < tol);
    assert(std::abs(bmg->grad - 645.5773207547171) < tol);
}

void test_duplicate_backprop() {
    auto a = std::make_shared<Value>(1.0);
    auto b = a + 4;
    auto c = (b * 3) + (b * 5);
    c->backward();
    assert(a->grad == 8.0);
    assert(b->grad == 8.0);
    assert(c->grad == 1.0);
}

int main() {
    test_sanity_check();
    test_more_ops();
    test_duplicate_backprop();
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
