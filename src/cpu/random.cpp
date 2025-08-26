#include "random.h"
#include <random>
#include <vector>

std::vector<int> random_channel() {
    constexpr int seed = 42;
    constexpr int n_channel = 8;
    static thread_local std::mt19937 gen(seed);
    static thread_local std::uniform_int_distribution<> distr(0, 7);
    int id_generated = distr(gen); 

    std::vector<int> channel(n_channel);
    channel[id_generated] = 1;

    return channel;
}





