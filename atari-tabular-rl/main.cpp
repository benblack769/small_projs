#include <thread>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <mutex>
#include <algorithm>
#include <random>
#include <inttypes.h>
#include <chrono>
#include <thread>
#include <algorithm>
#include "md5.h"
#include "min_ale.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

using std::cout; using std::endl;
#define rep2(x) x, x
#define rep4(x) rep2(x), rep2(x)
#define rep8(x) rep4(x), rep4(x)
#define rep16(x) rep8(x), rep8(x)
#define rep18(x) rep16(x), rep2(x)

constexpr int num_moves = 18;
struct Entry{
    uint8_t future_vals[num_moves]={rep18(255)};
    uint32_t visits=0;
    uint32_t tot_future_states=0;
    uint32_t tot_value=0;
};
std::vector<Entry> global_table;
std::mutex global_lock;


using rand_gen = std::mt19937;


int get_index(MD5Val hash){
    return hash.lower % global_table.size();
}
uint8_t future_val(volatile Entry & p,volatile Entry & c){
    uint32_t optimistic_fut_state_rate = (p.tot_future_states*2)/p.visits + 100;
    constexpr uint32_t OPTIMISTIC_COUNT = 10;
    static_assert(OPTIMISTIC_COUNT > 0, "OPTIMISTIC_COUNT must be greater than 0 for numerical stability");
    double optimistic_fut_stateval = (OPTIMISTIC_COUNT * optimistic_fut_state_rate + c.tot_future_states) / double(OPTIMISTIC_COUNT + c.visits);
    double avg_val = c.tot_value / double(c.visits);
    double log_val = log(avg_val + optimistic_fut_stateval)/log(1.5) + 100;
    return log_val;
}
void process_stack(std::vector<MD5Val> & hash_stack, std::vector<int> & reward_stack, std::vector<int> & action_stack){
    global_lock.lock();

    assert(hash_stack.size() == reward_stack.size() && hash_stack.size() == action_stack.size()+1);

    // std::vector<int> value_stack(reward_stack.size());
    int tot = 0;
    for(int & rew : reward_stack){
        rew *= 100000;
    }
    // for(int i = reward_stack.size()-1; i >= 0; i--){
    //     tot += reward_stack[i] + 1;
    //     value_stack[i] = tot;
    // }
    volatile Entry & final = global_table[get_index(hash_stack.back())];
    final.visits += 1;
    final.tot_future_states += 0;
    final.tot_value += reward_stack.back();
    for(int i = int(action_stack.size())-1; i >= 0 ; i--){
        int trans = action_stack[i];
        volatile Entry & parent = global_table[get_index(hash_stack[i])];
        volatile Entry & child = global_table[get_index(hash_stack[i+1])];
        assert (child.visits > 0);
        parent.visits += 1;
        parent.tot_value += child.visits ? child.tot_value/child.visits + reward_stack[i] : 0;
        parent.future_vals[trans] = future_val(parent, child);
        parent.tot_future_states += action_stack.size()-i;
    }

    global_lock.unlock();
}
int get_action(MD5Val cur_hash, rand_gen & gen){
    global_lock.lock();
    Entry e = global_table[get_index(cur_hash)];
    global_lock.unlock();
    uint8_t max_val = *std::max_element(e.future_vals,e.future_vals+num_moves);
    std::vector<int> equal_vals;
    for(size_t i = 0; i < num_moves; i++){
        if (max_val == e.future_vals[i]){
            equal_vals.push_back(i);
        }
    }
    if (equal_vals.size() == 1){
        return equal_vals.front();
    }
    else{
        std::uniform_int_distribution<unsigned> distrib(0, equal_vals.size()-1);
        return equal_vals[distrib(gen)];
    }
}
void eval_loop(std::string rom_path, int max_steps, int index, int * is_done){
    MinALE ale_example;
    rand_gen gen(index);
    ale_example.setLoggerMode(2);
    ale_example.loadROM(rom_path);
    ale_example.reset_game();
    int total_collisions = 0;
    int num_episodes = 0;
    int start_lives = ale_example.lives();
    std::vector<uint8_t> buffer(ale_example.getScreenWidth()*ale_example.getScreenHeight()*3);
    std::vector<MD5Val> hash_stack;
    std::vector<int> reward_stack;
    std::vector<int> action_stack;
    int reward = 0;
    for(int i = 0; i < max_steps; i++){
        ale_example.getScreenGrayscale(buffer.data());
        MD5Val hash = md5(buffer.data(),buffer.size());
        total_collisions += global_table[get_index(hash)].visits > 0;
        hash_stack.push_back(hash);
        reward_stack.push_back(reward);
        if (ale_example.game_over() || ale_example.lives() != start_lives) {
            ale_example.reset_game();
            process_stack(hash_stack, reward_stack, action_stack);
            reward = 0;
            hash_stack.clear();
            reward_stack.clear();
            action_stack.clear();
            num_episodes += 1;
            // std::cout << "reset\n";
        }
        else{
            int action = get_action(hash,gen);
            reward = ale_example.act(action);
            if(reward){
                std::cout << reward << std::endl;
            }
            action_stack.push_back(action);
        }
        // if(i % 10000 == 0){
        //     std::cout << total_collisions / double(i+1) << std::endl;
        // }
    }
    std::cout << num_episodes << std::endl;
    *is_done = true;
}

int run_main(std::string rom_path, size_t table_size, size_t num_threads, size_t max_steps) {
    global_table.clear();
    global_table.resize(table_size);
    std::vector<std::thread> threads;
    std::vector<int> are_done(num_threads);
    for(int i = 0; i < num_threads; i++){
        threads.emplace_back(eval_loop, rom_path, max_steps, i, &are_done[i]);
        //eval_loop(rom_path,0);
    }
    while(!std::all_of(are_done.begin(), are_done.end(), [](int x){return x;})) {
        global_lock.lock();
        long long num_visits = 0;
        long long tot_value = 0;
        long long histogram[256] = {0};
        for(volatile Entry & e : global_table){
            num_visits += e.visits;
            tot_value += e.tot_value;
            for(int i = 0; i < num_moves; i++){
                histogram[e.future_vals[i]]++;
            }
        }
        global_lock.unlock();
        std::cerr << "visits:" << num_visits << "\n";
        std::cerr << "value:" << tot_value << "\n";
        std::cerr << "histogram:\n";
        for(int i = 0; i < 256; i++){
            if(histogram[i] != 0){
                std::cerr << i << ": " << histogram[i] << "\n";
            }
        }
        std::cerr << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(5000));
        if (PyErr_CheckSignals() != 0)
            throw py::error_already_set();
    }
    for(int i = 0; i < num_threads; i++){
        threads[i].join();
    }

    return 0;
}

PYBIND11_MODULE(example, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("run_main", &run_main);
}
