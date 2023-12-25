#include <cassert>
#include <string>
#include <iostream>
#include <cstring>
#include <bit>
#include <mutex>
#include <random>
#include <algorithm>
#include <thread>

constexpr int PRIME_MAX = 50;

struct frac{
    int64_t num;
    int64_t denom;
    frac operator - (frac other){
        return frac{num * other.denom - other.num * denom, denom * other.denom};
    }

    frac operator * (frac other){
        return frac{num * other.num, denom * other.denom};
    } 
    frac operator / (frac other){
        frac d = (*this) * other;
        assert(d.num != 0);
        return frac{d.denom, d.num};
    }
    bool simplify(){
        int primes[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53};
        int num_primes = sizeof(primes) / sizeof(primes[0]);
        frac & f = *this;
        if(f.denom < 0 && f.num < 0){
            f.denom = -f.denom;
            f.num = -f.num;
        }
        if(f.denom == f.num){
            f.denom = 1;
            f.num = 1;
        }
        if(f.denom == 1){
            return false;
        }
        if(f.num == 0){
            f.denom = 1;
            return false;
        }
        for(int i = 0; i < num_primes; i++){
            int p = primes[i];
            while(abs(f.num) % p == 0 && abs(f.denom) % p == 0){
                f.num /= p;
                f.denom /= p;
            }
        }
        return (f.num > (1LL<<32) || f.denom > (1LL<<32));
    }
};
void print_mat(frac * data, size_t dims){
    for(size_t i = 0; i < dims; i++){
        for (size_t j = 0; j < dims*2; j++){
            frac f = data[i*dims*2+j];
            std::cout<<(f.num) << "/" <<  (f.denom) << "\t";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

int64_t gen_aug(std::mt19937 & rng, int wordlen, const char * word, frac * aug){
    for (int i = 0; i < wordlen; i++){
        aug[i] = frac{word[i] & 0x1f, 1};
    }
    for(int j = 0; j < wordlen; j++){
        aug[j+wordlen] = frac{j == 0, 1};
    }
    for (int i = 1; i < wordlen; i++){
        for(int j = 0; j < wordlen; j++){
            int size = 1+(rng() % 26);
            aug[i*wordlen*2+j] = (frac{int64_t(rng() % size) - int64_t(size / 2), 1});
        }
        for(int j = 0; j < wordlen; j++){
            aug[i*wordlen*2+j+wordlen] = frac{int64_t(i == j), 1};
        }
    }
    // print_mat(aug, wordlen);
    for(int i = 0; i < wordlen; i++){
        // print_mat(aug, wordlen);
        frac & mul = aug[i*wordlen*2+i];
        if(mul.num == 0){
            return -(1LL<<62);
        }
        assert(mul.num != 0);
        for(int j = i+1; j < wordlen*2; j++){
            frac & cur = aug[i*wordlen*2+j];
            if(cur.num != 0){
                cur = cur / mul;
                if(cur.simplify()){
                    return -(1LL<<62);
                }
            }
        }
        mul = frac{1,1};
        for(int k = 0; k < wordlen; k++){
            if(k == i){
                continue;
            }
            frac subv = aug[k*wordlen*2+i];
            if (subv.num == 0){
                continue;
            }
            for(int j = 0; j < wordlen*2; j++){ 
                frac & cur = aug[k*wordlen*2+j];
                frac p = subv * aug[i*wordlen*2+j];
                if(p.simplify()){
                    return -(1LL<<62);
                }
                cur = cur - p;
                if(cur.simplify()){
                    return -(1LL<<62);
                }
            }
        }
    }
    int64_t score = 0;
    for (int i = 0; i < wordlen; i++){
        for(int j = 0; j < wordlen; j++){
            int c1 = int(std::countl_zero((uint64_t)(abs(aug[i*wordlen*2+j+wordlen].denom))));
            int c2 = int(std::countl_zero((uint64_t)(abs(aug[i*wordlen*2+j+wordlen].num))));
            score += c1 + c2;
        }
    }
    return score;
    // print_mat(aug, wordlen);
    // print_mat(res, wordlen);
    // print_mat(inv, wordlen);
    // break;

}

void best_selector(size_t iters, size_t thread_rand_seed, size_t wordlen, const char * word, frac ** best_aug_global, int64_t * best_score_global){
    std::mt19937 rng(thread_rand_seed);

    frac * aug = new frac[wordlen*wordlen*2];
    frac * best_aug = new frac[wordlen*wordlen*2];

    int64_t best_score = -(1LL<<62);

    for(int l = 0; l < iters; l++){
        int64_t score = gen_aug(rng, wordlen, word, aug);
    // std::cout << score << "\n";
        if(score > best_score){
            best_score = score;
            // std::swap(aug, best_aug);
            frac * t = aug;
            aug = best_aug;
            best_aug = t;
        }
    }
    *best_score_global = best_score;
    *best_aug_global = best_aug;
    // print_mat(aug, wordlen);
}

int main(int argc, char ** argv){
    assert(argc == 3);
    const char * iters_ch = (argv[1]);
    const char * worddata = (argv[2]);
    int iters = atoi(iters_ch);
    char word[1000] = {1,0+64};
    strcpy(word+1, worddata);
    const size_t wordlen = strlen(word);
    // frac * matrix = new frac[wordlen*wordlen];
    constexpr size_t N_THREADS = 8;
    frac * best_augs[N_THREADS];
    int64_t best_scores[N_THREADS];
    std::cout << iters << "\n";
    
    std::random_device dev;
    std::vector<std::thread> threads;
    for(int i = 0; i < N_THREADS; i++){
        size_t thread_rand_seed = dev();
        // void best_selector(size_t thread_rand_seed, size_t wordlen, char * word, frac *& best_aug_global, int64_t & best_score_global){

        threads.push_back(std::thread(best_selector, iters, thread_rand_seed, wordlen, word, &best_augs[i], &best_scores[i]));
    }
    for(auto & t : threads){
        t.join();
    }
    frac * best_aug = nullptr;
    int64_t best_score = -(1LL<<62);
    for(int i = 0; i < N_THREADS; i++){
        if (best_scores[i] > best_score){
            best_aug = best_augs[i];
            best_score = best_scores[i];
        }
    }

    std::cout << "Score: " << best_score << "\n";
    print_mat(best_aug, wordlen);
    return 0;
}