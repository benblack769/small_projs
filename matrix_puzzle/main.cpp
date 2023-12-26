#include <cassert>
#include <string>
#include <iostream>
#include <cstring>
#include <bit>
#include <mutex>
#include <random>
#include <algorithm>
#include <thread>

// constexpr int DIV_MAX = 1000;

struct frac{
    int64_t num;
    int64_t denom;
    frac operator - (frac other){
        return frac{num * other.denom - other.num * denom, denom * other.denom};
    }
    frac operator + (frac other){
        // operates on a copy
        other.num = -other.num;
        return (*this) - other;
    }

    frac operator * (frac other){
        return frac{num * other.num, denom * other.denom};
    } 
    frac operator / (frac other){
        frac d = (*this) * other;
        assert(d.num != 0);
        return frac{num * other.denom, denom * other.num};
    }
    // template<int p>
    // void simplify_p(){
    //     frac & f = *this;
    //     while((f.num) % p == 0 && (f.denom) % p == 0){
    //         f.num /= p;
    //         f.denom /= p;
    //     }
    // }
    bool simplify(){
        int primes[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53};
        int num_primes = sizeof(primes) / sizeof(primes[0]);
        frac & f = *this;
        if(f.denom < 0){
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
            while((f.num) % p == 0 && (f.denom) % p == 0){
                f.num /= p;
                f.denom /= p;
            }
        }
        return (f.num > (1LL<<14) || f.denom > (1LL<<14));
    }
};
void print_mat(frac * data, size_t dims){
    for(size_t i = 0; i < dims; i++){
        for (size_t j = 0; j < dims; j++){
            frac f = data[i*dims+j];
            if (f.denom == 1){
                std::cout<< (f.num);
            }
            else{
                std::cout<<(f.num) << "/" <<  (f.denom);
            }
            if (j != dims-1){
                std::cout << ",";
            }
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}
void generate_matrix(frac * mat, std::mt19937 & rng, int wordlen, const char * word){
    for (int i = 0; i < wordlen; i++){
        mat[i] = frac{word[i] & 0x1f, 1};
    }
    for (int i = 1; i < wordlen; i++){
        for(int j = 0; j < wordlen; j++){
            int size = 1+(rng() % 26);
            mat[i*wordlen+j] = (frac{int64_t(rng() % size) - int64_t(size / 2), 1});
        }
    }
}
bool invert_matrix(int wordlen, frac * mat, frac * inv){
    frac aug[10000];
    for(int i = 0; i < wordlen; i++){
        for(int j = 0; j < wordlen; j++){
            aug[i*wordlen*2+j] = mat[i*wordlen+j];
            aug[i*wordlen*2+j+wordlen] = frac{int64_t(i == j), 1};
        }
    }
    // print_mat(aug, wordlen);
    for(int i = 0; i < wordlen; i++){
        // print_mat(aug, wordlen);
        frac mul = aug[i*wordlen*2+i];
        if(mul.num == 0){
            // std::cout << "bad mul num" << std::endl;
            return false;
        }
        assert(mul.num != 0);
        for(int j = 0; j < wordlen*2; j++){
            frac & cur = aug[i*wordlen*2+j];
            if(cur.num != 0){
                cur = cur / mul;
                if(cur.simplify()){
                    // std::cout << cur.num << '\t' << cur.denom << std::endl;
                    // std::cout << "simplerr1" << std::endl;
                    return false;
                }
            }
        }
        // mul = frac{1,1};
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
                    return false;
                }
                cur = cur - p;
                if(cur.simplify()){
                    return false;
                }
            }
        }
    }
    frac base[1000];
    for(int i = 0; i < wordlen; i++){
        for(int j = 0; j < wordlen; j++){
            base[i*wordlen+j] = aug[i*wordlen*2+j];
            inv[i*wordlen+j] = aug[i*wordlen*2+j+wordlen];
        }
    }
    for(int i = 0; i < wordlen; i++){
        for(int j = 0; j < wordlen; j++){
            frac tmplate = aug[i*wordlen*2+j];
            assert(((tmplate.num == int(i == j) && (tmplate.denom == 1))));
            // if(!((tmplate.num == int(i == j) && (tmplate.denom == 1)))){
            //     print_mat(base, wordlen);
            //     print_mat(inv, wordlen);
            //     return false;
            // }
        }
    }
    return true;
}

int64_t score_matrix(frac * mat, int wordlen, const char * word){
    frac inv[10000];
    bool result = invert_matrix(wordlen, mat, inv);
    if(!result){
        return -(1LL<<62);
    }
    // also check that the inverse is invertible
    frac inv2[10000];
     result = invert_matrix(wordlen, inv, inv2);
    if(!result){
        return -(1LL<<62);
    }
    // for (int i = 0; i < wordlen; i++){
    //     for(int j = 0; j < wordlen; j++){
    //         frac sum{0,1};
    //         for(int k = 0; k < wordlen; k++){
    //             sum = sum + mat[i*wordlen+k] * aug[k*wordlen*2+j+wordlen];
    //         }
    //         sum.simplify();
    //         int expected = int(i == j);
    //         if(sum.num != expected || sum.denom != 1){
    //             std::cout << sum.num << "\t" << sum.denom << "\n";
    //             return -(1LL<<62);
    //         }
    //     }
    // }
    int64_t score = 0;
    for (int i = 0; i < wordlen; i++){
        for(int j = 0; j < wordlen; j++){
            int c1 = int(std::countl_zero((uint64_t)(abs(inv[i*wordlen+j].denom))));
            int c2 = int(std::countl_zero((uint64_t)(abs(inv[i*wordlen+j].num))));
            score += c1 + c2;
        }
    }
    return score;
    // print_mat(aug, wordlen);
    // print_mat(res, wordlen);
    // print_mat(inv, wordlen);
    // break;

}

void best_selector(size_t iters, size_t thread_rand_seed, size_t wordlen, const char * word, frac ** best_mat_global, int64_t * best_score_global){
    std::mt19937 rng(thread_rand_seed);

    frac * mat = new frac[wordlen*wordlen];
    frac * best_mat = new frac[wordlen*wordlen];

    std::fill(mat, mat+wordlen*wordlen, frac{1,1});
    std::fill(best_mat, best_mat+wordlen*wordlen, frac{1,1});

    int64_t best_score = -(1LL<<62);

    for(int l = 0; l < iters; l++){
        generate_matrix(mat, rng, wordlen, word);
        int64_t score = score_matrix(mat, wordlen, word);
        // std::cout << score << "\n";
        if(score > best_score){
            best_score = score;
            std::swap(mat, best_mat);
        }
    }
    *best_score_global = best_score;
    *best_mat_global = best_mat;
    // print_mat(aug, wordlen);
}

int main(int argc, char ** argv){
    assert(argc == 3);
    const char * iters_ch = (argv[1]);
    const char * worddata = (argv[2]);
    int iters = atoi(iters_ch);
    char word[1000] = {1};
    strcpy(word+1, worddata);
    const size_t wordlen = strlen(word);
    // frac mat[1000];
    // frac inv[1000];
    // std::random_device dev;
    // std::mt19937 rng(dev());
    // do{
    // generate_matrix(mat, rng, wordlen, word);
    // }
    // while(!invert_matrix(wordlen, mat, inv));
    // std::cout << "Mat\n"; 
    // print_mat(mat, wordlen);
    
    // frac prod[10000];
    // for (int i = 0; i < wordlen; i++){
    //     for(int j = 0; j < wordlen; j++){
    //         frac sum{0,1};
    //         for(int k = 0; k < wordlen; k++){
    //             sum = sum + mat[i*wordlen+k] * inv[k*wordlen+j];
    //         }
    //         assert(!sum.simplify());
    //         prod[i*wordlen+j] = sum;
    //         // int expected = int(i == j);
    //         // if(sum.num != expected || sum.denom != 1){
    //         //     std::cout << sum.num << "\t" << sum.denom << "\n";
    //         //     return -(1LL<<62);
    //         // }
    //     }
    // }
    // std::cout << "Inv\n"; 
    // print_mat(inv, wordlen);
    // std::cout << "Prod\n"; 
    // print_mat(prod, wordlen);

    // return 0;
    frac * matrix = new frac[wordlen*wordlen];
    constexpr size_t N_THREADS = 8;
    frac * best_mats[N_THREADS];
    int64_t best_scores[N_THREADS];
    std::cout << iters << "\n";
    
    std::random_device dev;
    std::vector<std::thread> threads;
    for(int i = 0; i < N_THREADS; i++){
        size_t thread_rand_seed = dev();
        // void best_selector(size_t thread_rand_seed, size_t wordlen, char * word, frac *& best_aug_global, int64_t & best_score_global){

        threads.push_back(std::thread(best_selector, iters, thread_rand_seed, wordlen, word, &best_mats[i], &best_scores[i]));
    }
    for(auto & t : threads){
        t.join();
    }
    frac * best_mat = nullptr;
    int64_t best_score = -(1LL<<62);
    for(int i = 0; i < N_THREADS; i++){
        if (best_scores[i] > best_score){
            best_mat = best_mats[i];
            best_score = best_scores[i];
        }
    }
    frac inv[10000];
    print_mat(best_mat, wordlen);
    bool result = invert_matrix(wordlen, best_mat, inv);
    assert(result);
    frac prod[10000];
    for (int i = 0; i < wordlen; i++){
        for(int j = 0; j < wordlen; j++){
            frac sum{0,1};
            for(int k = 0; k < wordlen; k++){
                sum = sum + best_mat[i*wordlen+k] * inv[k*wordlen+j];
            }
            assert(!sum.simplify());
            prod[i*wordlen+j] = sum;
            // int expected = int(i == j);
            // if(sum.num != expected || sum.denom != 1){
            //     std::cout << sum.num << "\t" << sum.denom << "\n";
            //     return -(1LL<<62);
            // }
        }
    }
    std::cout << "Mat\n"; 
    print_mat(best_mat, wordlen);
    std::cout << "Inv\n"; 
    print_mat(inv, wordlen);
    std::cout << "Prod\n"; 
    print_mat(prod, wordlen);


    std::cout << "Score: " << best_score << "\n";
    // print_mat(best_mat, wordlen);
    return 0;
}