#include <cassert>
#include <string>
#include <iostream>
#include <cstring>
#include <bit>
#include <mutex>
#include <algorithm>

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
int64_t gen_aug(int wordlen, char * word, frac * aug){
    for (int i = 0; i < wordlen; i++){
        aug[i] = frac{word[i] & 0x1f, 1};
    }
    for(int j = 0; j < wordlen; j++){
        aug[j+wordlen] = frac{j == 0, 1};
    }
    for (int i = 1; i < wordlen; i++){
        for(int j = 0; j < wordlen; j++){
            int size = 1+(rand() % 26);
            aug[i*wordlen*2+j] = (frac{int64_t(rand() % size) - int64_t(size / 2), 1});
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
            int c = int(std::countl_zero((uint64_t)(abs(aug[i*wordlen*2+j+wordlen].denom))));
            score += c;
        }
    }
    return score;
    // print_mat(aug, wordlen);
    // print_mat(res, wordlen);
    // print_mat(inv, wordlen);
    // break;

}

int main(int argc, char ** argv){
    assert(argc == 2);
    srand((unsigned)time(0));
    const char * worddata = (argv[1]);
    char word[1000] = {1,0+64};
    strcpy(word+1, worddata);
    const size_t wordlen = strlen(word);
    // frac * matrix = new frac[wordlen*wordlen];
    frac * best_aug = new frac[wordlen*wordlen*2];
    frac * aug = new frac[wordlen*wordlen*2];
    
    int64_t best_score = -(1LL<<62);
    for(int l = 0; l < 1000000; l++){
        int64_t score = gen_aug(wordlen, word, aug);
        if(score > best_score){
            best_score = score;
            frac * t = aug;
            aug = best_aug;
            best_aug = t;
        }
    }
    std::cout << "Score: " << best_score << "\n";
    print_mat(best_aug, wordlen);
    return 0;
}