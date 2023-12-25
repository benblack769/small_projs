#include "fields.h"
#include "populate.h"
#include <cassert>
#include <iostream>
#include <algorithm> 
#include <string>  

int main(int argc, char ** argv){
  assert(argc == 2);
  const char * word_cstr = argv[1];
  populate();
  using namespace std;
  // Field I = letters['I'];
  // print_plot(plot(I));
  std::string word(word_cstr);
  transform(word.begin(), word.end(), word.begin(), ::toupper);
  for(char c : word){
    Field f = letters[c];
    char letmap[256];
    for(char c = 'A'; c <= 'Z'; c++){
      letmap[c] = c;
    }
    for(char c = 'A'; c <= 'Z'; c++){
      letmap[c] = letmap[rand()%26 + 'A'];
    }
    for(char c : f.sequence)
      cout << letmap[c];
    cout << '\n';
    for(OpTuple o : f.ops){
      cout << letmap[o.c1] << "," << letmap[o.c2] << ',' << o.weight << "\n";
    }
  }
}
