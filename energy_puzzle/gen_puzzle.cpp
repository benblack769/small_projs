
int main(){
  populate();
  using namespace std;
  // Field I = letters['I'];
  // print_plot(plot(I));
  std::string word = "BAIT";
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
