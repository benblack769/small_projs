#pragma once
#include <array>
#include <vector>
#include <iostream>
#include <unordered_map>

struct FieldPair{
  size_t i;
  double weight;
};
struct OpTuple{
  char c1;
  char c2;
  double weight;
};
using OpList = std::vector<OpTuple>;
using EnergyList = std::array<std::vector<FieldPair>,128>;
struct Field{
  std::string sequence;
  EnergyList energy_mat;
  OpList ops;
};
inline Field make_energy_mat(std::string s, OpList opp_energies){
  EnergyList res;
  for(size_t j = 0; j < opp_energies.size(); j++){
    OpTuple t = opp_energies[j];
    for(size_t i = 0; i < s.size(); i++){
    //  std::cout << t.weight << "\n";
      if (t.c2 == s[i])
        res[t.c1].push_back(FieldPair{i,t.weight});
    }
  }
  return Field{s,res, opp_energies};
}

std::unordered_map<char, Field> letters;
